"""Conv4D and Conv4DTranspose, ported from 4d-mJ-Net's Utils/layers_4D.py.

The original is TensorFlow 2.1 + Keras 2.3, which needs CUDA 10.1 and therefore cannot run
on any card we have (sm_86 and sm_89 postdate its kernels). The operator itself is not a CUDA
kernel though -- it builds a 4D convolution out of ordinary 3D convolutions that SHARE one
weight tensor, summing each 3D result into the output frame its kernel offset selects. That
ports one-for-one.

Layout: the reference works in channels_last (b, z, t, x, y, c); torch convolutions want
channels_first, so these modules take (b, c, z, t, x, y) and convert at the boundary only.

`reduce_dim` picks which of the two leading spatial-temporal axes is looped over (2 = loop t
using the z-dim inside Conv3d, 1 = the reverse), exactly as in the reference.
"""
import torch
import torch.nn as nn


class Conv4D(nn.Module):
    """4D convolution as a loop of weight-shared 3D convolutions. Input (b, c, z, t, x, y)."""

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding="same",
                 bias=True, reduce_dim=2, activation=True):
        super().__init__()
        assert len(kernel_size) == 4, "4D kernel size expected"
        assert padding == "same", "only 'same' is implemented, as in the reference"
        assert reduce_dim in (1, 2)
        self.k = tuple(kernel_size)
        self.reduce_dim = reduce_dim
        # out_axis is the axis the loop writes into; the other one stays inside Conv3d
        self.out_axis = 1 if reduce_dim == 2 else 2
        z_k, t_k, x_k, y_k = self.k
        keep_k = z_k if reduce_dim == 1 else t_k
        reduce_k = z_k if reduce_dim == 2 else t_k
        # ONE Conv3d per kernel frame i, each shared across the input frames j it touches
        # (layers_4D.py resets frame_conv3D at the top of the i loop, so the weights are
        # shared along j but NOT along i -- a single conv for all i was the first port's bug).
        self.convs = nn.ModuleList([
            nn.Conv3d(in_ch, out_ch, (keep_k, x_k, y_k), stride=stride, padding="same", bias=bias)
            for _ in range(reduce_k)
        ])
        # the reference puts elu INSIDE each Conv3D, before the frames are summed
        self.act = nn.Identity()

    def forward(self, x):
        b, c, z_i, t_i, x_i, y_i = x.shape
        reduce_k = self.k[self.out_axis - 1]
        reduce_i = z_i if self.out_axis == 1 else t_i
        reduce_o = reduce_i  # 'same'

        frames = [None] * reduce_o
        for i in range(reduce_k):
            for j in range(reduce_i):
                out_frame = j - (i - reduce_k // 2)
                if out_frame < 0 or out_frame >= reduce_o:
                    continue
                # take frame j along the reduced axis, leaving (b, c, keep, x, y) for Conv3d
                sub = x[:, :, j] if self.reduce_dim == 2 else x[:, :, :, j]
                out = torch.nn.functional.elu(self.convs[i](sub))
                frames[out_frame] = out if frames[out_frame] is None else frames[out_frame] + out

        # frames are (b, out_ch, keep, x, y); stack along the reduced axis
        out = torch.stack(frames, dim=2 if self.out_axis == 1 else 3)
        return self.act(out)


class Conv4DTranspose(Conv4D):
    """Transposed variant: identical bookkeeping, a ConvTranspose3d inside."""

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding="same",
                 bias=True, reduce_dim=2, activation=True):
        super().__init__(in_ch, out_ch, kernel_size, stride, padding, bias, reduce_dim, activation)
        z_k, t_k, x_k, y_k = self.k
        keep_k = z_k if reduce_dim == 1 else t_k
        reduce_k = z_k if reduce_dim == 2 else t_k
        s = stride if isinstance(stride, (tuple, list)) else (stride,) * 3
        # ConvTranspose3d has no padding='same'; for odd kernels this pairing reproduces it
        pad = tuple(kk // 2 for kk in (keep_k, x_k, y_k))
        outpad = tuple(ss - 1 for ss in s)
        self.convs = nn.ModuleList([
            nn.ConvTranspose3d(in_ch, out_ch, (keep_k, x_k, y_k), stride=s,
                               padding=pad, output_padding=outpad, bias=bias)
            for _ in range(reduce_k)
        ])
