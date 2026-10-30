"""Encoder variants tried against the plain 2D U-Net of cnn_baseline.

Each keeps that trunk unchanged -- same widths, normalisation, dropout and head -- and alters one
thing, so a comparison isolates the change rather than the capacity:

    tsnet   attention along time, per pixel, before the trunk
    cbnet   projection onto a learned gamma-variate curve basis before the trunk
    duonet  ln MTT emitted as a distribution, sampled for the symbolic decoder, after the trunk
    calnet  a calibrated mask head: learned temperature, a volume loss and a decisiveness penalty

Selected with at2d_sae.py --arch {unet,tsnet,cbnet,duonet,calnet}.
"""
