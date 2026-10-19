"""Colour-vision-deficiency check of a palette.

    python tests/cvd.py "#3f90da" "#ffa90e" ...

prints, for normal vision and simulated protanopia / deuteranopia / tritanopia
(Machado et al. 2009, full severity), the smallest CIELAB distance (Delta E 1976)
between any two colours and between any colour and the white background.
"""
import itertools
import sys

import numpy as np

MACHADO = {
    "normal": np.eye(3),
    "protan": np.array([[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]]),
    "deutan": np.array([[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]]),
    "tritan": np.array([[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.303900]]),
}
SRGB_TO_XYZ = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
WHITE_XYZ = SRGB_TO_XYZ @ np.ones(3)


def _linear(hex_color):
    c = np.array([int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)])
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _lab(lin):
    t = (SRGB_TO_XYZ @ np.clip(lin, 0, 1)) / WHITE_XYZ
    f = np.where(t > (6 / 29) ** 3, np.cbrt(t), t / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def lab(hex_color, vision="normal"):
    return _lab(MACHADO[vision] @ _linear(hex_color))


def min_distances(colors, vision="normal"):
    """(min Delta E between two colours, min Delta E between a colour and white)."""
    labs = [lab(c, vision) for c in colors]
    white = _lab(np.ones(3))
    pairs = min((np.linalg.norm(a - b) for a, b in itertools.combinations(labs, 2)), default=np.inf)
    bg = min(np.linalg.norm(a - white) for a in labs)
    return pairs, bg


def report(colors):
    for v in MACHADO:
        p, w = min_distances(colors, v)
        print(f"{v:7s}  min pair dE = {p:5.1f}   min dE to white = {w:5.1f}")


if __name__ == "__main__":
    report(sys.argv[1:])
