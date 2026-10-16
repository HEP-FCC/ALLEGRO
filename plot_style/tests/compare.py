"""Compare a matplotlib and a ROOT rendering of the same plot.

    python tests/compare.py mpl.png root.png out_prefix
writes out_prefix_side_by_side.png and out_prefix_overlay.png (ROOT in red, matplotlib in blue).
"""
import sys

import matplotlib.image as mpimg
import numpy as np


def gray(path):
    img = mpimg.imread(path)[..., :3]
    return img.mean(axis=2)


def frame_box(img, dark=0.5, min_fill=0.6):
    """(left, right, bottom, top) pixel positions of the long dark lines of the axes frame."""
    d = img < dark
    cols = np.where(d.mean(axis=0) > min_fill * d.mean(axis=0).max())[0]
    rows = np.where(d.mean(axis=1) > min_fill * d.mean(axis=1).max())[0]
    return cols.min(), cols.max(), rows.max(), rows.min()


def write_comparison(mpl_png, root_png, prefix):
    a, b = gray(mpl_png), gray(root_png)
    mpimg.imsave(f"{prefix}_side_by_side.png", np.hstack([a, np.ones((a.shape[0], 10)), b]), cmap="gray")
    # white where both are white; red = only ROOT ink, blue = only matplotlib ink, dark = both
    overlay = np.stack([a, np.minimum(a, b), b], axis=2)
    mpimg.imsave(f"{prefix}_overlay.png", overlay)
    return frame_box(a), frame_box(b)


if __name__ == "__main__":
    print(write_comparison(*sys.argv[1:4]))
