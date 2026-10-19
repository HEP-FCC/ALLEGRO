"""Render the same plots with matplotlib and ROOT and check that they coincide.

Images are left in tests/output (or $ALLEGRO_STYLE_OUTPUT) for visual inspection:
{mpl,root}_{points,hist,colz,styles}.png and .pdf, plus compare_<plot>_side_by_side.png and
compare_<plot>_overlay.png (red = only ROOT, blue = only matplotlib, grey/black = both).
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")
import matplotlib.image as mpimg  # noqa: E402
from compare import frame_box, gray, write_comparison  # noqa: E402

import allegro_style  # noqa: E402

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
PLOTS = ["points", "hist", "colz", "styles"]


def _render(outdir):
    """Run every example in outdir, as a user would: each writes its .png and .pdf there."""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(EXAMPLES.parent), os.environ.get("PYTHONPATH", "")]))
    for p in PLOTS:
        subprocess.run([sys.executable, str(EXAMPLES / f"mpl_{p}.py")], cwd=outdir, env=env, check=True)
        r = subprocess.run(["root", "-l", "-b", "-q", "--web=off", str(EXAMPLES / f"root_{p}.C")],
                           cwd=outdir, capture_output=True, text=True, timeout=300)
        assert r.returncode == 0 and "error" not in r.stderr.lower(), r.stderr


@pytest.fixture(scope="module")
def outdir(output_dir):
    if not shutil.which("root"):
        pytest.skip("no root executable")
    _render(output_dir)
    for p in PLOTS:
        write_comparison(output_dir / f"mpl_{p}.png", output_dir / f"root_{p}.png", output_dir / f"compare_{p}")
    return output_dir


def _load(outdir, plot):
    return gray(outdir / f"mpl_{plot}.png"), gray(outdir / f"root_{plot}.png")


def _ink_extent(img, rows, cols):
    ys, xs = np.where(img[rows, cols] < 0.5)
    return xs.min() + cols.start, xs.max() + cols.start, ys.min() + rows.start, ys.max() + rows.start


@pytest.mark.parametrize("plot", PLOTS)
def test_same_canvas_and_frame(outdir, plot):
    a, b = _load(outdir, plot)
    canvas = allegro_style.load_config()["canvas"]
    assert a.shape == b.shape == (canvas["height_px"], canvas["width_px"])
    for pa, pb in zip(frame_box(a), frame_box(b)):
        assert abs(int(pa) - int(pb)) <= 1


@pytest.mark.parametrize("plot", PLOTS)
def test_images_close(outdir, plot):
    a = mpimg.imread(outdir / f"mpl_{plot}.png")[..., :3]
    b = mpimg.imread(outdir / f"root_{plot}.png")[..., :3]
    assert np.abs(a - b).mean() < 0.035


# (rows, columns) in pixels of an 800x600 canvas; scaled to the actual canvas size
REGIONS = {
    "experiment label": ((0, 46), (100, 400)),
    "info lines": ((60, 125), (120, 400)),
    "legend text": ((60, 200), (590, 760)),
    "y title": ((40, 250), (0, 70)),
    "x title": ((540, 600), (560, 800)),
}


def _scaled(img, rows, cols):
    h, w = img.shape[:2]
    return (slice(rows[0] * h // 600, rows[1] * h // 600), slice(cols[0] * w // 800, cols[1] * w // 800))


@pytest.mark.parametrize("plot", ["points", "hist"])
@pytest.mark.parametrize("region", list(REGIONS))
def test_text_blocks_coincide(outdir, plot, region):
    a, b = _load(outdir, plot)
    rows, cols = _scaled(a, *REGIONS[region])
    for ea, eb in zip(_ink_extent(a, rows, cols), _ink_extent(b, rows, cols)):
        assert abs(int(ea) - int(eb)) <= 6, region


def test_colour_scale_coincides(outdir):
    a = mpimg.imread(outdir / "mpl_colz.png")[..., :3]
    b = mpimg.imread(outdir / "root_colz.png")[..., :3]
    frame = _scaled(a, (60, 500), (125, 660))
    scale = _scaled(a, (60, 500), (680, 708))
    assert np.abs(a[frame] - b[frame]).mean() < 0.01
    assert np.abs(a[scale] - b[scale]).mean() < 0.02


def test_pdf_output(outdir):
    """Both libraries write valid PDFs; matplotlib embeds the TeX Gyre Heros subsets,
    ROOT references the standard (non-embedded) Helvetica, as for any ROOT PDF."""
    for p in PLOTS:
        mpl_pdf, root_pdf = (outdir / f"mpl_{p}.pdf").read_bytes(), (outdir / f"root_{p}.pdf").read_bytes()
        assert mpl_pdf.startswith(b"%PDF") and root_pdf.startswith(b"%PDF")
        assert b"TeXGyreHeros-Regular" in mpl_pdf and b"TeXGyreHeros-BoldItalic" in mpl_pdf
        assert b"/Helvetica" in root_pdf and b"/Helvetica-BoldOblique" in root_pdf
