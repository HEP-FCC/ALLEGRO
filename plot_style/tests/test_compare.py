"""Render the same plots with matplotlib and ROOT and check that they coincide.

Images are left in tests/output (or $ALLEGRO_STYLE_OUTPUT) for visual inspection:
{mpl,root}_{points,hist,colz,styles}.png and .pdf, plus compare_<plot>_side_by_side.png and
compare_<plot>_overlay.png (red = only ROOT, blue = only matplotlib, grey/black = both).
"""
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")
import matplotlib.image as mpimg  # noqa: E402
from compare import frame_box, gray, write_comparison  # noqa: E402

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
PLOTS = ["points", "hist", "colz", "styles"]


@pytest.fixture(scope="module")
def outdir(output_dir):
    if not shutil.which("root"):
        pytest.skip("no root executable")
    subprocess.run([sys.executable, str(EXAMPLES / "example_mpl.py"), str(output_dir)], check=True)
    r = subprocess.run(["root", "-l", "-b", "-q", "--web=off", f'{EXAMPLES / "example_root.C"}("{output_dir}")'],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and "error" not in r.stderr.lower(), r.stderr
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
    assert a.shape == b.shape == (600, 800)
    for pa, pb in zip(frame_box(a), frame_box(b)):
        assert abs(int(pa) - int(pb)) <= 1


@pytest.mark.parametrize("plot", PLOTS)
def test_images_close(outdir, plot):
    a = mpimg.imread(outdir / f"mpl_{plot}.png")[..., :3]
    b = mpimg.imread(outdir / f"root_{plot}.png")[..., :3]
    assert np.abs(a - b).mean() < 0.035


REGIONS = {
    "experiment label": (slice(0, 46), slice(100, 400)),
    "info lines": (slice(60, 125), slice(120, 400)),
    "legend text": (slice(60, 200), slice(590, 760)),
    "y title": (slice(40, 250), slice(0, 70)),
    "x title": (slice(540, 600), slice(560, 800)),
}


@pytest.mark.parametrize("plot", ["points", "hist"])
@pytest.mark.parametrize("region", list(REGIONS))
def test_text_blocks_coincide(outdir, plot, region):
    a, b = _load(outdir, plot)
    rows, cols = REGIONS[region]
    for ea, eb in zip(_ink_extent(a, rows, cols), _ink_extent(b, rows, cols)):
        assert abs(int(ea) - int(eb)) <= 6, region


def test_colour_scale_coincides(outdir):
    a = mpimg.imread(outdir / "mpl_colz.png")[..., :3]
    b = mpimg.imread(outdir / "root_colz.png")[..., :3]
    frame = (slice(60, 500), slice(125, 660))
    scale = (slice(60, 500), slice(680, 708))
    assert np.abs(a[frame] - b[frame]).mean() < 0.01
    assert np.abs(a[scale] - b[scale]).mean() < 0.02


def test_pdf_output(outdir):
    """Both libraries write valid PDFs; matplotlib embeds the TeX Gyre Heros subsets,
    ROOT references the standard (non-embedded) Helvetica, as for any ROOT PDF."""
    subprocess.run([sys.executable, str(EXAMPLES / "example_mpl.py"), str(outdir), "pdf"], check=True)
    r = subprocess.run(["root", "-l", "-b", "-q", "--web=off", f'{EXAMPLES / "example_root.C"}("{outdir}", "pdf")'],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and "error" not in r.stderr.lower(), r.stderr
    for p in PLOTS:
        mpl_pdf, root_pdf = (outdir / f"mpl_{p}.pdf").read_bytes(), (outdir / f"root_{p}.pdf").read_bytes()
        assert mpl_pdf.startswith(b"%PDF") and root_pdf.startswith(b"%PDF")
        assert b"TeXGyreHeros-Regular" in mpl_pdf and b"TeXGyreHeros-BoldItalic" in mpl_pdf
        assert b"/Helvetica" in root_pdf and b"/Helvetica-BoldOblique" in root_pdf
