"""Same plots as example_root.C, drawn with matplotlib.

    python examples/example_mpl.py <outdir> [png|pdf]   ->  mpl_{points,hist,colz,styles}.png
"""
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from allegro_style import mpl as amp  # noqa: E402


def gaus(x, n, mu, sigma, w):
    return n * w * np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))


def points(out):
    fig, ax = plt.subplots()
    edges = np.linspace(-4, 4, 41)
    x, w = 0.5 * (edges[1:] + edges[:-1]), edges[1] - edges[0]
    peak = 0
    for i, (mu, sigma) in enumerate([(0.0, 1.0), (0.5, 0.8), (-0.5, 1.2)]):
        y = gaus(x, 5000, mu, sigma, w)
        peak = max(peak, y.max())
        ax.errorbar(x, y, yerr=np.sqrt(y), linestyle="none", label=f"Sample {i}", **amp.points(i))
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(0, 1.4 * peak)
    ax.set_xlabel(r"$E_\mathrm{reco}/E_\mathrm{true} - 1$")
    ax.set_ylabel("Events / 0.2")
    amp.label(ax)
    amp.info(ax, process=r"$Z\to q\bar{q}$", energy=91.2, lumi=10.8)
    amp.legend(ax)
    fig.savefig(out)


def hist(out):
    fig, ax = plt.subplots()
    edges = np.linspace(0, 200, 41)
    x, w = 0.5 * (edges[1:] + edges[:-1]), edges[1] - edges[0]
    bkg1 = 20000 * w * np.exp(-x / 50) / 50
    bkg2 = gaus(x, 5000, 91, 15, w)
    sig_a, sig_b = gaus(x, 2000, 125, 5, w), gaus(x, 2000, 150, 8, w)
    ax.stairs(bkg1, edges, fill=True, label="Background 1", **amp.fill(1))
    ax.stairs(bkg1 + bkg2, edges, baseline=bkg1, fill=True, label="Background 2", **amp.fill(4))
    ax.stairs(sig_a, edges, label="Signal A", **amp.line(0))
    ax.stairs(sig_b, edges, label="Signal B", **amp.line(3))
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(0, 1.4 * (bkg1 + bkg2).max())
    ax.set_xlabel(r"$m_{jj}$ [GeV]")
    ax.set_ylabel("Events / 5 GeV")
    amp.label(ax)
    amp.info(ax, process=r"$Z\to q\bar{q}$", energy=91.2, lumi=10.8)
    amp.legend(ax)
    fig.savefig(out)


def colz(out):
    fig, ax = plt.subplots()
    edges = np.linspace(-3, 3, 41)
    x = 0.5 * (edges[1:] + edges[:-1])
    X, Y = np.meshgrid(x, x, indexing="ij")
    Z = 1000 * np.exp(-0.5 * (X ** 2 + (Y - 0.5 * X) ** 2 / 0.5))
    mesh = ax.pcolormesh(edges, edges, Z.T, vmin=0, vmax=Z.max())
    amp.colorbar(mesh, ax, label="Events")
    ax.set_xlabel(r"$\theta$ [rad]")
    ax.set_ylabel(r"$\phi$ [rad]")
    amp.label(ax)
    amp.info(ax, energy=91.2, x=1, y=1.012, align="right", valign="bottom")
    fig.savefig(out)


def styles(out):
    """Reference chart: amp.line(i), amp.points(i) and amp.fill(i) for i = 0..9."""
    fig, ax = plt.subplots()
    for i in range(10):
        ax.plot([0.05, 0.55], [i, i], **amp.line(i))
        ax.plot([0.7], [i], linestyle="none", **amp.points(i))
        ax.add_patch(Rectangle((0.82, i - 0.3), 0.13, 0.6, **amp.fill(i, linewidth=0)))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 9.5)
    ax.tick_params(axis="x", which="both", labelbottom=False)
    ax.set_xlabel("line(i)            points(i)      fill(i)")
    ax.set_ylabel("i")
    amp.label(ax)
    fig.savefig(out)


def main(outdir, ext="png"):
    amp.use()
    outdir = Path(outdir)
    for make in (points, hist, colz, styles):
        make(outdir / f"mpl_{make.__name__}.{ext}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
