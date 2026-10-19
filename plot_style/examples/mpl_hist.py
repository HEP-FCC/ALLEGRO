"""Histograms: two stacked, filled backgrounds with two signal lines on top.

Run:  python examples/mpl_hist.py   ->  mpl_hist.png, mpl_hist.pdf
Same plot with ROOT: examples/root_hist.C
"""
import matplotlib.pyplot as plt
import numpy as np

from allegro_style import mpl as amp

amp.use()

fig, ax = plt.subplots()

# Example data: 40 bins of 5 GeV between 0 and 200 GeV
edges = np.linspace(0, 200, 41)
x = edges[:-1] + 2.5  # bin centres
background1 = 1900 * np.exp(-x / 50)
background2 = 700 * np.exp(-0.5 * ((x - 90) / 15) ** 2)
signal_a = 700 * np.exp(-0.5 * ((x - 125) / 5) ** 2)
signal_b = 470 * np.exp(-0.5 * ((x - 150) / 8) ** 2)

# Filled histograms: amp.fill(i) uses colour number i.
# Background 2 is stacked on top of background 1 (it starts at "baseline").
ax.stairs(background1, edges, fill=True, label="Background 1", **amp.fill(1))
ax.stairs(background1 + background2, edges, baseline=background1, fill=True,
          label="Background 2", **amp.fill(4))

# Line histograms: amp.line(i) uses colour number i and line style number i
ax.stairs(signal_a, edges, label="Signal A", **amp.line(0))  # navy, solid
ax.stairs(signal_b, edges, label="Signal B", **amp.line(3))  # red, dash-dot

ax.set_xlim(0, 200)
ax.set_ylim(0, 2700)
ax.set_xlabel(r"$m_{jj}$ [GeV]")
ax.set_ylabel("Events / 5 GeV")

amp.label(ax)
amp.info(ax, process=r"$Z\to q\bar{q}$", energy=91.2, lumi=10.8)
amp.legend(ax)

fig.savefig("mpl_hist.png")
fig.savefig("mpl_hist.pdf")
