"""Points with error bars, with legend, ALLEGRO label and info.
Sample 0 also has x error bars (half the bin width on each side).

Run:  python examples/mpl_points.py   ->  mpl_points.png, mpl_points.pdf
Same plot with ROOT: examples/root_points.C
"""
import matplotlib.pyplot as plt
import numpy as np

from allegro_style import mpl as amp

amp.use()  # apply the ALLEGRO style, before making the figure

fig, ax = plt.subplots()

# Example data: three Gaussian-shaped datasets, 40 points between -4 and 4
x = np.linspace(-3.9, 3.9, 40)
for i, (height, mean, width) in enumerate([(400, 0.0, 1.0), (500, 0.5, 0.8), (330, -0.5, 1.2)]):
    y = height * np.exp(-0.5 * ((x - mean) / width) ** 2)
    xerr = 0.1 if i == 0 else None  # x error bars only for sample 0
    # amp.points(i): colour number i and marker number i, so every dataset looks different
    ax.errorbar(x, y, xerr=xerr, yerr=np.sqrt(y), linestyle="none", label=f"Sample {i}", **amp.points(i))

ax.set_xlim(-4, 4)
ax.set_ylim(0, 700)
ax.set_xlabel(r"$E_\mathrm{reco}/E_\mathrm{true} - 1$")
ax.set_ylabel("Events / 0.2")

amp.label(ax)                                                     # "ALLEGRO Simulation"
amp.info(ax, process=r"$Z\to q\bar{q}$", energy=91.2, lumi=10.8)  # process, sqrt(s) [GeV], lumi [ab^-1]
amp.legend(ax)

fig.savefig("mpl_points.png")
fig.savefig("mpl_points.pdf")
