"""2D colour map with a colour scale.

Run:  python examples/mpl_colz.py   ->  mpl_colz.png, mpl_colz.pdf
Same plot with ROOT: examples/root_colz.C
"""
import matplotlib.pyplot as plt
import numpy as np

from allegro_style import mpl as amp

amp.use()

fig, ax = plt.subplots()

# Example data: 40 x 40 bins between -3 and 3, a tilted 2D Gaussian
edges = np.linspace(-3, 3, 41)
x, y = np.meshgrid(edges[:-1] + 0.075, edges[:-1] + 0.075)  # bin centres
z = 1000 * np.exp(-0.5 * (x**2 + (y - 0.5 * x) ** 2 / 0.5))

mesh = ax.pcolormesh(edges, edges, z, vmin=0)  # the colour map comes from the style
amp.colorbar(mesh, ax, label="Events")         # call before amp.label / amp.info

ax.set_xlabel(r"$\theta$ [rad]")
ax.set_ylabel(r"$\phi$ [rad]")

amp.label(ax)
# The frame is full, so put the info on the label's line, at the right
amp.info(ax, energy=91.2, x=1, y=1.012, align="right", valign="bottom")

fig.savefig("mpl_colz.png")
fig.savefig("mpl_colz.pdf")
