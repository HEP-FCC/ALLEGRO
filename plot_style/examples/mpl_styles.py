"""All styles: amp.line(i), amp.points(i) and amp.fill(i) for i = 0..9.

Run:  python examples/mpl_styles.py   ->  mpl_styles.png, mpl_styles.pdf
Same chart with ROOT: examples/root_styles.C
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from allegro_style import mpl as amp

amp.use()

fig, ax = plt.subplots()

# One row per number i: a line, a point and a filled box
for i in range(10):
    ax.plot([0.05, 0.55], [i, i], **amp.line(i))
    ax.plot([0.7], [i], **amp.points(i))
    ax.add_patch(Rectangle((0.82, i - 0.3), 0.13, 0.6, **amp.fill(i, linewidth=0)))

ax.set_xlim(0, 1)
ax.set_ylim(-0.5, 9.5)
ax.tick_params(axis="x", which="both", labelbottom=False)
ax.set_xlabel("line(i)            points(i)      fill(i)")
ax.set_ylabel("i")
amp.label(ax)

fig.savefig("mpl_styles.png")
fig.savefig("mpl_styles.pdf")
