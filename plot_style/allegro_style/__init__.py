"""ALLEGRO plotting style for matplotlib and ROOT, driven by a single style.json.

Positions (label, info, legend) are in frame coordinates: (0, 0) is the
bottom-left corner of the axes frame, (1, 1) the top-right one.
"""
import json
from pathlib import Path

STYLE_FILE = Path(__file__).with_name("style.json")


def load_config(path=STYLE_FILE):
    with open(path) as f:
        return json.load(f)


def series(i, marker=None, linestyle=None, config=None):
    """Colour i, marker `marker` and line style `linestyle` (both default to i).
    Each list wraps around on its own."""
    cfg = config or load_config()
    colors, m, ls = cfg["palette"]["colors"], cfg["markers"], cfg["line_styles"]
    j = (i if marker is None else marker) % len(m["root"])
    k = (i if linestyle is None else linestyle) % len(ls["root"])
    return {
        "color": colors[i % len(colors)],
        "marker": m["mpl"][j],
        "root_marker": m["root"][j],
        "filled": m["filled"][j],
        "mpl_scale": m["mpl_scale"][j],
        "root_line": ls["root"][k],
        "dashes_root": [float(d) for d in ls["dashes_root"][k].split()],
    }
