"""matplotlib adapter: use(), points(i), line(i), fill(i), label(ax), info(ax, ...), legend(ax), colorbar(m, ax)."""
import glob
import os
import shutil
import subprocess

import matplotlib as mpl
from matplotlib import font_manager, ticker
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import FontProperties

from . import load_config
from . import series as _series

ROOT_TEXT_SCALE = 0.93376068  # ROOT renders TTF text at this fraction of the nominal size (TTF.cxx kScale)
ROOT_DASH_SCALE = 0.25        # ROOT's PNG output draws SetLineStyleString lengths at 1/4 (measured)
LEGEND_SYMBOL = 0.75          # TLegend symbols span 75% of the symbol column width and of the row height

_AUTO_LOCATOR_INIT = ticker.AutoLocator.__init__


def _register_root_fonts():
    """Make the fonts shipped with ROOT (TeX Gyre Heros) visible to matplotlib."""
    prefix = os.environ.get("ROOTSYS")
    if not prefix and shutil.which("root-config"):
        prefix = subprocess.run(["root-config", "--prefix"], capture_output=True,
                                text=True).stdout.strip()
    if not prefix:
        return
    for d in ("share/root/fonts", "fonts"):
        for f in glob.glob(os.path.join(prefix, d, "texgyreheros-*.otf")):
            font_manager.fontManager.addfont(f)


def _font_family(cfg):
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in [cfg["font"]["family"]] + cfg["font"]["fallbacks"]:
        if name in available:
            return name
    return "DejaVu Sans"


def _pt(cfg, frac):
    """Fraction of the canvas height -> points."""
    return frac * cfg["canvas"]["height_px"] / cfg["canvas"]["dpi"] * 72


def _font(cfg, frac):
    """Text size as a fraction of the canvas height (ROOT precision-2 convention) -> points."""
    return _pt(cfg, frac) * ROOT_TEXT_SCALE


def colormap(cfg=None):
    """Colour map built from the config stops, registered in matplotlib as 'allegro'."""
    cfg = cfg or load_config()
    cmap = LinearSegmentedColormap.from_list("allegro", cfg["colormap"]["stops"])
    if "allegro" not in mpl.colormaps or mpl.colormaps["allegro"] != cmap:
        mpl.colormaps.register(cmap, force=True)
    return cmap


def rc_params(cfg=None, family=None):
    cfg = cfg or load_config()
    family = family or _font_family(cfg)
    c, m, t, a, ln = cfg["canvas"], cfg["margins"], cfg["text"], cfg["axes"], cfg["lines"]
    px = 72 / c["dpi"]

    rc = {
        "figure.figsize": (c["width_px"] / c["dpi"], c["height_px"] / c["dpi"]),
        "figure.dpi": c["dpi"],
        "savefig.dpi": c["dpi"],
        "figure.subplot.left": m["left"],
        "figure.subplot.right": 1 - m["right"],
        "figure.subplot.bottom": m["bottom"],
        "figure.subplot.top": 1 - m["top"],
        "font.family": "sans-serif",
        "font.sans-serif": [family] + cfg["font"]["fallbacks"],
        "font.size": _font(cfg, t["axis_label_size"]),
        "mathtext.fontset": "custom",
        "mathtext.rm": family,
        "mathtext.it": f"{family}:italic",
        "mathtext.bf": f"{family}:bold",
        "mathtext.cal": family,
        "axes.labelsize": _font(cfg, t["axis_title_size"]),
        "axes.linewidth": a["frame_width_px"] * px,
        "axes.formatter.use_mathtext": True,
        "axes.prop_cycle": mpl.cycler(color=cfg["palette"]["colors"]),
        "xaxis.labellocation": a["x_title_loc"],
        "yaxis.labellocation": a["y_title_loc"],
        "legend.frameon": False,
        "legend.fontsize": _font(cfg, t["legend_size"]),
        "legend.borderaxespad": 0,
        "lines.linewidth": ln["width_px"] * px,
        "patch.linewidth": ln["width_px"] * px,
        # matplotlib adds the edge stroke on top of the marker size; ROOT does not
        "lines.markersize": (ln["marker_size_px"] - 1) * px,
        "lines.markeredgewidth": px,
        "lines.scale_dashes": False,
        "errorbar.capsize": 0,
        "image.cmap": colormap(cfg).name,
    }
    for ax in ("xtick", "ytick"):
        rc.update({
            f"{ax}.direction": "in",
            f"{ax}.labelsize": _font(cfg, t["axis_label_size"]),
            f"{ax}.major.size": _pt(cfg, a["tick_length"]),
            f"{ax}.minor.size": _pt(cfg, a["minor_tick_length"]),
            f"{ax}.major.width": a["frame_width_px"] * px,
            f"{ax}.minor.width": a["frame_width_px"] * px,
            f"{ax}.minor.visible": a["minor_ticks"],
            f"{ax}.minor.ndivs": a["ndivisions"] // 100,
        })
    rc["xtick.top"] = rc["ytick.right"] = a["ticks_all_sides"]
    return rc


def use(cfg=None, root_ticks=True):
    """Apply the ALLEGRO style globally.

    root_ticks: choose major tick steps like ROOT (at most ndivisions % 100 intervals of
    1, 2 or 5 x 10^n). This changes matplotlib's default tick locator; False restores it.
    """
    cfg = cfg or load_config()
    _register_root_fonts()
    mpl.rcParams.update(rc_params(cfg))
    if root_ticks:
        nbins = cfg["axes"]["ndivisions"] % 100

        def _init(self):
            ticker.MaxNLocator.__init__(self, nbins=nbins, steps=[1, 2, 5, 10])
        ticker.AutoLocator.__init__ = _init
    else:
        ticker.AutoLocator.__init__ = _AUTO_LOCATOR_INIT


def _to_figure(ax, x, y):
    """Frame coordinates -> figure fraction."""
    p = ax.get_position()
    return p.x0 + x * p.width, p.y0 + y * p.height


def _get(cfg, section, x, y, align):
    s = cfg[section]
    return (s["x"] if x is None else x, s["y"] if y is None else y,
            s["align"] if align is None else align)


def place_titles(ax, cfg=None):
    """Put the axis titles at a fixed distance from the frame, as ROOT does
    (matplotlib otherwise pushes them out when the tick labels get wider)."""
    cfg = cfg or load_config()
    a, c = cfg["axes"], cfg["canvas"]
    inch = c["height_px"] / c["dpi"]
    shift = lambda dx, dy: ax.transAxes + mpl.transforms.ScaledTranslation(dx, dy, ax.figure.dpi_scale_trans)
    ax.yaxis.set_label_coords(0, 1, transform=shift(-a["mpl_title_distance_y"] * inch, 0))
    ax.xaxis.set_label_coords(1, 0, transform=shift(0, -a["mpl_title_distance_x"] * inch))


def label(ax, status=None, x=None, y=None, align=None, cfg=None):
    """'ALLEGRO Simulation'; (x, y) is the left (or right) end of the baseline.
    Also places the axis titles like ROOT (see place_titles)."""
    cfg = cfg or load_config()
    if ax.name != "3d":
        place_titles(ax, cfg)
    x, y, align = _get(cfg, "label", x, y, align)
    size = _font(cfg, cfg["text"]["label_size"])
    status = cfg["label"]["status"] if status is None else status
    exp = dict(text=cfg["label"]["experiment"], size=size, weight="bold", style="italic")
    words = [exp, dict(text=status, size=size)] if status else [exp]
    if align == "right":
        words.reverse()
    sign = 1 if align == "left" else -1
    fx, fy = _to_figure(ax, x, y)
    first = words[0]
    fig = ax.figure
    texts = [fig.text(fx, fy, first.pop("text"), ha=align, va="baseline", **first)]
    for w in words[1:]:
        texts.append(fig.add_artist(mpl.text.Annotation(
            w.pop("text"), xy=((1 + sign) / 2, 0), xycoords=texts[-1],
            xytext=(sign * 0.4 * size, 0), textcoords="offset points", ha=align, va="bottom", **w)))
    return texts


def info(ax, process=None, energy=None, lumi=None, x=None, y=None, align=None, valign=None, cfg=None):
    """Process (several lines separated by '\\n'), then sqrt(s) [GeV] and luminosity [ab^-1].
    valign="top": (x, y) is the top of the first line; valign="bottom": the baseline of the
    last line (use y = label y to put the info on the same line as 'ALLEGRO Simulation')."""
    cfg = cfg or load_config()
    x, y, align = _get(cfg, "info", x, y, align)
    c = cfg["info"]
    valign = c["valign"] if valign is None else valign
    lines = process.split("\n") if process else []
    sub = []
    if energy is not None:
        sub.append(rf"$\sqrt{{s}}$ = {energy:g} {c['energy_unit']}")
    if lumi is not None:
        sub.append(f"{lumi:g} {c['lumi_unit_mpl']}")
    if sub:
        lines.append(", ".join(sub))

    size = cfg["text"]["info_size"]
    step = size * c["line_spacing"]
    fx, fy = _to_figure(ax, x, y)
    top = valign == "top"
    return [ax.figure.text(fx, fy - k * step if top else fy + (len(lines) - 1 - k) * step, line,
                           ha=align, va="top" if top else "baseline", size=_font(cfg, size))
            for k, line in enumerate(lines)]


def legend(ax, x=None, y=None, align=None, cfg=None, **kwargs):
    """ax.legend laid out like ROOT's TLegend: a box of fixed width whose top-right
    (or top-left) corner is at (x, y), symbols centred in the first `margin` of the box."""
    cfg = cfg or load_config()
    x, y, align = _get(cfg, "legend", x, y, align)
    c, lg = cfg["canvas"], cfg["legend"]
    if align == "right":
        x -= lg["width"]
    frame = ax.get_position()
    font_pt = _font(cfg, cfg["text"]["legend_size"])
    font = font_pt * c["dpi"] / 72                                       # px
    row = lg["row_height"] * c["height_px"]                              # px
    column = lg["margin"] * lg["width"] * frame.width * c["width_px"]    # px
    handleheight = LEGEND_SYMBOL * row / font
    # height of one legend row as matplotlib lays it out (Legend._init_legend_box, HPacker baseline)
    _, text_h, text_d = RendererAgg(1, 1, c["dpi"]).get_text_width_height_descent(
        "lp", FontProperties(size=font_pt), ismath=False)
    sym_d = 0.35 * font * (handleheight - 0.7)
    content = max(font * handleheight - 2 * sym_d, text_h - text_d) + max(sym_d, text_d)
    pad = (row - content) / 2
    # matplotlib pads the box by `pad` on every side; shift so the symbol column starts at x
    shift = (column * (1 - LEGEND_SYMBOL) / 2 - pad) / (c["width_px"])
    fx, fy = _to_figure(ax, x, y)
    opts = dict(
        loc="upper left", bbox_to_anchor=(fx + shift, fy), bbox_transform=ax.figure.transFigure,
        borderpad=pad / font, labelspacing=(row - content) / font,
        handlelength=LEGEND_SYMBOL * column / font, handleheight=handleheight,
        handletextpad=column * (1 - LEGEND_SYMBOL) / 2 / font, borderaxespad=0,
    )
    opts.update(kwargs)
    return ax.legend(**opts)


def points(i, marker=None, cfg=None, **extra):
    """Style for points: colour i and marker `marker` (default i).
    Use as ax.errorbar(x, y, yerr, linestyle="none", **amp.points(i)).
    Extra keyword arguments are matplotlib options that override the style."""
    cfg = cfg or load_config()
    s = _series(i, marker=marker, config=cfg)
    c, px = s["color"], 72 / cfg["canvas"]["dpi"]
    # mpl_scale matches the size of markers that ROOT draws smaller (e.g. diamonds)
    return {"color": c, "marker": s["marker"], "markeredgecolor": c,
            "markerfacecolor": c if s["filled"] else "none",
            "markersize": (cfg["lines"]["marker_size_px"] - 1) * px * s["mpl_scale"]} | extra


def line(i, linestyle=None, cfg=None, **extra):
    """Style for lines without markers: colour i and line style `linestyle` (default i).
    Use as ax.stairs(counts, edges, **amp.line(i)) or ax.plot(x, y, **amp.line(i)).
    Extra keyword arguments are matplotlib options that override the style."""
    cfg = cfg or load_config()
    s = _series(i, linestyle=linestyle, config=cfg)
    px = 72 / cfg["canvas"]["dpi"]
    dashes = [d * ROOT_DASH_SCALE * px for d in s["dashes_root"]]
    return {"color": s["color"], "linewidth": cfg["lines"]["width_px"] * px,
            "linestyle": (0, dashes) if dashes else "solid"} | extra


def fill(i, cfg=None, **extra):
    """Style for filled areas in colour i.
    Use as ax.stairs(counts, edges, fill=True, **amp.fill(i)) or ax.fill_between(..., **amp.fill(i)).
    Extra keyword arguments are matplotlib options that override the style."""
    c = _series(i, config=cfg or load_config())["color"]
    return {"facecolor": c, "edgecolor": c} | extra


def colorbar(mappable, ax, label=None, cfg=None):
    """Colour scale placed like ROOT's COLZ palette; makes room by using margins.right_colz."""
    cfg = cfg or load_config()
    fig, p = ax.figure, ax.get_position()
    right = 1 - cfg["margins"]["right_colz"]
    ax.set_position([p.x0, p.y0, right - p.x0, p.height])
    cax = fig.add_axes([right + 0.005, p.y0, 0.045, p.height])
    cbar = fig.colorbar(mappable, cax=cax)
    cax.tick_params(which="both", left=False, top=False)
    if label:
        cbar.set_label(label, loc="top")
    return cbar
