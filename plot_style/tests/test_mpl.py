import pytest

mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import allegro_style  # noqa: E402
from allegro_style import mpl as amp  # noqa: E402


@pytest.fixture
def styled():
    with mpl.rc_context():
        amp.use()
        yield
    plt.close("all")


def _bbox(artist):
    return artist.get_window_extent(artist.figure.canvas.get_renderer())


def test_rc_params_valid_for_matplotlib():
    with mpl.rc_context(amp.rc_params()):
        pass


def test_sizes_follow_config(styled):
    cfg = allegro_style.load_config()
    assert mpl.rcParams["axes.labelsize"] == pytest.approx(amp._font(cfg, cfg["text"]["axis_title_size"]))
    assert mpl.rcParams["figure.subplot.top"] == pytest.approx(1 - cfg["margins"]["top"])
    assert tuple(mpl.rcParams["figure.figsize"]) == (8, 6)


def test_ticks_everywhere(styled):
    for key in ["xtick.top", "ytick.right", "xtick.minor.visible", "ytick.minor.visible"]:
        assert mpl.rcParams[key] is True
    assert mpl.rcParams["xtick.direction"] == mpl.rcParams["ytick.direction"] == "in"


def test_colour_cycle_is_palette(styled):
    cfg = allegro_style.load_config()
    assert mpl.rcParams["axes.prop_cycle"].by_key()["color"] == cfg["palette"]["colors"]


def test_root_font_used_when_root_available(styled):
    if amp._font_family(allegro_style.load_config()) != "TeX Gyre Heros":
        pytest.skip("TeX Gyre Heros not found (no ROOT installation?)")
    assert mpl.rcParams["font.sans-serif"][0] == "TeX Gyre Heros"


def test_points_open_and_filled():
    assert amp.points(0)["markerfacecolor"] != "none"
    assert amp.points(5)["markerfacecolor"] == "none"


def test_label_default_above_frame(styled):
    fig, ax = plt.subplots()
    exp, status = amp.label(ax)
    fig.canvas.draw()
    frame = _bbox(ax)
    assert exp.get_text() == "ALLEGRO" and status.get_text() == "Simulation"
    assert _bbox(exp).x0 == pytest.approx(frame.x0, abs=1)
    assert _bbox(exp).x1 < _bbox(status).x0
    baseline = fig.transFigure.transform(exp.get_position())[1]
    assert frame.y1 < baseline < frame.y1 + 10  # just above the frame
    for t in (exp, status):
        assert _bbox(t).y1 <= fig.bbox.y1


def test_label_custom_position_and_status(styled):
    fig, ax = plt.subplots()
    texts = amp.label(ax, status="Preliminary", x=0.95, y=0.9, align="right")
    fig.canvas.draw()
    frame = _bbox(ax)
    assert [t.get_text() for t in texts] == ["Preliminary", "ALLEGRO"]
    assert _bbox(texts[0]).x1 == pytest.approx(frame.x0 + 0.95 * frame.width, abs=1)
    assert _bbox(texts[1]).x1 < _bbox(texts[0]).x0
    assert len(amp.label(ax, status="")) == 1


def test_info_inside_frame_below_label(styled):
    fig, ax = plt.subplots()
    exp, _ = amp.label(ax)
    lines = amp.info(ax, process=r"$Z\to q\bar{q}$", energy=91.2, lumi=10.8)
    fig.canvas.draw()
    frame = _bbox(ax)
    assert len(lines) == 2
    assert "91.2" in lines[1].get_text() and "10.8" in lines[1].get_text()
    for t in lines:
        b = _bbox(t)
        assert frame.x0 < b.x0 and b.x1 < frame.x1 and frame.y0 < b.y0 and b.y1 < frame.y1
    assert _bbox(lines[0]).y0 > _bbox(lines[1]).y1
    assert _bbox(lines[0]).y1 < _bbox(exp).y0


def test_info_partial_and_multiline(styled):
    fig, ax = plt.subplots()
    assert [t.get_text() for t in amp.info(ax, process="A\nB")] == ["A", "B"]
    assert len(amp.info(ax, energy=240)) == 1
    assert amp.info(ax) == []


def test_legend_top_right_and_movable(styled):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label="a", **amp.points(0))
    leg = amp.legend(ax)
    fig.canvas.draw()
    frame, b = _bbox(ax), _bbox(leg)
    cfg = allegro_style.load_config()["legend"]
    column = cfg["margin"] * cfg["width"] * frame.width
    # like TLegend: top at y, text starts after the symbol column of a box of fixed width
    assert b.y1 == pytest.approx(frame.y0 + cfg["y"] * frame.height, abs=1)
    text_x0 = _bbox(leg.get_texts()[0]).x0
    assert text_x0 == pytest.approx(frame.x0 + (cfg["x"] - cfg["width"]) * frame.width + column, abs=2)
    leg = amp.legend(ax, x=0.05, y=0.5, align="left")
    fig.canvas.draw()
    assert _bbox(leg.get_texts()[0]).x0 == pytest.approx(frame.x0 + 0.05 * frame.width + column, abs=2)


def test_styles_work_with_matplotlib_functions(styled):
    import numpy as np
    fig, ax = plt.subplots()
    edges = np.arange(6)
    ax.errorbar([1, 2], [1, 2], yerr=[0.1, 0.1], linestyle="none", **amp.points(0))
    ax.plot([0, 1], [1, 2], **amp.line(1))
    ax.stairs([1, 2, 3, 2, 1], edges, **amp.line(1))
    ax.stairs([1, 2, 3, 2, 1], edges, fill=True, **amp.fill(2))
    ax.hist([0.5, 1.5], bins=edges, histtype="step", **amp.line(3))
    ax.fill_between([0, 1], [0, 1], **amp.fill(4))
    assert amp.line(0)["linestyle"] == "solid"
    assert amp.line(1)["linestyle"][1]  # dashed
    fig.canvas.draw()


def test_root_like_ticks(styled):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 200)
    ticks = [t for t in ax.get_xticks() if 0 <= t <= 200]
    assert ticks == list(range(0, 201, 20))  # ROOT: steps of 1, 2, 5 x 10^n, up to 10 divisions
    amp.use(root_ticks=False)
    fig, ax = plt.subplots()
    ax.set_xlim(0, 200)
    assert 25 in ax.get_xticks()  # matplotlib default allows steps of 2.5
    amp.use()


def test_colormap_from_config(styled):
    cfg = allegro_style.load_config()
    cmap = mpl.colormaps[mpl.rcParams["image.cmap"]]
    assert mpl.colors.to_hex(cmap(0.0)) == cfg["colormap"]["stops"][0]
    assert mpl.colors.to_hex(cmap(1.0)) == cfg["colormap"]["stops"][-1]


def test_colorbar_like_root_colz(styled):
    cfg = allegro_style.load_config()
    fig, ax = plt.subplots()
    mesh = ax.pcolormesh([[0, 1], [2, 3]])
    cbar = amp.colorbar(mesh, ax, label="Events")
    right = 1 - cfg["margins"]["right_colz"]
    assert ax.get_position().x1 == pytest.approx(right)
    assert cbar.ax.get_position().x0 == pytest.approx(right + 0.005)
    assert cbar.ax.get_position().y0 == pytest.approx(ax.get_position().y0)


def test_colour_marker_linestyle_independent():
    colors = allegro_style.load_config()["palette"]["colors"]
    solid_red = amp.line(3, linestyle=0)
    assert solid_red["color"] == colors[3] and solid_red["linestyle"] == "solid"
    dashed_navy = amp.line(0, linestyle=1)
    assert dashed_navy["color"] == colors[0] and dashed_navy["linestyle"] != "solid"
    open_circle = amp.points(0, marker=5)
    assert open_circle["marker"] == "o" and open_circle["markerfacecolor"] == "none"
    assert open_circle["color"] == colors[0]
    assert amp.fill(4)["facecolor"] == colors[4]


def test_extra_options_override_style():
    assert amp.line(0, linewidth=3)["linewidth"] == 3
    assert amp.points(1, markersize=12)["markersize"] == 12
    f = amp.fill(2, alpha=0.5)
    assert f["alpha"] == 0.5 and f["facecolor"] == amp.fill(2)["facecolor"]
