import math
import re

import pytest

import allegro_style
from cvd import MACHADO, min_distances

ROOT_MARKERS_FILLED = {20, 21, 22, 23, 29, 33, 34, 39, 41, 43, 45, 47, 49}
ROOT_MARKERS_OPEN = {24, 25, 26, 27, 28, 30, 32, 42, 44, 46}
ROOT_TO_MPL = {20: "o", 24: "o", 21: "s", 25: "s", 22: "^", 26: "^", 23: "v", 32: "v", 33: "d", 27: "d"}


@pytest.fixture(scope="module")
def cfg():
    return allegro_style.load_config()


def test_required_sections(cfg):
    for key in ["canvas", "margins", "font", "text", "axes", "lines", "palette",
                "markers", "line_styles", "colormap", "label", "info", "legend"]:
        assert key in cfg, key


def test_margins_leave_room_for_plot(cfg):
    m = cfg["margins"]
    assert all(0 <= v < 0.5 for v in m.values())
    assert m["left"] + m["right"] < 0.5
    assert m["bottom"] + m["top"] < 0.5


def test_label_fits_above_frame(cfg):
    m, lab = cfg["margins"], cfg["label"]
    frame_h = 1 - m["top"] - m["bottom"]
    if lab["y"] >= 1:  # drawn outside the frame: must fit in the top margin
        baseline = 1 - m["top"] + (lab["y"] - 1) * frame_h
        assert baseline + cfg["text"]["label_size"] <= 1


def test_positions_and_alignment(cfg):
    for section in ["label", "info", "legend"]:
        assert cfg[section]["align"] in ("left", "right")
        assert -0.5 < cfg[section]["x"] < 1.5 and -0.5 < cfg[section]["y"] < 1.5


def test_relative_sizes_are_fractions(cfg):
    for v in cfg["text"].values():
        assert 0 < v < 0.2
    assert 0 < cfg["axes"]["minor_tick_length"] < cfg["axes"]["tick_length"] < 0.1


def test_palette_hex_and_unique(cfg):
    colors = cfg["palette"]["colors"]
    assert all(re.fullmatch(r"#[0-9a-f]{6}", c) for c in colors)
    assert len(set(colors)) == len(colors)


@pytest.mark.parametrize("vision", list(MACHADO))
def test_palette_colourblind_safe(cfg, vision):
    # every pair of colours stays distinguishable (CIELAB dE > 15) for each simulated deficiency
    pair, _ = min_distances(cfg["palette"]["colors"], vision)
    assert pair > 15


def test_markers_valid_and_consistent(cfg):
    m = cfg["markers"]
    assert len(m["root"]) == len(m["mpl"]) == len(m["filled"]) == len(m["mpl_scale"])
    assert all(0 < k <= 2 for k in m["mpl_scale"])
    for root, mpl, filled in zip(m["root"], m["mpl"], m["filled"]):
        assert root in (ROOT_MARKERS_FILLED if filled else ROOT_MARKERS_OPEN)
        assert ROOT_TO_MPL[root] == mpl
    # each (shape, fill) pair is used once, so series are distinguishable without colour
    assert len(set(zip(m["mpl"], m["filled"]))) == len(m["mpl"])


def test_line_styles(cfg):
    ls = cfg["line_styles"]
    assert len(ls["root"]) == len(ls["dashes_root"])
    assert ls["root"][0] == 1 and ls["dashes_root"][0] == ""
    for code, dashes in zip(ls["root"][1:], ls["dashes_root"][1:]):
        assert 2 <= code <= 10
        values = [float(d) for d in dashes.split()]
        assert len(values) % 2 == 0 and all(v > 0 for v in values)


def test_colormap_stops(cfg):
    stops = cfg["colormap"]["stops"]
    assert len(stops) >= 2 and all(re.fullmatch(r"#[0-9a-f]{6}", c) for c in stops)


def test_root_fonts_are_precision_2(cfg):
    # precision 2 => text size is a fraction of the pad height, as everywhere in the config
    f = cfg["font"]
    for key in ["root_regular", "root_italic", "root_bold", "root_bold_italic"]:
        assert f[key] % 10 == 2


def test_series_cycles(cfg):
    n = math.lcm(len(cfg["palette"]["colors"]), len(cfg["markers"]["root"]), len(cfg["line_styles"]["root"]))
    assert allegro_style.series(0) == allegro_style.series(n)
    s = allegro_style.series(5)
    assert s["color"] == cfg["palette"]["colors"][5]
    assert s["root_marker"] == cfg["markers"]["root"][5]


def test_series_marker_and_linestyle_independent(cfg):
    s = allegro_style.series(3, marker=5, linestyle=2)
    assert s["color"] == cfg["palette"]["colors"][3]
    assert s["root_marker"] == cfg["markers"]["root"][5]
    assert s["root_line"] == cfg["line_styles"]["root"][2]
    # choosing the line style leaves colour and marker alone
    base, dashed = allegro_style.series(0), allegro_style.series(0, linestyle=1)
    assert {k for k in base if base[k] != dashed[k]} == {"root_line", "dashes_root"}
