import pytest

import allegro_style
from allegro_style import root as aroot


def test_header_up_to_date():
    assert aroot.HEADER.read_text() == aroot.header_text(), \
        "style.json changed: run `python -m allegro_style.root` to regenerate AllegroStyleConfig.h"


def test_header_contains_every_scalar():
    text = aroot.header_text()
    assert 'label_experiment = "ALLEGRO"' in text
    assert "const int markers_root_n = 10;" in text
    assert "const double margins_left" in text
    mixed = aroot.header_text({"a": [1, 0.75], "b": [1, 2], "c": ["x"], "d": [True]})
    assert "const double a[] = {1.0, 0.75};" in mixed and "const int b[] = {1, 2};" in mixed
    assert 'const char* const c[] = {"x"};' in mixed and "const bool d[] = {true};" in mixed


@pytest.fixture(scope="module")
def R():
    ROOT = pytest.importorskip("ROOT")
    ROOT.gROOT.SetBatch(True)
    ROOT.gROOT.SetWebDisplay("off")  # ROOT >= 6.38 otherwise needs a browser to save images/PDF
    aroot.use()
    return ROOT


def test_style_applied(R):
    cfg = allegro_style.load_config()
    s = R.gStyle
    assert s.GetName() == "ALLEGRO"
    assert s.GetPadLeftMargin() == pytest.approx(cfg["margins"]["left"])
    assert s.GetPadTopMargin() == pytest.approx(cfg["margins"]["top"])
    assert s.GetPadTickX() == 1 and s.GetPadTickY() == 1
    assert s.GetOptStat() == 0 and s.GetOptTitle() == 0
    assert s.GetLabelFont("X") == cfg["font"]["root_regular"]
    assert s.GetTitleSize("Y") == pytest.approx(cfg["text"]["axis_title_size"])


def test_colours_and_markers(R):
    cfg = allegro_style.load_config()
    for i, hexc in enumerate(cfg["palette"]["colors"]):
        assert R.gROOT.GetColor(R.AllegroColor(i)).AsHexString().lower() == hexc
        assert R.AllegroMarker(i) == cfg["markers"]["root"][i % len(cfg["markers"]["root"])]
    h = R.TH1D("h_points", "", 10, 0, 1)
    R.AllegroPoints(h, 1)
    assert h.GetMarkerStyle() == cfg["markers"]["root"][1]
    assert h.GetLineColor() == h.GetMarkerColor() == R.AllegroColor(1)


def test_label_info_legend_positions(R):
    cfg = allegro_style.load_config()
    c = R.AllegroCanvas("c_pos")
    lab = R.AllegroLabel()
    m = cfg["margins"]
    assert lab.GetX() == pytest.approx(m["left"] + cfg["label"]["x"] * (1 - m["left"] - m["right"]))
    assert lab.GetTitle() == "#bf{#it{ALLEGRO}} Simulation"
    R.AllegroInfo("A\nB", 91.2, 10.8)
    texts = [o for o in c.GetListOfPrimitives() if o.ClassName() == "TLatex"]
    assert [t.GetTitle() for t in texts[1:]] == ["A", "B", "#sqrt{#it{s}} = 91.2 GeV, 10.8 ab#kern[-0.5]{^{#minus1}}"]
    leg = R.AllegroLegend(3)
    assert leg.GetX2() == pytest.approx(m["left"] + cfg["legend"]["x"] * (1 - m["left"] - m["right"]))
    assert leg.GetY2() - leg.GetY1() == pytest.approx(3 * cfg["legend"]["row_height"])
    left = R.AllegroLegend(2, 0.05, 0.5, "left")
    assert left.GetX1() == pytest.approx(m["left"] + 0.05 * (1 - m["left"] - m["right"]))


def test_line_and_fill(R):
    cfg = allegro_style.load_config()
    h = R.TH1D("h_kinds", "", 10, 0, 1)
    R.AllegroLine(h, 1)
    assert h.GetLineStyle() == cfg["line_styles"]["root"][1]
    assert h.GetLineWidth() == cfg["lines"]["width_px"]
    assert R.gStyle.GetLineStyleString(h.GetLineStyle()).split() == cfg["line_styles"]["dashes_root"][1].split()
    R.AllegroFill(h, 2)
    assert h.GetFillColor() == h.GetLineColor() == R.AllegroColor(2)
    assert h.GetFillStyle() == 1001


def test_colormap_and_colz_canvas(R):
    cfg = allegro_style.load_config()
    assert R.gStyle.GetNumberContours() == 255
    first = R.gROOT.GetColor(R.gStyle.GetColorPalette(0))
    last = R.gROOT.GetColor(R.gStyle.GetColorPalette(R.gStyle.GetNumberOfColors() - 1))
    for c, hexc in [(first, cfg["colormap"]["stops"][0]), (last, cfg["colormap"]["stops"][-1])]:
        rgb = [int(hexc[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        assert [c.GetRed(), c.GetGreen(), c.GetBlue()] == pytest.approx(rgb, abs=0.01)
    c = R.AllegroCanvas("c_colz_test", "", True)
    assert c.GetRightMargin() == pytest.approx(cfg["margins"]["right_colz"])
    assert (c.GetWw(), c.GetWh()) == (cfg["canvas"]["width_px"], cfg["canvas"]["height_px"])


def test_pyroot_pdf(R, tmp_path):
    c = R.AllegroCanvas("c_pdf")
    h = R.TH1D("h_pdf", "", 10, 0, 1)
    h.FillRandom("pol0", 100)
    R.AllegroLine(h, 1)
    h.Draw("HIST")
    R.AllegroLabel()
    R.AllegroInfo("Z #rightarrow q#bar{q}", 91.2, 10.8)
    out = tmp_path / "pyroot.pdf"
    c.SaveAs(str(out))
    assert out.read_bytes().startswith(b"%PDF") and out.stat().st_size > 1000


def test_colour_marker_linestyle_independent(R):
    cfg = allegro_style.load_config()
    h = R.TH1D("h_indep", "", 10, 0, 1)
    R.AllegroLine(h, 3, 0)
    assert h.GetLineColor() == R.AllegroColor(3) and h.GetLineStyle() == cfg["line_styles"]["root"][0]
    R.AllegroLine(h, 0, 2)
    assert h.GetLineColor() == R.AllegroColor(0) and h.GetLineStyle() == cfg["line_styles"]["root"][2]
    R.AllegroPoints(h, 1, 5)
    assert h.GetMarkerColor() == R.AllegroColor(1) and h.GetMarkerStyle() == cfg["markers"]["root"][5]
