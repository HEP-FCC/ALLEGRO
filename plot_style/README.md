# ALLEGRO plot style

One plotting style for **matplotlib** and **ROOT**. All settings (sizes, margins, font,
colours, markers, line styles, label text, positions) live in a single file,
[`allegro_style/style.json`](allegro_style/style.json). Both libraries read from it, so
the same plot made with either one looks the same.

Tested with the Key4hep release (matplotlib 3.10, ROOT 6.38) and the Key4hep nightlies
(matplotlib 3.11, ROOT 6.40). No other dependencies.

## Setup

```bash
source /cvmfs/sw.hsf.org/key4hep/setup.sh          # or /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh
export PYTHONPATH=/path/to/plot-style:$PYTHONPATH
```

## Quick start

### matplotlib

```python
import matplotlib.pyplot as plt
from allegro_style import mpl as amp

amp.use()                                    # once, before creating figures
fig, ax = plt.subplots()

ax.errorbar(x, y, yerr=e, linestyle="none", label="Data", **amp.points(0))
ax.stairs(counts, edges, label="Signal", **amp.line(1))
ax.stairs(counts, edges, fill=True, label="Background", **amp.fill(2))

ax.set_xlabel(r"$m_{jj}$ [GeV]")
ax.set_ylabel("Events / 5 GeV")
amp.label(ax)                                # "ALLEGRO Simulation" above the frame
amp.info(ax, process=r"$Z\to q\bar{q}$", energy=91.2, lumi=10.8)
amp.legend(ax)                               # top right
fig.savefig("plot.pdf")
```

The data in this example:
- `x`, `y`, `e` are the positions, values and errors of the points.
- `counts`, `edges` describe a histogram: the content of each bin (N numbers) and the
  bin boundaries (N+1 numbers). They are what NumPy returns, e.g.
  `counts, edges = np.histogram(values, bins=40, range=(0, 200))`. For a ROOT `TH1`
  read with `uproot`, `h.to_numpy()` returns the same pair.

`fig` is the whole figure (like a `TCanvas`) and `ax` is the plotting area inside it.
`amp.points(0)` returns the style (colour, marker, size…) as a set of matplotlib
options. The `**` in front passes those options to the plotting function. It is the
same as writing `color="#0f0f4b", marker="o", ...` by hand, and it works with any
matplotlib function (`errorbar`, `plot`, `stairs`, `hist`, `fill_between`, …).

### ROOT (C++ macro)

```cpp
#include "/path/to/plot-style/allegro_style/root/AllegroStyle.C"

void plot() {
   SetAllegroStyle();
   auto c = AllegroCanvas();                 // use this instead of new TCanvas(...)
   AllegroPoints(hData, 0);  hData->Draw("E X0");
   AllegroLine(hSig, 1);     hSig->Draw("HIST SAME");
   AllegroFill(hBkg, 2);     // e.g. in a THStack drawn with "HIST"
   AllegroLabel();
   AllegroInfo("Z #rightarrow q#bar{q}", 91.2, 10.8);
   auto leg = AllegroLegend(3);              // number of entries
   leg->AddEntry(hData, "Data", "pe");
   leg->Draw();
   c->SaveAs("plot.pdf");
}
```

### PyROOT

```python
import ROOT
ROOT.gROOT.SetWebDisplay("off")              # see the note below
from allegro_style import root as aroot
R = aroot.use()                              # loads AllegroStyle.C and calls SetAllegroStyle()
c = R.AllegroCanvas()
R.AllegroPoints(h, 0); h.Draw("E X0")
R.AllegroLabel(); R.AllegroInfo("Z #rightarrow q#bar{q}", 91.2, 10.8)
c.SaveAs("plot.pdf")
```

**Note (ROOT ≥ 6.38):** ROOT draws canvases through a web browser by default, and on
machines without one (lxplus, batch jobs) saving PNG *or* PDF fails. Turn it off with
`root --web=off` for macros, or `ROOT.gROOT.SetWebDisplay("off")` in PyROOT.

## Points, lines and fills

There is one function for each way of drawing data, with the same name in both
libraries:

| Drawing | matplotlib | ROOT | sets |
|---|---|---|---|
| points (with error bars) | `amp.points(i)` | `AllegroPoints(h, i)`, draw `"E X0"` | colour + marker |
| lines without markers | `amp.line(i)` | `AllegroLine(h, i)`, draw `"HIST"` | colour + line style |
| filled areas | `amp.fill(i)` | `AllegroFill(h, i)`, draw `"HIST"` | colour |

The ROOT functions work on any histogram, graph, function, `TLine`, `TMarker` or
`TBox`.

### Choosing colour, marker and line style

The number `i` picks the **colour**. For points and lines it also picks the **marker**
or **line style**, so drawing 0, 1, 2… gives every dataset a different colour *and*
shape, and the plot stays readable without colour. To choose the marker or line style
yourself, give a second number:

```python
amp.points(2)               # yellow, filled triangle (colour 2, marker 2)
amp.points(2, marker=5)     # yellow, open circle     (colour 2, marker 5)
amp.line(3)                 # red, dash-dot           (colour 3, line style 3)
amp.line(3, linestyle=0)    # red, solid              (colour 3, line style 0)
amp.fill(4)                 # light cyan
```
```cpp
AllegroPoints(h, 2);        // yellow, filled triangle
AllegroPoints(h, 2, 5);     // yellow, open circle
AllegroLine(h, 3);          // red, dash-dot
AllegroLine(h, 3, 0);       // red, solid
AllegroFill(h, 4);          // light cyan
```

The numbers refer to these lists, defined in `style.json`:

| number | colour | marker | line style |
|---|---|---|---|
| 0 | navy `#0f0f4b` | ● filled circle | solid |
| 1 | blue `#1a75bc` | ■ filled square | dashed |
| 2 | yellow `#e8c200` | ▲ filled triangle | dotted |
| 3 | red `#bd1f01` | ▼ filled triangle down | dash-dot |
| 4 | light cyan `#92dadd` | ◆ filled diamond | |
| 5 | brown `#a96b59` | ○ open circle | |
| 6 | | □ open square | |
| 7 | | △ open triangle | |
| 8 | | ▽ open triangle down | |
| 9 | | ◇ open diamond | |

A number past the end of a list starts again from the top: colour 6 is navy, line
style 4 is solid. The `styles` example draws numbers 0–9 as line, points and fill (see
[Examples](#examples)).

The colours are the ALLEGRO logo navy, blue and yellow, plus red, light cyan and brown.
The tests check that they stay distinguishable for colour-blind readers. Yellow has low
contrast on white, so use it for filled areas or thick markers rather than thin lines.

### Anything else (transparency, width…)

matplotlib: add the option inside the call, and it overrides the style:

```python
ax.stairs(counts, edges, fill=True, **amp.fill(1, alpha=0.5))   # half-transparent
ax.plot(x, y, **amp.line(0, linewidth=3))                       # thicker line
```

ROOT: set it afterwards with the usual ROOT setters:

```cpp
AllegroFill(h, 1); h->SetFillColorAlpha(AllegroColor(1), 0.5);  // half-transparent
AllegroLine(g, 0); g->SetLineWidth(3);                          // thicker line
```

For your own objects, `AllegroColor(n)`, `AllegroMarker(n)` and `AllegroLineStyle(n)`
return single values, e.g. `box->SetLineColor(AllegroColor(2))`. In Python,
`amp.line(2)["color"]` gives colour 2 as a hex string.

## Label, info and legend

| | matplotlib | ROOT |
|---|---|---|
| "ALLEGRO Simulation" | `amp.label(ax)` | `AllegroLabel()` |
| process, √s, luminosity | `amp.info(ax, process, energy, lumi)` | `AllegroInfo(process, energy, lumi)` |
| legend | `amp.legend(ax)` | `AllegroLegend(n_entries)` |

By default the label sits just above the top-left corner of the frame, the info inside
the frame below it, and the legend in the top-right corner.

Positions are in **frame coordinates**: (0, 0) is the bottom-left corner of the frame
and (1, 1) the top-right one. Each call can change the text and the position:

```python
amp.label(ax, status="Preliminary")                                    # "ALLEGRO Preliminary"
amp.label(ax, x=1, y=1.012, align="right")                             # top-right instead
amp.info(ax, process="Z→qq\nbarrel only", energy=240)                  # '\n' = new line; no lumi
amp.info(ax, energy=240, x=1, y=1.012, align="right", valign="bottom") # on the label's line
amp.legend(ax, x=0.04, y=0.95, align="left")                           # top-left
```
```cpp
AllegroLabel("Preliminary");
AllegroLabel("Simulation", 1, 1.012, "right");
AllegroInfo("Z #rightarrow q#bar{q}\nbarrel only", 240);              // lumi omitted
AllegroInfo("", 240, -1, 1, 1.012, "right", "bottom");                // -1 = no lumi
AllegroLegend(3, 0.04, 0.95, "left");
```

`align` says which end of the text is at `x`. For `info`, `valign="top"` (default)
puts the top of the first line at `y`, and `valign="bottom"` puts the baseline of the
last line at `y`. Use the latter with the label's `y` (1.012) to place the info on the
same line as "ALLEGRO Simulation", e.g. on 2D plots where the frame is full.

## 2D and 3D plots

- **2D colour map:**
  - matplotlib: draw with `ax.pcolormesh(...)` (or `ax.imshow`), then call
    `amp.colorbar(mesh, ax, label="Events")`, *before* `amp.label` / `amp.info`, since
    it narrows the frame.
  - ROOT: create the canvas with `AllegroCanvas("c", "", true)` to leave room for the
    colour scale, then `h2->Draw("COLZ")`.
- **3D (surface, lego):** the font, colours and colour map apply, but the two libraries
  draw the 3D box differently, so these plots don't match exactly.

## Examples

The two example files draw the same four plots, one with each library:

| Function in [`examples/example_mpl.py`](examples/example_mpl.py) and [`examples/example_root.C`](examples/example_root.C) | Shows |
|---|---|
| `points` | points with error bars, legend, label, info |
| `hist` | stacked filled histograms with line histograms on top |
| `colz` | 2D colour map with colour scale; info on the label's line |
| `styles` | numbers 0–9 as `line`, `points` and `fill` |

Run them directly:

```bash
python examples/example_mpl.py outdir/ png                          # or pdf
root -l -b -q --web=off 'examples/example_root.C("outdir/", "png")'
```

or run the tests (below), which write all of them to `tests/output/`: `mpl_points.png`,
`root_points.png`, …, `mpl_styles.png`, `root_styles.png`.

## PDF output

`fig.savefig("plot.pdf")` and `c->SaveAs("plot.pdf")` both work and give the same layout
as the PNGs.
- matplotlib embeds the TeX Gyre Heros font in the PDF.
- ROOT, as always, refers to the standard Helvetica without embedding it. Every viewer
  shows an identical font, but if a journal insists on embedded fonts, re-embed them
  with Ghostscript (`gs -o out.pdf -sDEVICE=pdfwrite -dEmbedAllFonts=true plot.pdf`).

## Changing the style

Every setting is in `allegro_style/style.json`, shared by matplotlib and ROOT. For
example, to switch to "ALLEGRO Preliminary", put the legend top left, enlarge the axis
titles and use thicker lines:

```json
"text":   { "axis_title_size": 0.05, ... },
"lines":  { "width_px": 3, ... },
"label":  { "experiment": "ALLEGRO", "status": "Preliminary", ... },
"legend": { "x": 0.04, "y": 0.95, "align": "left", ... }
```

Units:
- sizes are fractions of the canvas height;
- positions are in frame coordinates;
- line widths and marker sizes are in pixels of the 800×600 canvas;
- colours are hex codes.

After editing:

1. Run `python -m allegro_style.root` to regenerate `AllegroStyleConfig.h`, the C++ copy
   of the settings that ROOT reads. The tests fail if you forget. matplotlib reads the
   JSON directly.
2. Run the tests and look at the images in `tests/output/`.

To check new colours for colour-blind safety: `python tests/cvd.py "#0f0f4b" "#1a75bc" ...`.

## Tests

```bash
python -m pytest tests
```

Besides unit tests, this draws the four example plots with both libraries and checks
that they coincide, as PNG and PDF. The files are written to `tests/output/` for visual
inspection: `mpl_*.png|pdf`, `root_*.png|pdf`, and `compare_*_overlay.png`, where red
is ink only in ROOT, blue only in matplotlib and grey in both.

## Behaviour to be aware of (matplotlib)

- `amp.use()` makes matplotlib choose tick steps like ROOT (1, 2 or 5 × 10ⁿ, at most 10
  divisions). `amp.use(root_ticks=False)` keeps matplotlib's default.
- `amp.label(ax)` also puts the axis titles at a fixed distance from the frame, as ROOT
  does. `amp.place_titles(ax)` does only that.
