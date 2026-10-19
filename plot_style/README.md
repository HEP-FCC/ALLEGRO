# ALLEGRO plot style

A common plot style for ALLEGRO, for **matplotlib** and **ROOT**. A plot made with
either library looks the same. All settings are in one file,
[`allegro_style/style.json`](allegro_style/style.json), shared by both.

Contents: [Setup](#setup) · [Quick start](#quick-start) · [Examples](#examples) ·
[Adapting your plots](#adapting-your-plots) · [Saving](#saving) ·
[Function reference](#function-reference) · [Settings](#settings) · [Tests](#tests)

## Setup

```bash
source /cvmfs/sw.hsf.org/key4hep/setup.sh          # or /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh
export PYTHONPATH=/path/to/ALLEGRO/plot_style:$PYTHONPATH  # for Python
```

`/path/to/plot_style` is the folder containing this README. Tested with the Key4hep
release (matplotlib 3.10, ROOT 6.38) and nightlies (matplotlib 3.11, ROOT 6.40).

## Quick start

A plot always has the same steps: apply the style, draw with an ALLEGRO style, add the
label, info and legend, save. Here `counts, edges, x, y, err` and `h`, `hSim`, `hData`
stand for your data; the [examples](#examples) are complete versions.

**matplotlib**

```python
import matplotlib.pyplot as plt
from allegro_style import mpl as amp

amp.use()                                              # apply the style
fig, ax = plt.subplots()

ax.stairs(counts, edges, label="Simulation", **amp.line(0))         # line histogram
ax.errorbar(x, y, yerr=err, linestyle="none", label="Data", **amp.points(1))  # points

amp.label(ax)                                          # "ALLEGRO Simulation"
amp.info(ax, process=r"$Z\to q\bar{q}$", energy=91.2, lumi=10.8)
amp.legend(ax)
fig.savefig("plot.pdf")
```

**ROOT (C++ macro)**, run with `root -l -b -q --web=off plot.C`:

```cpp
#include "/path/to/plot_style/allegro_style/root/AllegroStyle.C"

void plot()
{
   SetAllegroStyle();                   // apply the style
   auto c = AllegroCanvas();            // use instead of new TCanvas(...)

   AllegroLine(hSim, 0);    hSim->Draw("HIST");        // line histogram
   AllegroPoints(hData, 1); hData->Draw("E X0 SAME");  // points

   AllegroLabel();                      // "ALLEGRO Simulation"
   AllegroInfo("Z #rightarrow q#bar{q}", 91.2, 10.8);
   auto leg = AllegroLegend(2);         // legend with 2 entries
   leg->AddEntry(hSim, "Simulation", "l");
   leg->AddEntry(hData, "Data", "pe");
   leg->Draw();
   c->SaveAs("plot.pdf");
}
```

**PyROOT**: the same ROOT functions, loaded from Python.

```python
import ROOT
ROOT.gROOT.SetWebDisplay("off")
from allegro_style import root as aroot

R = aroot.use()                         # load and apply the style
c = R.AllegroCanvas()
R.AllegroLine(h, 0); h.Draw("HIST")
R.AllegroLabel()
R.AllegroInfo("Z #rightarrow q#bar{q}", 91.2, 10.8)
c.SaveAs("plot.pdf")
```

## Examples

Complete, runnable examples are in [`examples/`](examples/), each made once with
matplotlib and once with ROOT. Copy the one closest to your plot.

```bash
python examples/mpl_hist.py                     # writes mpl_hist.png and .pdf
root -l -b -q --web=off examples/root_hist.C    # writes root_hist.png and .pdf
```

| Plot | matplotlib | ROOT |
|---|---|---|
| points with x and y error bars | [`mpl_points.py`](examples/mpl_points.py) | [`root_points.C`](examples/root_points.C) |
| stacked and line histograms | [`mpl_hist.py`](examples/mpl_hist.py) | [`root_hist.C`](examples/root_hist.C) |
| 2D colour map | [`mpl_colz.py`](examples/mpl_colz.py) | [`root_colz.C`](examples/root_colz.C) |
| all colours, markers and line styles | [`mpl_styles.py`](examples/mpl_styles.py) | [`root_styles.C`](examples/root_styles.C) |

## Adapting your plots

### Points, lines and fills

| Drawing | matplotlib | ROOT |
|---|---|---|
| points | `**amp.points(i)` | `AllegroPoints(h, i)`, draw with `"E X0"` or `"E"` (see [Error bars](#error-bars)) |
| line histogram or curve | `**amp.line(i)` | `AllegroLine(h, i)`, draw with `"HIST"` |
| filled area | `**amp.fill(i)` | `AllegroFill(h, i)`, draw with `"HIST"` |

In matplotlib, put `**amp.points(i)` etc. inside any plotting call (`errorbar`,
`plot`, `stairs`, `hist`, `fill_between`…). In ROOT, the functions work on
histograms, graphs, functions, `TLine`, `TMarker` and `TBox`.

### Error bars

matplotlib: pass `yerr=` and/or `xerr=` to `ax.errorbar`. Each can be one number, one
value per point, or `[lower, upper]` for asymmetric errors.

```python
ax.errorbar(x, y, yerr=err, linestyle="none", **amp.points(0))             # y errors
ax.errorbar(x, y, xerr=0.1, yerr=err, linestyle="none", **amp.points(0))   # x and y errors
```

ROOT histograms: the draw option chooses.

```cpp
h->Draw("E X0");   // y errors only
h->Draw("E");      // x errors (half the bin width on each side) and y errors
```

For other x errors, use a `TGraphErrors` (or `TGraphAsymmErrors` for asymmetric ones):

```cpp
auto g = new TGraphErrors(n, x, y, ex, ey);
AllegroPoints(g, 0);
g->Draw("P");      // "AP" if it is the first thing drawn on the canvas
```

The *points* example shows both: sample 0 with x and y errors, the others with y
errors only.

### Colours, markers and line styles

The number `i` picks the colour. For points and lines it also picks the marker or line
style, so datasets drawn with 0, 1, 2… differ in colour *and* shape. Give a second
number to choose the marker or line style yourself:

| Number | Colour | Marker | Line style |
|---|---|---|---|
| 0 | navy | ● filled circle | solid |
| 1 | blue | ■ filled square | dashed |
| 2 | yellow | ▲ filled triangle | dotted |
| 3 | red | ▼ filled triangle down | dash-dot |
| 4 | light cyan | ◆ filled diamond | |
| 5 | brown | ○ open circle | |
| 6–9 | | □ △ ▽ ◇ open shapes | |

After the last entry a list starts again (colour 6 = navy). The *All styles* example
shows them all.

```python
amp.points(2)               # yellow, filled triangle
amp.points(2, marker=5)     # yellow, open circle
amp.line(3)                 # red, dash-dot
amp.line(3, linestyle=0)    # red, solid
amp.fill(4)                 # light cyan
```
```cpp
AllegroPoints(h, 2);        // yellow, filled triangle
AllegroPoints(h, 2, 5);     // yellow, open circle
AllegroLine(h, 3);          // red, dash-dot
AllegroLine(h, 3, 0);       // red, solid
AllegroFill(h, 4);          // light cyan
```

Yellow is hard to see as a thin line on white; prefer it for fills and markers.

### Other options (transparency, width…)

```python
ax.stairs(counts, edges, fill=True, **amp.fill(1, alpha=0.5))   # half-transparent
ax.plot(x, y, **amp.line(0, linewidth=3))                       # thicker line
```
```cpp
AllegroFill(h, 1); h->SetFillColorAlpha(AllegroColor(1), 0.5);  // half-transparent
AllegroLine(g, 0); g->SetLineWidth(3);                          // thicker line
```

In ROOT, `AllegroColor(n)`, `AllegroMarker(n)` and `AllegroLineStyle(n)` give single
values for your own objects. In Python, `amp.line(n)["color"]` gives colour `n`.

### Label, info and legend

Default positions:

```
 ALLEGRO Simulation                       <- label
+----------------------------------------+
| Z -> qq                    • Sample 0  |  <- info (top left), legend (top right)
| √s = 91.2 GeV, 10.8 ab^-1  ■ Sample 1  |
|                                        |
+----------------------------------------+
```

Positions are in **frame coordinates**: (0, 0) is the bottom-left corner of the frame,
(1, 1) the top-right one. Values above 1 are above the frame.

The arguments have the same names in matplotlib and ROOT. In matplotlib, give only the
ones you need (`amp.info(ax, energy=240)`). In ROOT, give them in the order of the
table; to skip one, use the "none" value shown, or `kAllegroDefault` for positions.

**`amp.label(ax, ...)`, `AllegroLabel(...)`**: "ALLEGRO Simulation"

| Argument | Default | Meaning |
|---|---|---|
| `status` | `"Simulation"` | Text after "ALLEGRO", e.g. `"Preliminary"`; `""` for "ALLEGRO" alone. |
| `x`, `y` | 0, 1.012 | Position: just above the top-left corner of the frame. |
| `align` | `"left"` | Which end of the text is at `x`: `"left"` or `"right"`. |

**`amp.info(ax, ...)`, `AllegroInfo(...)`**: process, √s and luminosity

| Argument | Default | Meaning |
|---|---|---|
| `process` | none (`""`) | First line. `"\n"` starts a new line. matplotlib: `r"$Z\to q\bar{q}$"`; ROOT: `"Z #rightarrow q#bar{q}"`. |
| `energy` | none (`-1`) | √s in GeV: writes "√s = 91.2 GeV". |
| `lumi` | none (`-1`) | Luminosity in ab⁻¹: adds ", 10.8 ab⁻¹". |
| `x`, `y` | 0.04, 0.95 | Position: top left, inside the frame. |
| `align` | `"left"` | Which end of the text is at `x`: `"left"` or `"right"`. |
| `valign` | `"top"` | `"top"`: `y` is the top of the first line. `"bottom"`: `y` is the bottom of the last line. |

So `AllegroInfo("Z #rightarrow q#bar{q}", 91.2, 10.8)` writes two lines,
"Z → qq̄" and "√s = 91.2 GeV, 10.8 ab⁻¹". Whatever you leave out is not written.

**`amp.legend(ax, ...)`, `AllegroLegend(n, ...)`**

| Argument | Default | Meaning |
|---|---|---|
| `n` | | ROOT only: number of entries, which sets the legend height. |
| `x`, `y` | 0.96, 0.95 | Position of the legend's top corner. |
| `align` | `"right"` | `"right"`: (x, y) is the top-right corner; `"left"`: the top-left one. |

In matplotlib the entries are the `label=` of what you drew, and extra arguments go to
`ax.legend` (e.g. `ncols=2`). In ROOT add them with `leg->AddEntry(h, "text", "pe")`
(`"pe"` point with error, `"l"` line, `"f"` fill), then `leg->Draw()`.

Examples:

```python
amp.label(ax, status="Preliminary")                                     # "ALLEGRO Preliminary"
amp.info(ax, energy=240)                                                # only "√s = 240 GeV"
amp.info(ax, process="ZH\nbarrel only", energy=240, lumi=10.8)          # three lines
amp.info(ax, energy=91.2, x=1, y=1.012, align="right", valign="bottom") # on the label's line, right
amp.legend(ax, x=0.04, y=0.75, align="left")                            # left, below the info
```
```cpp
AllegroLabel("Preliminary");
AllegroInfo("", 240);
AllegroInfo("ZH\nbarrel only", 240, 10.8);
AllegroInfo("", 91.2, -1, 1, 1.012, "right", "bottom");
AllegroLegend(3, 0.04, 0.75, "left");
AllegroLegend(3, kAllegroDefault, 0.6);                                 // default x, lower
```

To change the defaults for every plot, edit `label`, `info` and `legend` in
`style.json` (see [Settings](#settings)).

### 2D plots

- matplotlib: after `mesh = ax.pcolormesh(...)`, call `amp.colorbar(mesh, ax, label="Events")`.
  Call it before `amp.label` and `amp.info`.
- ROOT: create the canvas with `AllegroCanvas("c", "", true)` to leave room for the
  colour scale, then `h2->Draw("COLZ")`.

For 3D plots the font and colours apply, but the layout differs between the two
libraries.

## Saving

Save as PNG or PDF as usual: `fig.savefig("plot.pdf")`, `c->SaveAs("plot.pdf")`.

With ROOT 6.38 or newer, on machines without a web browser (lxplus, batch jobs), saving
fails unless you run `root --web=off`, or call `ROOT.gROOT.SetWebDisplay("off")` in
PyROOT.

ROOT PDFs don't embed the font. Every viewer shows it correctly, but if a journal
requires embedded fonts, use
`gs -o out.pdf -sDEVICE=pdfwrite -dEmbedAllFonts=true plot.pdf`.

## Function reference

`x`, `y`, `align` default to the positions in `style.json`.

**matplotlib** (`from allegro_style import mpl as amp`)

| Function | |
|---|---|
| `amp.use()` | Apply the style. Call once, before making figures. |
| `amp.points(i, marker=i, **options)` | Style for points. |
| `amp.line(i, linestyle=i, **options)` | Style for lines. |
| `amp.fill(i, **options)` | Style for filled areas. |
| `amp.label(ax, status="Simulation", x, y, align)` | "ALLEGRO Simulation". |
| `amp.info(ax, process, energy, lumi, x, y, align, valign)` | Process, √s [GeV], luminosity [ab⁻¹]. |
| `amp.legend(ax, x, y, align, **options)` | Legend. |
| `amp.colorbar(mesh, ax, label)` | Colour scale of a 2D plot. |

`**options` are matplotlib options (e.g. `alpha=0.5`); they override the style.

**ROOT** (`#include ".../AllegroStyle.C"`, or `R = aroot.use()` in PyROOT)

| Function | |
|---|---|
| `SetAllegroStyle()` | Apply the style. Call once, before making canvases. |
| `AllegroCanvas(name="c", title="", colz=false)` | 800×600 canvas; `colz=true` for 2D plots. |
| `AllegroPoints(obj, i, marker=i)` | Style for points. |
| `AllegroLine(obj, i, linestyle=i)` | Style for lines. |
| `AllegroFill(obj, i)` | Style for filled areas. |
| `AllegroLabel(status="Simulation", x, y, align)` | "ALLEGRO Simulation". |
| `AllegroInfo(process="", energy=-1, lumi=-1, x, y, align, valign)` | Process, √s [GeV], luminosity [ab⁻¹]; `-1` = leave out. |
| `AllegroLegend(n, x, y, align)` | `TLegend` for `n` entries; then `AddEntry` and `Draw`. |
| `AllegroColor(n)`, `AllegroMarker(n)`, `AllegroLineStyle(n)` | Single colour, marker or line style. |

## Settings

There are two ways to change how plots look:

- **For one plot**, in your script, with the function arguments described in
  [Adapting your plots](#adapting-your-plots): colours, markers, line styles, label
  text, positions…
- **For every plot**, by editing [`allegro_style/style.json`](allegro_style/style.json).
  Both matplotlib and ROOT read this file.

### Editing `style.json`

For example, to make "ALLEGRO Preliminary" the default, move the legend to the left
below the info, and use thicker lines, change these entries and leave the others as
they are:

```json
"label":  { "status": "Preliminary", ... },
"legend": { "x": 0.04, "y": 0.75, "align": "left", ... },
"lines":  { "width_px": 3, ... }
```

Then run, from this folder:

```bash
python -m allegro_style.root      # copy the new settings to ROOT (AllegroStyleConfig.h)
python -m pytest tests            # check that matplotlib and ROOT still agree
```

The first command is needed for ROOT only; matplotlib reads the JSON directly. If you
forget it, the tests tell you.

### What each setting does

Units used below:
- **fraction of height**: fraction of the canvas height, e.g. 0.05 = 30 px of 600;
- **frame coordinates**: (0, 0) is the bottom-left corner of the frame, (1, 1) the top-right;
- **px**: pixels of the 800×600 canvas.

**`canvas`**

| Setting | Default | Meaning |
|---|---|---|
| `width_px`, `height_px` | 800, 600 | Size of the plot. |
| `dpi` | 100 | matplotlib only: pixels per inch. Keep it; it doesn't change the look. |

**`margins`** (space between the canvas edge and the frame)

| Setting | Default | Meaning |
|---|---|---|
| `left`, `right`, `bottom`, `top` | 0.14, 0.05, 0.14, 0.08 | Fraction of the canvas width (left, right) or height (bottom, top). |
| `right_colz` | 0.16 | Right margin of 2D plots, to leave room for the colour scale. |

**`font`**

| Setting | Default | Meaning |
|---|---|---|
| `family` | TeX Gyre Heros | matplotlib font. It is the font ROOT uses, so keep it to match ROOT. |
| `fallbacks` | Nimbus Sans, … | matplotlib fonts used if `family` is not installed. |
| `root_regular` | 42 | ROOT font code (42 = Helvetica-like, sizes as fractions). |

**`text`** (sizes, as fraction of height)

| Setting | Default | Meaning |
|---|---|---|
| `axis_title_size` | 0.045 | Axis titles. |
| `axis_label_size` | 0.04 | Numbers on the axes. |
| `legend_size` | 0.035 | Legend text. |
| `label_size` | 0.045 | "ALLEGRO Simulation". |
| `info_size` | 0.035 | Process, √s and luminosity. |

**`axes`**

| Setting | Default | Meaning |
|---|---|---|
| `ticks_all_sides` | true | Ticks also on the top and right edges of the frame. |
| `minor_ticks` | true | Show minor ticks. |
| `tick_length` | 0.03 | Length of major ticks, fraction of height. |
| `ndivisions` | 510 | ROOT code: up to 10 major divisions, each with 5 minor ones. |
| `frame_width_px` | 1 | Width of the frame line. |
| `title_offset_x`, `title_offset_y` | 0.98, 1.33 | Distance of the axis titles from the axes, in ROOT units (larger = further). |
| `minor_tick_length`, `x_title_loc`, `y_title_loc` | 0.015, right, top | matplotlib only. Keep them: ROOT always uses half-length minor ticks and titles at the axis ends. |

**`lines`**

| Setting | Default | Meaning |
|---|---|---|
| `width_px` | 2 | Width of lines (`line`, error bars). |
| `marker_size_px` | 9 | Size of markers (`points`). |

**`palette`**, **`markers`**, **`line_styles`** (the numbered lists used by `points`,
`line` and `fill`)

| Setting | Meaning |
|---|---|
| `palette.colors` | Colours as hex codes, in order: number 0, 1, 2… |
| `markers.root` | ROOT marker codes (20 = filled circle, 24 = open circle, …). |
| `markers.mpl` | The same markers in matplotlib (`"o"` circle, `"s"` square, `"^"` triangle, …). |
| `markers.filled` | `true` for filled, `false` for open markers. |
| `markers.mpl_scale` | Size correction for matplotlib (diamonds look bigger there; 0.75 matches ROOT). |
| `line_styles.root` | ROOT line style codes (1 solid, 2 dashed, 3 dotted, 4 dash-dot). |
| `line_styles.dashes_root` | Dash pattern of each style: lengths of dash and gap, repeated. `""` = solid. |

The four `markers` lists must have the same length and describe the same markers.
New colours can be checked for colour-blind safety with
`python tests/cvd.py "#0f0f4b" "#1a75bc" ...` (the tests also check them).

**`colormap`** (2D plots)

| Setting | Meaning |
|---|---|
| `stops` | Colours from the lowest to the highest value; the map blends between them. |
| `name` | Just a name for the list. |

**`label`** ("ALLEGRO Simulation")

| Setting | Default | Meaning |
|---|---|---|
| `experiment` | ALLEGRO | First word, in bold italic. |
| `status` | Simulation | Second word, e.g. Preliminary, Internal. |
| `x`, `y` | 0, 1.012 | Position in frame coordinates: just above the top-left corner. |
| `align` | left | Which end of the text is at `x`: `left` or `right`. |

**`info`** (process, √s, luminosity)

| Setting | Default | Meaning |
|---|---|---|
| `x`, `y` | 0.04, 0.95 | Position in frame coordinates. |
| `align` | left | Which end of the text is at `x`: `left` or `right`. |
| `valign` | top | `top`: `y` is the top of the first line; `bottom`: the bottom of the last line. |
| `line_spacing` | 1.35 | Distance between lines, in units of `info_size`. |
| `energy_unit` | GeV | Unit written after √s. |
| `lumi_unit_mpl`, `lumi_unit_root` | ab⁻¹ | Luminosity unit, written for matplotlib and for ROOT. |

**`legend`**

| Setting | Default | Meaning |
|---|---|---|
| `x`, `y` | 0.96, 0.95 | Position of the top corner, in frame coordinates. |
| `align` | right | Which corner is at `x`: `right` (top-right) or `left` (top-left). |
| `width` | 0.3 | Width of the legend, fraction of the frame width. |
| `row_height` | 0.05 | Height of one entry, fraction of height. |
| `margin` | 0.25 | Part of the legend width used for the symbols. |

## Tests

```bash
python -m pytest tests
```

The tests run every example with both libraries and check that the matplotlib and ROOT
plots coincide. The plots are written to `tests/output/`. There,
`compare_<plot>_overlay.png` shows both on top of each other: red is ink only in ROOT,
blue only in matplotlib, grey in both.


*This documentation was in part generated with AI and reviewed by a human.*
