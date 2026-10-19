// ALLEGRO plotting style for ROOT.
//
//   gROOT->LoadMacro("<path>/allegro_style/root/AllegroStyle.C");  // or #include it
//   SetAllegroStyle();
//   ... draw ...
//   AllegroLabel();                                   // "ALLEGRO Simulation" above the frame
//   AllegroInfo("Z #rightarrow q#bar{q}", 91.2, 10.8); // process, sqrt(s) [GeV], lumi [ab^-1]
//   auto leg = AllegroLegend(3); leg->AddEntry(h, "Sample", "pe"); leg->Draw();
//
// Positions are in frame coordinates ((0,0) bottom-left, (1,1) top-right of the
// axes frame); defaults come from style.json via AllegroStyleConfig.h.
#ifndef ALLEGRO_STYLE_C
#define ALLEGRO_STYLE_C

#include "AllegroStyleConfig.h"

#include <cstring>
#include <string>
#include <vector>

#include "TAttFill.h"
#include "TAttLine.h"
#include "TAttMarker.h"
#include "TCanvas.h"
#include "TColor.h"
#include "TLatex.h"
#include "TLegend.h"
#include "TPad.h"
#include "TROOT.h"
#include "TString.h"
#include "TStyle.h"

// Colour map for COLZ etc., built from the same stops as the matplotlib one.
inline void AllegroColormap(int ncontours = 255)
{
   using namespace AllegroCfg;
   const int n = colormap_stops_n;
   double pos[n], r[n], g[n], b[n];
   for (int k = 0; k < n; ++k) {
      TColor *c = gROOT->GetColor(TColor::GetColor(colormap_stops[k]));
      pos[k] = double(k) / (n - 1);
      r[k] = c->GetRed();
      g[k] = c->GetGreen();
      b[k] = c->GetBlue();
   }
   TColor::CreateGradientColorTable(n, pos, r, g, b, ncontours);
   gStyle->SetNumberContours(ncontours);
}

inline void SetAllegroStyle()
{
   using namespace AllegroCfg;
   TStyle *s = gROOT->GetStyle("ALLEGRO");
   if (!s)
      s = new TStyle("ALLEGRO", "ALLEGRO style");

   s->SetCanvasDefW(canvas_width_px);
   s->SetCanvasDefH(canvas_height_px);
   s->SetCanvasColor(0);
   s->SetPadColor(0);
   s->SetFrameFillColor(0);
   s->SetStatColor(0);
   s->SetCanvasBorderMode(0);
   s->SetPadBorderMode(0);
   s->SetFrameBorderMode(0);

   s->SetPadLeftMargin(margins_left);
   s->SetPadRightMargin(margins_right);
   s->SetPadBottomMargin(margins_bottom);
   s->SetPadTopMargin(margins_top);

   s->SetTextFont(font_root_regular);
   s->SetLabelFont(font_root_regular, "XYZ");
   s->SetTitleFont(font_root_regular, "XYZ");
   s->SetLegendFont(font_root_regular);
   s->SetTextSize(text_axis_label_size);
   s->SetLabelSize(text_axis_label_size, "XYZ");
   s->SetTitleSize(text_axis_title_size, "XYZ");
   s->SetLegendTextSize(text_legend_size);
   s->SetTitleXOffset(axes_title_offset_x);
   s->SetTitleYOffset(axes_title_offset_y);

   s->SetTickLength(axes_tick_length, "XYZ");
   s->SetNdivisions(axes_minor_ticks ? axes_ndivisions : axes_ndivisions % 100, "XYZ");
   s->SetPadTickX(axes_ticks_all_sides);
   s->SetPadTickY(axes_ticks_all_sides);
   s->SetFrameLineWidth(axes_frame_width_px);

   s->SetHistLineWidth(lines_width_px);
   s->SetFuncWidth(lines_width_px);
   s->SetMarkerStyle(markers_root[0]);
   s->SetMarkerSize(lines_marker_size_px / 8.); // ROOT marker size 1 = 8 px
   s->SetEndErrorSize(0);
   s->SetErrorX(0.5); // histogram x errors = half bin width: "E" draws them, "E X0" hides them
   for (int k = 0; k < line_styles_root_n; ++k)
      if (*line_styles_dashes_root[k])
         s->SetLineStyleString(line_styles_root[k], line_styles_dashes_root[k]);

   s->SetOptStat(0);
   s->SetOptTitle(0);
   s->SetOptFit(0);
   s->SetLegendBorderSize(0);
   s->SetLegendFillColor(0);
   gROOT->SetStyle("ALLEGRO");
   gROOT->ForceStyle();
   AllegroColormap();
}

// TCanvas(name, title, w, h) subtracts window decorations; this gives exactly w x h pixels.
// colz = true leaves room on the right for the COLZ colour scale.
inline TCanvas *AllegroCanvas(const char *name = "c", const char *title = "", bool colz = false)
{
   auto c = new TCanvas(name, title, AllegroCfg::canvas_width_px, AllegroCfg::canvas_height_px);
   c->SetCanvasSize(AllegroCfg::canvas_width_px, AllegroCfg::canvas_height_px);
   if (colz)
      c->SetRightMargin(AllegroCfg::margins_right_colz);
   return c;
}

inline int AllegroColor(int i) { return TColor::GetColor(AllegroCfg::palette_colors[i % AllegroCfg::palette_colors_n]); }
inline int AllegroMarker(int i) { return AllegroCfg::markers_root[i % AllegroCfg::markers_root_n]; }
inline int AllegroLineStyle(int i) { return AllegroCfg::line_styles_root[i % AllegroCfg::line_styles_root_n]; }

// Styles for any TH1 / TGraph / TF1 / TLine / TMarker / TBox ...
// The number i is the colour; for points and lines it also picks the marker / line
// style, unless a second number is given.

// Points (draw with "E X0" or "P"): colour i, marker `marker` (default i).
inline void AllegroPoints(TObject *o, int i, int marker = -1)
{
   int c = AllegroColor(i);
   if (auto l = dynamic_cast<TAttLine *>(o)) { // error bars
      l->SetLineColor(c);
      l->SetLineWidth(AllegroCfg::lines_width_px);
   }
   if (auto m = dynamic_cast<TAttMarker *>(o)) {
      m->SetMarkerColor(c);
      m->SetMarkerStyle(AllegroMarker(marker < 0 ? i : marker));
      m->SetMarkerSize(AllegroCfg::lines_marker_size_px / 8.);
   }
}

// Line without markers (draw with "HIST" or "L"): colour i, line style `linestyle` (default i).
inline void AllegroLine(TObject *o, int i, int linestyle = -1)
{
   if (auto l = dynamic_cast<TAttLine *>(o)) {
      l->SetLineColor(AllegroColor(i));
      l->SetLineStyle(AllegroLineStyle(linestyle < 0 ? i : linestyle));
      l->SetLineWidth(AllegroCfg::lines_width_px);
   }
}

// Filled area (draw with "HIST", e.g. in a THStack): colour i.
inline void AllegroFill(TObject *o, int i)
{
   int c = AllegroColor(i);
   if (auto l = dynamic_cast<TAttLine *>(o))
      l->SetLineColor(c);
   if (auto f = dynamic_cast<TAttFill *>(o)) {
      f->SetFillColor(c);
      f->SetFillStyle(1001);
   }
}

inline double AllegroNdcX(double x) { return gPad->GetLeftMargin() + x * (1 - gPad->GetLeftMargin() - gPad->GetRightMargin()); }
inline double AllegroNdcY(double y) { return gPad->GetBottomMargin() + y * (1 - gPad->GetBottomMargin() - gPad->GetTopMargin()); }
inline bool AllegroIsLeft(const char *align) { return std::strcmp(align, "right") != 0; }

const double kAllegroDefault = -999;

// "ALLEGRO Simulation"; (x, y) is the bottom-left (or bottom-right) corner.
inline TLatex *AllegroLabel(const char *status = nullptr, double x = kAllegroDefault, double y = kAllegroDefault,
                            const char *align = nullptr)
{
   using namespace AllegroCfg;
   if (!status) status = label_status;
   if (x == kAllegroDefault) x = label_x;
   if (y == kAllegroDefault) y = label_y;
   if (!align) align = label_align;
   std::string txt = std::string("#bf{#it{") + label_experiment + "}}";
   if (*status) txt += std::string(" ") + status;
   auto t = new TLatex(AllegroNdcX(x), AllegroNdcY(y), txt.c_str());
   t->SetNDC();
   t->SetTextFont(font_root_regular);
   t->SetTextSize(text_label_size);
   t->SetTextAlign(AllegroIsLeft(align) ? 11 : 31);
   t->Draw();
   return t;
}

// Process (lines separated by '\n'), then sqrt(s) [GeV] and luminosity [ab^-1] (skipped if < 0).
// valign "top": (x, y) is the top of the first line; "bottom": the baseline of the last line
// (use y = label y to put the info on the same line as "ALLEGRO Simulation").
inline void AllegroInfo(const char *process = "", double energy = -1, double lumi = -1, double x = kAllegroDefault,
                        double y = kAllegroDefault, const char *align = nullptr, const char *valign = nullptr)
{
   using namespace AllegroCfg;
   if (x == kAllegroDefault) x = info_x;
   if (y == kAllegroDefault) y = info_y;
   if (!align) align = info_align;
   if (!valign) valign = info_valign;
   bool top = std::strcmp(valign, "bottom") != 0;

   TString lines = process;
   TString sub;
   if (energy >= 0)
      sub += TString::Format("#sqrt{#it{s}} = %g %s", energy, info_energy_unit);
   if (lumi >= 0)
      sub += TString::Format("%s%g %s", sub.IsNull() ? "" : ", ", lumi, info_lumi_unit_root);
   if (!sub.IsNull())
      lines += (lines.IsNull() ? "" : "\n") + sub;

   std::vector<std::string> split;
   for (const char *line = lines.Data(); *line;) {
      const char *end = std::strchr(line, '\n');
      split.emplace_back(end ? std::string(line, end) : std::string(line));
      line = end ? end + 1 : line + split.back().size();
   }
   const double step = text_info_size * info_line_spacing, ndcY = AllegroNdcY(y);
   const int n = split.size();
   for (int k = 0; k < n; ++k) {
      auto t = new TLatex(AllegroNdcX(x), top ? ndcY - k * step : ndcY + (n - 1 - k) * step, split[k].c_str());
      t->SetNDC();
      t->SetTextFont(font_root_regular);
      t->SetTextSize(text_info_size);
      t->SetTextAlign((AllegroIsLeft(align) ? 10 : 30) + (top ? 3 : 1));
      t->Draw();
   }
}

// Legend for n entries with its top-right (or top-left) corner at (x, y).
inline TLegend *AllegroLegend(int n, double x = kAllegroDefault, double y = kAllegroDefault, const char *align = nullptr)
{
   using namespace AllegroCfg;
   if (x == kAllegroDefault) x = legend_x;
   if (y == kAllegroDefault) y = legend_y;
   if (!align) align = legend_align;
   double frameW = 1 - gPad->GetLeftMargin() - gPad->GetRightMargin();
   double w = legend_width * frameW;
   double x0 = AllegroIsLeft(align) ? AllegroNdcX(x) : AllegroNdcX(x) - w;
   double y1 = AllegroNdcY(y);
   auto leg = new TLegend(x0, y1 - n * legend_row_height, x0 + w, y1);
   leg->SetBorderSize(0);
   leg->SetFillStyle(0);
   leg->SetTextFont(font_root_regular);
   leg->SetTextSize(text_legend_size);
   leg->SetMargin(legend_margin);
   return leg;
}

#endif
