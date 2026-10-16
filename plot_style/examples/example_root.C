// Same plots as example_mpl.py, drawn with ROOT (plain C++, no Python needed).
//   root -l -b -q --web=off 'examples/example_root.C("<outdir>", "png")'   (or "pdf")
//   ->  root_{points,hist,colz,styles}.png
#include "../allegro_style/root/AllegroStyle.C"

#include "TH1D.h"
#include "TBox.h"
#include "TH2D.h"
#include "THStack.h"
#include "TLine.h"
#include "TMarker.h"
#include "TMath.h"

double Gaus(double x, double n, double mu, double sigma, double w) { return n * w * TMath::Gaus(x, mu, sigma, true); }

void points(TString out)
{
   auto c = AllegroCanvas("c_points");
   const double mu[] = {0.0, 0.5, -0.5}, sigma[] = {1.0, 0.8, 1.2};
   TH1D *h[3];
   double peak = 0;
   for (int i = 0; i < 3; ++i) {
      h[i] = new TH1D(Form("h%d", i), "", 40, -4, 4);
      for (int b = 1; b <= 40; ++b) {
         double y = Gaus(h[i]->GetBinCenter(b), 5000, mu[i], sigma[i], 0.2);
         h[i]->SetBinContent(b, y);
         h[i]->SetBinError(b, std::sqrt(y));
         peak = std::max(peak, y);
      }
      AllegroPoints(h[i], i);
   }
   h[0]->SetMinimum(0);
   h[0]->SetMaximum(1.4 * peak);
   h[0]->GetXaxis()->SetTitle("#it{E}_{reco}/#it{E}_{true} #minus 1");
   h[0]->GetYaxis()->SetTitle("Events / 0.2");
   for (int i = 0; i < 3; ++i)
      h[i]->Draw(i ? "E X0 SAME" : "E X0");
   AllegroLabel();
   AllegroInfo("Z #rightarrow q#bar{q}", 91.2, 10.8);
   auto leg = AllegroLegend(3);
   for (int i = 0; i < 3; ++i)
      leg->AddEntry(h[i], Form("Sample %d", i), "pe");
   leg->Draw();
   c->SaveAs(out);
}

void hist(TString out)
{
   auto c = AllegroCanvas("c_hist");
   auto bkg1 = new TH1D("bkg1", "", 40, 0, 200), bkg2 = new TH1D("bkg2", "", 40, 0, 200);
   auto sigA = new TH1D("sigA", "", 40, 0, 200), sigB = new TH1D("sigB", "", 40, 0, 200);
   for (int b = 1; b <= 40; ++b) {
      double x = bkg1->GetBinCenter(b);
      bkg1->SetBinContent(b, 20000 * 5 * std::exp(-x / 50) / 50);
      bkg2->SetBinContent(b, Gaus(x, 5000, 91, 15, 5));
      sigA->SetBinContent(b, Gaus(x, 2000, 125, 5, 5));
      sigB->SetBinContent(b, Gaus(x, 2000, 150, 8, 5));
   }
   AllegroFill(bkg1, 1);
   AllegroFill(bkg2, 4);
   AllegroLine(sigA, 0);
   AllegroLine(sigB, 3);

   auto stack = new THStack("stack", "");
   stack->Add(bkg1);
   stack->Add(bkg2);
   double peak = 0;
   for (int b = 1; b <= 40; ++b)
      peak = std::max(peak, bkg1->GetBinContent(b) + bkg2->GetBinContent(b));
   stack->SetMinimum(0);
   stack->SetMaximum(1.4 * peak);
   stack->Draw("HIST");
   stack->GetXaxis()->SetTitle("#it{m}_{jj} [GeV]");
   stack->GetYaxis()->SetTitle("Events / 5 GeV");
   sigA->Draw("HIST SAME");
   sigB->Draw("HIST SAME");
   gPad->RedrawAxis();

   AllegroLabel();
   AllegroInfo("Z #rightarrow q#bar{q}", 91.2, 10.8);
   auto leg = AllegroLegend(4);
   leg->AddEntry(bkg1, "Background 1", "f");
   leg->AddEntry(bkg2, "Background 2", "f");
   leg->AddEntry(sigA, "Signal A", "l");
   leg->AddEntry(sigB, "Signal B", "l");
   leg->Draw();
   c->SaveAs(out);
}

void colz(TString out)
{
   auto c = AllegroCanvas("c_colz", "", true);
   auto h = new TH2D("h_colz", "", 40, -3, 3, 40, -3, 3);
   for (int i = 1; i <= 40; ++i)
      for (int j = 1; j <= 40; ++j) {
         double x = h->GetXaxis()->GetBinCenter(i), y = h->GetYaxis()->GetBinCenter(j);
         h->SetBinContent(i, j, 1000 * std::exp(-0.5 * (x * x + (y - 0.5 * x) * (y - 0.5 * x) / 0.5)));
      }
   h->SetMinimum(0);
   h->GetXaxis()->SetTitle("#it{#theta} [rad]");
   h->GetYaxis()->SetTitle("#it{#phi} [rad]");
   h->GetZaxis()->SetTitle("Events");
   h->Draw("COLZ");
   AllegroLabel();
   AllegroInfo("", 91.2, -1, 1, 1.012, "right", "bottom");
   c->SaveAs(out);
}

// Reference chart: AllegroLine(obj, i), AllegroPoints(obj, i) and AllegroFill(obj, i) for i = 0..9.
void styles(TString out)
{
   auto c = AllegroCanvas("c_styles");
   auto frame = c->DrawFrame(0, -0.5, 1, 9.5);
   frame->GetXaxis()->SetLabelSize(0);
   frame->GetXaxis()->SetTitle("line(i)            points(i)      fill(i)");
   frame->GetYaxis()->SetTitle("i");
   for (int i = 0; i < 10; ++i) {
      auto line = new TLine(0.05, i, 0.55, i);
      AllegroLine(line, i);
      line->Draw();
      auto marker = new TMarker(0.7, i, 1);
      AllegroPoints(marker, i);
      marker->Draw();
      auto box = new TBox(0.82, i - 0.3, 0.95, i + 0.3);
      AllegroFill(box, i);
      box->Draw();
   }
   gPad->RedrawAxis();
   AllegroLabel();
   c->SaveAs(out);
}

void example_root(TString outdir = ".", TString ext = "png")
{
   SetAllegroStyle();
   points(outdir + "/root_points." + ext);
   hist(outdir + "/root_hist." + ext);
   colz(outdir + "/root_colz." + ext);
   styles(outdir + "/root_styles." + ext);
}
