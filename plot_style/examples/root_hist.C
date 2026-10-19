// Histograms: two stacked, filled backgrounds with two signal lines on top.
//
// Run:  root -l -b -q --web=off examples/root_hist.C   ->  root_hist.png, root_hist.pdf
// Same plot with matplotlib: examples/mpl_hist.py

#include "../allegro_style/root/AllegroStyle.C"

void root_hist()
{
   SetAllegroStyle();
   auto c = AllegroCanvas();

   // Example data: 40 bins of 5 GeV between 0 and 200 GeV
   auto background1 = new TH1D("background1", "", 40, 0, 200);
   auto background2 = new TH1D("background2", "", 40, 0, 200);
   auto signalA = new TH1D("signalA", "", 40, 0, 200);
   auto signalB = new TH1D("signalB", "", 40, 0, 200);
   for (int bin = 1; bin <= 40; ++bin) {
      double x = background1->GetBinCenter(bin);
      background1->SetBinContent(bin, 1900 * std::exp(-x / 50));
      background2->SetBinContent(bin, 700 * std::exp(-0.5 * std::pow((x - 90) / 15, 2)));
      signalA->SetBinContent(bin, 700 * std::exp(-0.5 * std::pow((x - 125) / 5, 2)));
      signalB->SetBinContent(bin, 470 * std::exp(-0.5 * std::pow((x - 150) / 8, 2)));
   }

   // Filled histograms: AllegroFill(h, i) uses colour number i
   AllegroFill(background1, 1);
   AllegroFill(background2, 4);
   // Line histograms: AllegroLine(h, i) uses colour number i and line style number i
   AllegroLine(signalA, 0); // navy, solid
   AllegroLine(signalB, 3); // red, dash-dot

   auto stack = new THStack("stack", ""); // background 2 on top of background 1
   stack->Add(background1);
   stack->Add(background2);
   stack->SetMinimum(0);
   stack->SetMaximum(2700);
   stack->Draw("HIST");
   stack->GetXaxis()->SetTitle("#it{m}_{jj} [GeV]");
   stack->GetYaxis()->SetTitle("Events / 5 GeV");
   signalA->Draw("HIST SAME");
   signalB->Draw("HIST SAME");
   gPad->RedrawAxis(); // ticks on top of the filled areas

   AllegroLabel();
   AllegroInfo("Z #rightarrow q#bar{q}", 91.2, 10.8);
   auto leg = AllegroLegend(4);
   leg->AddEntry(background1, "Background 1", "f");
   leg->AddEntry(background2, "Background 2", "f");
   leg->AddEntry(signalA, "Signal A", "l");
   leg->AddEntry(signalB, "Signal B", "l");
   leg->Draw();

   c->SaveAs("root_hist.png");
   c->SaveAs("root_hist.pdf");
}
