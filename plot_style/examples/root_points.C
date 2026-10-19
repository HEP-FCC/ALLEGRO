// Points with error bars, with legend, ALLEGRO label and info.
// Sample 0 also has x error bars (half the bin width on each side).
//
// Run:  root -l -b -q --web=off examples/root_points.C   ->  root_points.png, root_points.pdf
// Same plot with matplotlib: examples/mpl_points.py

#include "../allegro_style/root/AllegroStyle.C"

void root_points()
{
   SetAllegroStyle();         // apply the ALLEGRO style, before making the canvas
   auto c = AllegroCanvas();  // use instead of new TCanvas(...)

   // Example data: three Gaussian-shaped histograms, 40 bins between -4 and 4
   const double height[] = {400, 500, 330}, mean[] = {0.0, 0.5, -0.5}, width[] = {1.0, 0.8, 1.2};
   TH1D *h[3];
   for (int i = 0; i < 3; ++i) {
      h[i] = new TH1D(Form("h%d", i), "", 40, -4, 4);
      for (int bin = 1; bin <= 40; ++bin) {
         double x = h[i]->GetBinCenter(bin);
         double y = height[i] * std::exp(-0.5 * std::pow((x - mean[i]) / width[i], 2));
         h[i]->SetBinContent(bin, y);
         h[i]->SetBinError(bin, std::sqrt(y));
      }
      // AllegroPoints(h, i): colour number i and marker number i, so every dataset looks different
      AllegroPoints(h[i], i);
   }

   h[0]->SetMinimum(0);
   h[0]->SetMaximum(700);
   h[0]->GetXaxis()->SetTitle("#it{E}_{reco}/#it{E}_{true} #minus 1");
   h[0]->GetYaxis()->SetTitle("Events / 0.2");
   h[0]->Draw("E");         // "E": x and y error bars
   h[1]->Draw("E X0 SAME"); // "E X0": y error bars only
   h[2]->Draw("E X0 SAME");

   AllegroLabel();                                    // "ALLEGRO Simulation"
   AllegroInfo("Z #rightarrow q#bar{q}", 91.2, 10.8); // process, sqrt(s) [GeV], lumi [ab^-1]
   auto leg = AllegroLegend(3);                       // legend with 3 entries
   for (int i = 0; i < 3; ++i)
      leg->AddEntry(h[i], Form("Sample %d", i), "pe");
   leg->Draw();

   c->SaveAs("root_points.png");
   c->SaveAs("root_points.pdf");
}
