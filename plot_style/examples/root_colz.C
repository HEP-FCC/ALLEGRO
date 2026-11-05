// 2D colour map with a colour scale.
//
// Run:  root -l -b -q --web=off examples/root_colz.C   ->  root_colz.png, root_colz.pdf
// Same plot with matplotlib: examples/mpl_colz.py

#include "../allegro_style/root/AllegroStyle.C"

void root_colz()
{
   SetAllegroStyle();
   auto c = AllegroCanvas("c", "", true); // true: leave room for the colour scale

   // Example data: 40 x 40 bins between -3 and 3, a tilted 2D Gaussian
   auto h = new TH2D("h", "", 40, -3, 3, 40, -3, 3);
   for (int i = 1; i <= 40; ++i) {
      for (int j = 1; j <= 40; ++j) {
         double x = h->GetXaxis()->GetBinCenter(i);
         double y = h->GetYaxis()->GetBinCenter(j);
         h->SetBinContent(i, j, 1000 * std::exp(-0.5 * (x * x + (y - 0.5 * x) * (y - 0.5 * x) / 0.5)));
      }
   }

   h->SetMinimum(0);
   h->GetXaxis()->SetTitle("#it{#theta} [rad]");
   h->GetYaxis()->SetTitle("#it{#phi} [rad]");
   h->GetZaxis()->SetTitle("Events");
   h->Draw("COLZ"); // the colour map comes from the style

   AllegroLabel();
   // The frame is full, so put the info on the label's line, at the right.
   // Arguments: process, sqrt(s), lumi (-1 = none), x, y, align, valign
   AllegroInfo("", 91.2, -1, 1, 1.012, "right", "bottom");

   c->SaveAs("root_colz.png");
   c->SaveAs("root_colz.pdf");
}
