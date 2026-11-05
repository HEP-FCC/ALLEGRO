// All styles: AllegroLine(obj, i), AllegroPoints(obj, i) and AllegroFill(obj, i) for i = 0..9.
//
// Run:  root -l -b -q --web=off examples/root_styles.C   ->  root_styles.png, root_styles.pdf
// Same chart with matplotlib: examples/mpl_styles.py

#include "../allegro_style/root/AllegroStyle.C"

void root_styles()
{
   SetAllegroStyle();
   auto c = AllegroCanvas();

   auto frame = c->DrawFrame(0, -0.5, 1, 9.5);
   frame->GetXaxis()->SetLabelSize(0);
   frame->GetXaxis()->SetTitle("line(i)            points(i)      fill(i)");
   frame->GetYaxis()->SetTitle("i");

   // One row per number i: a line, a point and a filled box
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

   c->SaveAs("root_styles.png");
   c->SaveAs("root_styles.pdf");
}
