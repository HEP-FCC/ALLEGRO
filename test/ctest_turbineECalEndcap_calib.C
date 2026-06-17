#ifndef ctest_turbineECalEndcap_calib_C
#define ctest_turbineECalEndcap_calib_C
#include "../sampling_fractions/turbineECalEndcap_autoCalib_analytic_topo.C"

void ctest_turbineECalEndcap_calib() {
  TFile f("allegro_v03_ecal_v52_evts_1000_pdg_11_MomentumMinMax_40_40_GeV_ThetaMinMax_5.2_40.0_PhiMinMax_0_6.28_0_digi_reco.root");

  TTree * t = (TTree*) f.Get("events");

  turbineECalEndcap_autoCalib_analytic_topo a(t);

  a.Calibrate();

}


#endif
