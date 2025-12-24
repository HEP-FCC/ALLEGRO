#!/bin/bash

# set-up the Key4hep environment if not already set
if [[ -z "${KEY4HEP_STACK}" ]]; then
  source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh
else
  echo "The Key4hep stack was already loaded in this environment."
fi

#get the reco output file needed for calibration
wget https://fccsw.web.cern.ch/fccsw/filesForSimDigiReco/ALLEGRO/ALLEGRO_o1_v03/forTests/allegro_v03_ecal_v52_evts_1000_pdg_11_MomentumMinMax_40_40_GeV_ThetaMinMax_5.2_40.0_PhiMinMax_0_6.28_0_digi_reco.root

root -l test/ctest_turbineECalEndcap_calib.C -b -q

