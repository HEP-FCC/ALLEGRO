#!/bin/bash

# set-up the Key4hep environment if not already set
if [[ -z "${KEY4HEP_STACK}" ]]; then
  source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh
else
  echo "The Key4hep stack was already loaded in this environment."
fi

if [ -z "${ALLEGRO+x}" ]; then
    ALLEGRO=../
fi

#get the reco output file needed for calibration
wget https://fccsw.web.cern.ch/fccsw/filesForSimDigiReco/ALLEGRO/ALLEGRO_o1_v03/forTests/allegro_v03_ecal_v52_evts_10_pdg_22_MomentumMinMax_10_10_GeV_ThetaMinMax_5.2_174.8_PhiMinMax_0_6.28_digi_reco.root

root -l $ALLEGRO/test/ctest_turbineECalEndcap_calib.C -b -q

