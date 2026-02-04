#!/bin/bash

# set-up the Key4hep environment if not already set
if [[ -z "${KEY4HEP_STACK}" ]]; then
  source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh
else
  echo "The Key4hep stack was already loaded in this environment."
fi
printenv

# create the neighbour map
k4run neighbor_maps/neighbours.py --ecalb --ecalec --hcalb --hcalec --link-calos --link-hcal

# for debug: compare to the one in the repository
echo "Comparing new map to reference one. If the test fails, you might need to update the reference"
refFile="neighbours_map_ecalB_thetamodulemerged_ecalE_turbine_hcalB_hcalEndcap_phitheta.root"
if [ ! -f $refFile ]; then
    wget https://fccsw.web.cern.ch/fccsw/filesForSimDigiReco/ALLEGRO/ALLEGRO_o1_v03/$refFile
fi
python utils/compareMaps.py neighbours neighbours_map_ecalB_ecalE_hcalB_hcalE.root $refFile  --debugevts 5

