#!/bin/bash

# set-up the Key4hep environment if not already set
if [[ -z "${KEY4HEP_STACK}" ]]; then
  source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh
else
  echo "The Key4hep stack was already loaded in this environment."
fi
# for debug
# printenv

# create the neighbour map
outFile=neighbours_map_ecalB_ecalE_hcalB_hcalE.root
rm -f $outFile
k4run neighbor_maps/neighbours.py --ecalb --ecalec --hcalb --hcalec --link-calos --link-ecal --link-hcal

if [ ! -f $outFile ]; then
    echo "Output file missing"
    exit -1
fi

# for debug: compare to the one in the repository
echo "Comparing new map to reference one. If the test fails, you might need to update the reference"
refFile="neighbours_map_ecalB_thetamodulemerged_ecalE_turbine_hcalB_hcalEndcap_phitheta.root"
if [ ! -f $refFile ]; then
    wget https://fccsw.web.cern.ch/fccsw/filesForSimDigiReco/ALLEGRO/ALLEGRO_o1_v03/$refFile
fi
if [ ! -f $refFile ]; then
    echo "Failed to download reference file"
    exit -1
fi
python utils/compareMaps.py neighbours neighbours_map_ecalB_ecalE_hcalB_hcalE.root $refFile  --debugevts 5

