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


# create the neighbour map
echo
echo "##############################"
echo "# Creating the neighbour map #"
echo "##############################"
echo
outFile=neighbours_map_ecalB_ecalE_hcalB_hcalE.root
rm -f $outFile
k4run $ALLEGRO/neighbor_maps/neighbours.py --ecalb --ecalec --hcalb --hcalec --link-calos --link-ecal --link-hcal || exit 1

if [ ! -f $outFile ]; then
    echo "Output file missing"
    exit -1
fi

# for debug: compare to the one in the repository
echo
echo "#############################"
echo "Comparing new map to reference one. If the test fails, you might need to update the reference"
echo "#############################"
echo
refFile="neighbours_map_ecalB_thetamodulemerged_ecalE_turbine_hcalB_hcalEndcap_phitheta.root"
if [ ! -f $refFile ]; then
    wget https://fccsw.web.cern.ch/fccsw/filesForSimDigiReco/ALLEGRO/ALLEGRO_o1_v03/$refFile
fi
if [ ! -f $refFile ]; then
    echo "Failed to download reference file"
    exit -1
fi
python $ALLEGRO/utils/compareMaps.py neighbours neighbours_map_ecalB_ecalE_hcalB_hcalE.root $refFile  --debugevts 5 || exit 1
rm $refFile

echo
echo "#############################"
echo "# SUCCESS                   #"
echo "#############################"
echo
