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
echo "Note (27/5/2026): due to rounding some (4) edge cells in the ECAL barrel may have slightly different"
echo "neighbours in the ECAL endcap when running the test locally rather than in github CI"
echo "#############################"
echo
refFile="neighbours_map_ecalB_thetamodulemerged_ecalE_turbine_hcalB_hcalEndcap_phitheta.root"
if [ ! -f $refFile ]; then
    wget -nv https://fccsw.web.cern.ch/fccsw/filesForSimDigiReco/ALLEGRO/ALLEGRO_o1_v03/$refFile
fi
if [ ! -f $refFile ]; then
    echo "Failed to download reference file"
    exit -1
fi
$ALLEGRO/install/bin/compareMaps neighbours $refFile neighbours_map_ecalB_ecalE_hcalB_hcalE.root --debug-events 5 -m 1 || exit 1
rm $refFile

echo
echo "#############################"
echo "# SUCCESS                   #"
echo "#############################"
echo
