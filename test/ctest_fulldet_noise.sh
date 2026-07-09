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

# create noise map
echo
echo "#############################"
echo "# Creating noise map  #"
echo "#############################"
echo
k4run $ALLEGRO/noise_maps/noise_map.py --subdetectors ecalb ecale hcalb hcale  --detector FCCee/ALLEGRO/compact/ALLEGRO_o1_v03/ALLEGRO_o1_v03.xml || exit 1
outFile=cellNoise_map_electronicsNoiseLevel_ecalB_ECalBarrelModuleThetaMerged_ecalE_ECalEndcapTurbine_hcalB_HCalBarrelReadout_hcalE_HCalEndcapReadout.root
if [ ! -f $outFile ]; then
    echo "Output file missing"
    exit -1
fi

# compare created noise map to default one used in reconstruction
echo
echo "#############################"
echo "Comparing new map to reference one. If the test fails, you might need to update the reference"
echo "#############################"
echo
mkdir -p tmp
mv $outFile tmp
refFile=$outFile
outFile=tmp/$outFile
if [ ! -f $refFile ]; then
    wget -nv https://fccsw.web.cern.ch/fccsw/filesForSimDigiReco/ALLEGRO/ALLEGRO_o1_v03/$refFile
fi
if [ ! -f $refFile ]; then
    echo "Failed to download reference file"
    exit -1
fi
python $ALLEGRO/utils/compareMaps.py noise $outFile $refFile  --debugevts 5 || exit 1
rm $refFile
mv $outFile .

# remove tmp if empty
if [ -z "$(find tmp -mindepth 1 -print -quit)" ]; then
    rmdir tmp
fi

echo
echo "#############################"
echo "# SUCCESS                   #"
echo "#############################"
echo
