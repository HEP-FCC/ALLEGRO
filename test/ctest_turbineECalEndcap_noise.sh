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


# test creating root file with capacitance values
echo
echo "#############################"
echo "# Creating capacitance file #"
echo "#############################"
echo
python $ALLEGRO/noise_maps/create_capacitance_file_ecalendcap_simple.py FCCee/ALLEGRO/compact/ALLEGRO_o1_v03/ALLEGRO_o1_v03.xml || exit 1
outFile=endcap_capacitances.root
if [ ! -f $outFile ]; then
    echo "Output file missing"
    exit -1
fi

# test creating root file with noise values
echo
echo "#############################"
echo "# Creating noise histograms #"
echo "#############################"
echo
python $ALLEGRO/noise_maps/create_noise_file_ecalendcap.py FCCee/ALLEGRO/compact/ALLEGRO_o1_v03/ALLEGRO_o1_v03.xml || exit 1
outFile=noise_capa_ecalendcap/elecNoise_ecalendcap.root
if [ ! -f $outFile ]; then
    echo "Output file missing"
    exit -1
fi

# compare noise histograms to reference file
echo
echo "#############################"
echo "# Comparing new noise histograms to reference. If the test fails, you might need to update the reference"
echo "#############################"
echo
refFile=elecNoise_ecalendcap.root
if [ ! -f $refFile ]; then
    wget https://fccsw.web.cern.ch/fccsw/filesForSimDigiReco/ALLEGRO/ALLEGRO_o1_v03/$refFile
fi
if [ ! -f $refFile ]; then
    echo "Failed to download reference file"
    exit -1
fi
python $ALLEGRO/utils/compare_ecalendcap_noisehists.py $refFile $outFile || exit 1
rm $refFile

# test creating final noise map
echo
echo "#############################"
echo "# Creating final noise map  #"
echo "#############################"
echo
k4run $ALLEGRO/noise_maps/noise_map_endcapturbine.py --compactFile FCCee/ALLEGRO/compact/ALLEGRO_o1_v03/ALLEGRO_o1_v03.xml || exit 1
outFile=cellNoise_map_endcapTurbine_electronicsNoiseLevel.root
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
outFile=tmp/$refFile
if [ ! -f $refFile ]; then
    wget https://fccsw.web.cern.ch/fccsw/filesForSimDigiReco/ALLEGRO/ALLEGRO_o1_v03/$refFile
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
