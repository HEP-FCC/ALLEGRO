#!/bin/bash

# set-up the Key4hep environment if not already set
if [[ -z "${KEY4HEP_STACK}" ]]; then
  source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh
else
  echo "The Key4hep stack was already loaded in this environment."
fi

#test creating root file with capacitance values
python noise_maps/create_capacitance_file_ecalendcap_simple.py FCCee/ALLEGRO/compact/ALLEGRO_o1_v03/ALLEGRO_o1_v03.xml

#test creating root file with noise values
python noise_maps/create_noise_file_ecalendcap.py FCCee/ALLEGRO/compact/ALLEGRO_o1_v03/ALLEGRO_o1_v03.xml




