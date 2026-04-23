# Recipes to update the noise maps

## Barrel
# Step 0: log in to lxplus, clone or copy scripts to the working directory
# Step 1: initialize FCCSW environment:
$ source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh

# Step 2: run capacitance script for calculating capacitance maps:
python create_capacitance_file_theta_update2025.py

# Step 3: run capacitance-to-noise conversion script for getting noise maps:
python create_noise_file_chargePreAmp_theta_update2025.py

# Now you should have the new noise maps and supporting plots in a newly created directory


## Endcap

The first step is the prepare a root file with the capacitances of all the cells.  The current implementation only considers the transmission line capacitances (the absorber/pad capacitance in small by comparison).  
Currently this is done by

create_capacitance_file_ecalendcap_simple.py <compact file name (XML)>

The "simple" in the filename reflects the fact that this is a very basic implementation and considers only the distance from the readout cell to the back of the detector, not the distance of the actual path a trace might take when the readout board lyout is done.

Output of the above is endcap_capacitances.root.

Next, convert the capacitances to noise. 

python create_noise_file_ecalendcap.py <compact file name (XML)>

This first converts from capacitance to noise in electrons, assuming cold
electronics, with that conversion taken from Omega lab measurements as reported in https://indico.cern.ch/event/1545838/contributions/6831866/attachments/3194785/5686292/capa_and_noise_juska_v0.pdf.

Next the number of electrons is converted to MeV, based on the assumption that 23.6 eV is needed on average to create an electron/ion pair, and tha the recombination rate is 4%.  The output is noise_capa_ecalendcap/elecNoise_ecalendcap.root.

Now cd into your k4geo working area and do the following to create noise map file that can be used in reconstruction. Note that you may need to unset the FCCDETECTORS variable if you are using a local version of the detector XML:

k4run ../ALLEGRO/noise_maps/noise_map_endcapturbine.py
 

