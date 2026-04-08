# Recipes to update the noise maps

## Barrel

## Endcap

The first step is the prepare a root file with the capacitances of all the cells.  The current implementation only considers the transmission line capacitances (the absorber/pad capacitance in small by comparison).  
Currently this is done by

create_capacitance_file_ecalendcap_simple.py

note that any updates to the geometry (particularly the blade angle and the number of cells) need to be implemented in that python script (they are not automatically read in from an html file).  The "simple" in the filename reflects the fact that this is a very basic implementation and considers only the distance from the readout cell to the back of the detector, not the distance of the actual path a trace might take when the readout board lyout is done.

Output of the above is endcap_capacitances.root.

Next, convert the capacitances to noise. 

python create_noise_file_ecalendcap.py 

This first converts from capacitance to noise in electrons, assuming cold
electronics, with that conversion taken from Omega lab measurements as reported in https://indico.cern.ch/event/1545838/contributions/6831866/attachments/3194785/5686292/capa_and_noise_juska_v0.pdf.

Next the number of electrons is converted to MeV, based on the assumption that 23.6 eV is needed on average to create an electron/ion pair, and tha the recombination rate is 4%.  The output is noise_capa_ecalendcap/elecNoise_ecalendcap.root, which can be used as input to the routine in k4RecCalorimeter that creates the noise map.
