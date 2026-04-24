# Recipes to update the noise maps

## Barrel

### Step 1

initialize FCCSW environment

```source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh```

### Step 2

run capacitance script for calculating capacitance maps

```python create_capacitance_file_theta_update2025.py```

# Step 3

run capacitance-to-noise conversion script for getting noise maps

```python create_noise_file_chargePreAmp_theta_update2025.py```

Now you should have the new noise maps and supporting plots in a newly created directory

# Step 4

create the noise map

```k4run noise_map.py --detector FCCee/ALLEGRO/compact/ALLEGRO_o1_v03/ALLEGRO_o1_v03.xml --subdetectors ecale```

The output is `cellNoise_map_electronicsNoiseLevel_ecalB_ECalBarrelModuleThetaMerged.root`.


## Endcap

### Step 1

initialize FCCSW environment

```source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh```

### Step 2

Prepare a root file with the capacitances of all the cells.  The current implementation only considers the transmission line capacitances (the absorber/pad capacitance in small by comparison).
Currently this is done by executing

```python create_capacitance_file_ecalendcap_simple.py FCCee/ALLEGRO/compact/ALLEGRO_o1_v03/ALLEGRO_o1_v03.xml```

The "simple" in the filename reflects the fact that this is a very basic implementation and considers only the distance from the readout cell to the back of the detector, not the distance of the actual path a trace might take when the readout board lyout is done.

The output is `endcap_capacitances.root`.

### Step 3

Convert the capacitances to noise histograms.

```python create_noise_file_ecalendcap.py FCCee/ALLEGRO/compact/ALLEGRO_o1_v03/ALLEGRO_o1_v03.xml```

This first converts from capacitance to noise in electrons, assuming cold
electronics, with that conversion taken from Omega lab measurements as reported in https://indico.cern.ch/event/1545838/contributions/6831866/attachments/3194785/5686292/capa_and_noise_juska_v0.pdf.

Next the number of electrons is converted to MeV, based on the assumption that 23.6 eV is needed on average to create an electron/ion pair, and tha the recombination rate is 4%.

The output is `noise_capa_ecalendcap/elecNoise_ecalendcap.root`.


### Step 4

Create the noise map file that can be used in reconstruction.

```k4run noise_map.py --detector FCCee/ALLEGRO/compact/ALLEGRO_o1_v03/ALLEGRO_o1_v03.xml --subdetectors ecale```

The output is `cellNoise_map_electronicsNoiseLevel_ecalE_ECalEndcapTurbine.root`.
