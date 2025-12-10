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

