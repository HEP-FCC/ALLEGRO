# Recipes to update sampling fractions

## Barrel

## Endcap

There are two different approaches avaialble for the endcap, one based on a direct assessment of the sampling fraction of each cell, and one based on an empirical determination of the calibration constant for each cell.

### Direct assessment of sampling fractions
For the direct assessment, one follows a procedure similar to that used for the barrel.  First, run a set of single particles using the *calib* version of the detector xml file, where the absorbers and readout PCBs are set to sensitive.   You only need to run the simulation step, not reconstruction.  Then run:

k4run fcc_ee_samplingFraction_turbineECalEndcap_siminput.py

with the following lines edited to match your configuration:

\# Electron momentum in GeV  
momentum = 40

evtsvc.input = "root/allegro_v03_evts_10000_*sim.root"

This will produce a file histSF_fccee_turbine.root.

The last step is to fit the histograms from that root file to extract the sampling fractions:

python plot_samplingFraction.py histSF_fccee_turbine.root 10 --axisMin=-0.01 --axisMax=0.5

**Notes/caveats**:  The extraction of the sampling fractions tends to not be as reliable for the endcap as the barrel.  This is because each cluster essentually deposits some energy in all of the calibration layers of the barrel (since the layers are arranged in depth), but that is not the case for the endcap, where layers are arranged in both depth and in the rho coordinate.  This means that most layers are empty for any one given cluster, leading to a large low-side tail in the sampling fraction plots that can make the fit unreliable.  It is generally the case that using the sampling fractions as calibration constants allows one to reconstruct the correct cluster energy to within 1% for the barrel; for the endcap the variation is closer to 10%.  So for precise studies of the endcap performance, the empirical determination of calibration constants described below is preferable. 

### Empricial determination of calibration constants

For this procedure, one generates a large set of single-particle events (single photons may be best, as they are not affected by the B field).  Typically I have used 100k particles at 40 GeV, uniformly distributed in theta and phi across the theta values relevant to the endcap (5 to 40 degrees).  I generally run with the upstream detectors removed so that the performance of the endcap calorimeter can be isolated.  Events are reconstructed using sampling fractions of 1 for each cell.

The resulting reconstructed events are then processed by the ROOT macro autoCalib_analytic_topo.C, as follows:

TChain *t =  new TChain("events")  
t->Add(<set of reconstructed ntuples>)  
.L autoCalib_analytic_topo.C+  
autoCalib_analytic_topo a(t)  
a.Calibrate()  

The result will be a file samplingFractionsAnalytic.txt that contains the calibration constants is a format that can be directly inserted into the reconstruction steering file.

**Notes/caveats**: this process is based on the assumption that the true energy of each cluster is 40 GeV.  If you generated the particles at the different energy, you will need to edit the following line in autoCalib_analytic_topo.C:

\#define Etrue 40.0

Also, since this procedure operates under the assumption that all of the energy was deposited in the endcap, it does not give appropriate calibrations for the cells at the outer edge of the endcap that are "shadowed" by the barrel.  It is much better to use the direct sampling fraction approach for multi-subsystem studies that include that region.



