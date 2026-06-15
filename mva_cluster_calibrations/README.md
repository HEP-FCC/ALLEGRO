# Recipes to re-train the mva based calibration

The mva based calibration has been developed for the ECAL barrel but should be easily extendable to the ECAL Endcap and to ECAL+HCAL clusters.
It uses as input the total cluster energy and the fraction of energy in each longitudinal layer.
Additional input features can be used such as cluster theta (parameter useExtraFeaures)
The output target is the ratio between particle energy and cluster energy.

The code in this folder can be used to train and test the calibration.
It uses LightGBM to optimise the BDT weights.
The output model is saved to ONNX.
The input features and output target can be read from ROOT files produced with the ALLEGRO simulation, or from CSV or pickle files produced by a previous run of the script.

[!WARNING]
Models trained with `useExtraFeatures=True` use additional input variables compared to the default configuration. At the moment, these models may not be directly usable in reconstruction with the default `CalibrateCaloClusters` input-shape check, which expects `m_numLayersTotal + 1` input features.
If the exported ONNX model contains the extra features, the corresponding reconstruction-side configuration/code must be updated consistently, e.g. to expect `m_numLayersTotal + 3` input features for this case.

Models trained with `useExtraFeatures=False` keep the default input layout and can be used with the current default reconstruction setup.

## Installation

## Barrel

### Preliminary requirements
To perform the calibration, particle gun samples of photons of various energies containing the clusters to be calibrated have to be produced.

### Installation of the code
Make a local clone of the gitlab repository and setup virtual environment the first time with
```
source setup.sh
```

After that, every time you start from a fresh shell, do
```
source env.sh
```
### Execution
To train the BDT: adjust properly the parameters in the script `train_calibration.py`, including the name of the cluster collections to calibrate and the list of input files, and execute it with:

```
python train_calibration.py

```

To evaluate the BDT performance: adjust properly the parameters in the script `test_calibration.py`, and execute it with:

```
python test_calibration.py
```

The plot comparing the energy resolution for LAr+Pb and LKr+W barrel ECAL for the ALLEGRO paper was done as follows:
* extract sampling fractions for the two scenarios, using options `doCalibrationFiles` and `doSamplingFractions` in file [run_all_chain.sh](https://gitlab.cern.ch:8443/gmarchio/FCC-scripts/-/blob/main/run_all_chain.sh)
* simulate and reconstruct events for training, using option `doClustersForMVATraining` in file [run_all_chain.sh](https://gitlab.cern.ch:8443/gmarchio/FCC-scripts/-/blob/main/run_all_chain.sh)
* simulate and reconstruct events for testing, using option `doClustersForMVAEvaluation` in file [run_all_chain.sh](https://gitlab.cern.ch:8443/gmarchio/FCC-scripts/-/blob/main/run_all_chain.sh)
* when running the simulation and reconstruction, make sure the proper detector models and sampling fractions are used in the reconstruction steering script [run_ALLEGRO_reco.py](https://gitlab.cern.ch:8443/gmarchio/FCC-scripts/-/blob/main/run_ALLEGRO_reco.py) and in the other scripts ([runParallel.py](https://gitlab.cern.ch:8443/gmarchio/FCC-scripts/-/blob/main/runParallel.py), [run_all_chain.sh](https://gitlab.cern.ch:8443/gmarchio/FCC-scripts/-/blob/main/run_all_chain.sh))
* train and test the calibration with the scripts described above, adjusting the names and directories of the input files in the python script, as well as the suffix of the produced output files (models, json with results, plots)
* overlay the two resolution curves in a single plot: `python plot_paper.py`

## Endcap

The same code used to train and test the MVA calibration for the barrel also works for the endcap. One has to edit the parameters in the training and testing scripts (`train_calibration.py` and `test_calibration.py`) and then execute

```
python train_calibration.py
python test_calibration.py
```
