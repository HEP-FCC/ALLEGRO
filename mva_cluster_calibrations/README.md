# Recipes to re-train the mva based calibration

The mva based calibration has been developed for the ECAL barrel but should be easily extendable to the ECAL Endcap and to ECAL+HCAL clusters.
It uses as input the total cluster energy and the fraction of energy in each longitudinal layer.
Additional input features can be used such as cluster theta (parameter useExtraFeaures)
The output target is the ratio between particle energy and cluster energy.

The code in this folder can be used to train and test the calibration.
It uses LightGBM to optimise the BDT weights.
The output model is saved to ONNX.
The input features and output target can be read from ROOT files produced with the ALLEGRO simulation, or from CSV or pickle files produced by a previous run of the script.

Both kinds of models can be used in the reconstruction by `CalibrateCaloClusters` (k4RecCalorimeter), which takes the
layout of the inputs from the input width of the ONNX model:
- models trained with `useExtraFeatures=False` have `m_numLayersTotal + 1` inputs: the energy fractions in the layers
  and the cluster energy;
- models trained with `useExtraFeatures=True` have `m_numLayersTotal + 4` inputs: the energy fractions in the layers,
  the cluster theta, theta modulo the theta size of the cells (`deltaThetaCalo`), phi modulo the phi size of the cells
  (`deltaPhiCalo`), and the cluster energy. They need k4RecCalorimeter v0.1.0pre19 or later (#261): older versions
  reject them in the check of the input width.

The position of the cluster (hence the extra features) and, for the sliding-window clusters, the energy depend on
the options of the clustering (e.g. `recentrePositionWindow`): the model should be trained on samples reconstructed
with the same options as the samples to calibrate.

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

The settings at the top of both scripts can also be overridden with command line options (`--help`), e.g. to train
on given files and evaluate on samples with given energies:

```
python train_calibration.py --input <files> --clusters EMBCaloClusters --suffix _test --no-extra-features
python test_calibration.py --input-pattern 'clusters/reconstruction_energy_{energy}_theta_90_particle_gamma.root' \
    --energies 1000 10000 100000 --particle gamma --clusters EMBCaloClusters --suffix _test --no-extra-features
```
`--theta-range MIN MAX` evaluates the performance only for particles with min(theta, 180 - theta) in [MIN, MAX)
degrees, e.g. on samples with a flat theta distribution.

The plot comparing the energy resolution for LAr+Pb and LKr+W barrel ECAL for the ALLEGRO paper was done as follows:
* extract sampling fractions for the two scenarios, using options `doCalibrationFiles` and `doSamplingFractions` in file [run_all_chain.sh](https://github.com/HEP-FCC/ALLEGRO/blob/main/utils/run_all_chain.sh) as described [here](https://github.com/HEP-FCC/ALLEGRO/sampling_fractions/README.md)
* simulate and reconstruct events for training, using option `doClustersForMVATraining` in file [run_all_chain.sh](https://github.com/HEP-FCC/ALLEGRO/blob/main/utils/run_all_chain.sh)
* simulate and reconstruct events for testing, using option `doClustersForMVAEvaluation` in file [run_all_chain.sh](https://github.com/HEP-FCC/ALLEGRO/blob/main/utils/run_all_chain.sh)
* when running the simulation and reconstruction, make sure the proper detector models and sampling fractions are used in the official reconstruction ([run_digi_reco_common.py](https://github.com/HEP-FCC/FCC-config/blob/main/FCCee/FullSim/ALLEGRO/ALLEGRO_common/run_digi_reco_common.py) in FCC-config, which `runParallel.py` copies to the output folder of each production) and in the other scripts ([runParallel.py](https://github.com/HEP-FCC/ALLEGRO/blob/main/utils/runParallel.py), [run_all_chain.sh](https://github.com/HEP-FCC/ALLEGRO/blob/main/utils/run_all_chain.sh))
* train and test the calibration with the scripts described above, adjusting the names and directories of the input files in the python script, as well as the suffix of the produced output files (models, json with results, plots)
* overlay the two resolution curves in a single plot: `python plot_paper.py`

## Endcap

The same code used to train and test the MVA calibration for the barrel also works for the endcap. One has to edit the parameters in the training and testing scripts (`train_calibration.py` and `test_calibration.py`) and then execute

```
python train_calibration.py
python test_calibration.py
```
