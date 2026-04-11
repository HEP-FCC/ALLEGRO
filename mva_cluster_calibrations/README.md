# Recipes to re-train the mva based calibration

The mva based calibration has been developed for the ECAL barrel but should be easily extendable to the ECAL Endcap and to ECAL+HCAL clusters.
It uses as input the total cluster energy and the fraction of energy in each longitudinal layer.
Additional input features can be used such as cluster theta (parameter useExtraFeaures)
The output target is the ratio between particle energy and cluster energy.

The code in this folder can be used to train and test the calibration.
It uses LightGBM to optimise the BDT weights.
The output model is saved to ONNX.
The input features and output target can be read from ROOT files produced with the ALLEGRO simulation, or from CSV or pickle files produced by a previous run of the script.


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
To train the BDT: adjust properly the parameters in the script train_calibration.py, including the name of the cluster collections to calibrate and the list of input files, and execute it with:

```
python train_calibration.py

```

To evaluate the BDT performance: adjust properly the parameters in the script test_calibration.py, and execute it with:

```
python test_calibration.py
```

## Endcap
