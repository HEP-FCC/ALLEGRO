# BDT-based photonID for ALLEGRO ECAL barrel

## Description
Train and test a boosted decision tree for photon/pi0 separation with the ALLEGRO electromagnetic calorimeter.
The input features are read from the cluster shapeParameters array.
Code moved from https://gitlab.cern.ch:8443/gmarchio/fcc-lar-photonid

## Installation
```
source setup.sh
```

Note: the code reads the name of the variables in the shapeParameters array from the file metadata using the FCCSW software.
It assumes that it has been setup in the directory ~/work/fcc/allegro/fullsim following the instructions in https://gitlab.cern.ch/gmarchio/FCC-scripts/-/blob/main/README.md
If the directory path is different, it should be modified in the script getMetaData.sh.

## Usage
In a fresh shell, setup the environment:
```
source env.sh
```

To train and test the BDT in a given energy range:
```
python train_BDT.py --emin 0 --emax 1000 --outdir inclusive
```

To compare the inclusive BDT and the 5 BDTs in 0-20, 20-40, 40-60, 60-80, 80-100 GeV ranges:
```
python compare.py
```

## Roadmap
1. Currently all input features are fed to the BDT. It would be nice to reduce the list keeping the same BDT performance
2. No optimisation has been performed of the training settings. This should be revisited
3. The performance have been studied on relatively small samples of photons and pi0s (100k each, 70% used for training, 30% for testing). They should be evaluated on more events
4. Some overtraining was seen in the studies on 100k photons and pi0 when splitting the samples in 5 cluster energy intervals (0-20, 20-40, 40-60, 60-80 and 80-100 GeV). The studies in the previous points should also try to address this issue.
5. Input settings (for BDT training, including also directory where input files are located) could be passed via command line through json file to be copied also in output directory in order to persist this important information

## WIP
Other files in the directory:
- optimise.py: attempt to choose by brute-force the variables to be included, by adding variables one-by-one and check improvement in AUC. Seems to give results similar to just picking the top-ranked variables from the training with all variables
- train2.py: copied from train_BDT.py with an added function to do hyperparameter optimisation. Unfortunately on XGBoost at some point the optimisation fails because the GPU memory is saturated. Not yet fixed, and latest developments in train_BDT.py not ported to train2.py.

## Author
Giovanni Marchiori (giovanni.marchiori@cern.ch)
