# Recipes to update the crosstalk maps

## Barrel
Crosstalk maps are created with the `runCaloXTalkNeighbours.py` script by executing

`k4run runCaloXTalkNeighbours.py`

after having setup the key4hep stack with e.g.

`source /cvmfs/sw-nightlies.hsf.org/k4hep/setup.sh`

The default output file is "xtalk_neighbours_map_ecalB_thetamodulemerged.root", in which the cross-talk neighbours and coefficients are saved for each cell.

The code relies on
* types of cross-talk neighbours, defined in the files "xtalk_neighbors_moduleThetaMergedSegmentation.h" and "xtalk_neighbors_moduleThetaMergedSegmentation.cpp" in the k4geo package, in the directories https://github.com/key4hep/k4geo/tree/main/detectorCommon/include/detectorCommon and https://github.com/key4hep/k4geo/tree/main/detectorCommon/src
* functions for the generation of the barrel region cross-talk map, implemented in "CreateFCCeeCaloXTalkNeighbours.h" and "CreateFCCeeCaloXTalkNeighbours.cpp" in the k4RecCalorimeter package, in the directory https://github.com/HEP-FCC/k4RecCalorimeter/tree/main/RecFCCeeCalorimeter/src/components

Currently, there are four types of cross-talk neighbours: direct radial, direct theta, diagonal and theta tower (between signal traces and cells).
The four cross-talk coefficients are configurable properties that can be modified in `runCaloXTalkNeighbours.py`: xtalkCoefRadial, xtalkCoefTheta, xtalkCoefDiagonal and xtalkCoefTower.

To define additional type of crosstalk neighbours the user should
* clone locally the k4geo and k4RecCalorimeter packages
* modify the files
```
xtalk_neighbors_moduleThetaMergedSegmentation.h
xtalk_neighbors_moduleThetaMergedSegmentation.cpp
CreateFCCeeCaloXTalkNeighbours.h
CreateFCCeeCaloXTalkNeighbours.cpp
runCaloXTalkNeighbours.py
```
* setup the environment and compile
* execute the `runCaloXTalkNeighbours.py` script


## Endcap

