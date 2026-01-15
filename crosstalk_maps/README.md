# Recipes to update the crosstalk maps

## Barrel
1. *Define types of cross-talk neighbours in*

  xtalk_neighbors_moduleThetaMergedSegmentation.cpp

  Currently, there are four types of cross-talk neighbours: direct radial, direct theta, diagonal and theta tower (between signal traces and cells).
  Any update of "xtalk_neighbors_moduleThetaMergedSegmentation.cpp" and "xtalk_neighbors_moduleThetaMergedSegmentation.h" should be pushed to the k4geo repository: https://github.com/key4hep/k4geo/tree/main/detectorCommon/include/detectorCommon and https://github.com/key4hep/k4geo/tree/main/detectorCommon/src
  

2. *Functions for the generation of the barrel region cross-talk map are included in*

   CreateFCCeeCaloXTalkNeighbours.cpp

   Any update of "CreateFCCeeCaloXTalkNeighbours.cpp" and "CreateFCCeeCaloXTalkNeighbours.h" should be pushed to the k4RecCalorimeter repository: https://github.com/HEP-FCC/k4RecCalorimeter/tree/main/RecFCCeeCalorimeter/src/components

3. **Generate the cross-talk map by running**

  `k4run runCaloXTalkNeighbours.py`

  The four cross-talk coefficients are configurable: xtalkCoefRadial, xtalkCoefTheta, xtalkCoefDiagonal and xtalkCoefTower.

  The default output file is "xtalk_neighbours_map_ecalB_thetamodulemerged.root", in which the cross-talk neighbours and coefficients are saved for each cell.

## Endcap

