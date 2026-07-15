# Recipes to update the photon/pi0 identification algorithms

## Barrel

### BDT-based

A simple BDT-based photon/pi0 discrimination algorithm was developed in 2024 to perform photon/pi0 discrimination based on shower shape variables (inspired from ATLAS) calculated using the longitudinal and lateral (phi/theta) energy profile of the cluster (see initial presentation [here](https://indico.cern.ch/event/1430231/contributions/6022836/attachments/2884833/5055904/photon_ID_ALLEGRO_TongLi_240626.pdf]

The BDT is trained with LightGBM, and the model is converted to onnx. The shower shapes are calculated by the AugmentCaloCluster algorithm and the inference is run by the PhotonIDTool algorithm, both written at the same time of the development of the BDT code.

While the BDT is likely being phased out in favour of more advanced taggers, it is kept here since it provides a useful baseline and can be trained very quickly, so that it can still be used to e.g. assess performance differences across detector variations.

The code lives in the [bdt](https://github.com/HEP-FCC/ALLEGRO/tree/main/pi0_photon_identification/bdt) subfolder and the instructions are in the [README.md](https://github.com/HEP-FCC/ALLEGRO/tree/main/pi0_photon_identification/bdt/README.md) file there.

### Transformed-based (TRAPPIST)

A more avdanced algorithm, based on the transformed architecture, has been developed in 2026.

## Endcap
No dedicated photon identification algorithm develoep/trained yet (but should be fairly easy to adapt cell-based photon-ID algorithm to the endcap)
