# TRAnsformer-based Photon-neutral PIon SeparaTor

This folder documents TRAPPIST, an ML tool to discriminate between single photon showers and showers produced by two collimated photons from a neutral pion in the ECAL. 
- `dataset-prod`: scripts for dataset generation, including automatic condor job submission
- `training`: lightining implementation of the model
- `onnx-conv`: scripts to convert a trained model in ONNX to run inference via Gaudi functionals

Each subfolder has its own README. 

TRAPPIST is currently part of the [ALLEGRO reconstruction chain](https://github.com/HEP-FCC/FCC-config/blob/627161c53a744da479850d512cb3dfbefd7e1c9c/FCCee/FullSim/ALLEGRO/ALLEGRO_common/run_digi_reco_common.py#L52):
- [Inference on clusters](https://github.com/HEP-FCC/k4RecCalorimeter/blob/main/RecFCCeeCalorimeter/src/components/TRAPPISTPi0PhotonInference.cpp)
- [`RecoParticle` production](https://github.com/HEP-FCC/k4RecCalorimeter/blob/main/RecFCCeeCalorimeter/src/components/ClusterPi0PhotonID.cpp)
