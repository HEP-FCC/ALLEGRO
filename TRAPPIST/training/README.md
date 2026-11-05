# Model training and evaluation

The docker container at `docker://dologarcia/gatr:v9` has the dependencies required to train and test the model. The script `slurm-job.sh` shows how to get the environment and launch a training on BSC cluster and can be easily adapted to other machines. 

To run fast detector optimization campaigns, you may want to simplify the model: this can be done for instance by lowering the number of transformer blocks in `Gatr_pf_e_binary.py`. 

For evaluation, the same environment can be used. However, TRAPPIST is part of key4hep, so it is recommended to convert the checkpoint to onnx with the script provided in `onnx-conv/` and use the relevant Gaudi algorithms to run evaluation directly with EDM4hep inputs. 
If needed, the script `slurm-job-eval.sh` shows how to produce the `.pt` file with the scores.
