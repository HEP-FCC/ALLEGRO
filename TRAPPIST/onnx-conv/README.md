## ONNX conversion 

To use the trained model in a Gaudi algorithm it must be converted in ONNX. A specific environment has to be used
```
docker pull justdrew/onnxconversion
```
To convert a checkpoint, first make sure that the GATr implementation at `training/src/models/GATr/Gatr_onnx.py` is equivalent to the model that has been used for training (e.g. number of blocks, input/output dimensions, ...), then
```
 python checkpointToONNX.py c your.ckpt --o output/path -n outputName.onnx
```
Then, check that the difference between results obtained with the original file and the converted one is small
```
python comparePythonOnnx.py -w your.ckpt -o your.onnx
```