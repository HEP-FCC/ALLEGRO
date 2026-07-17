import onnx
import lightgbm as lgb
from onnxconverter_common.data_types import FloatTensorType
from onnxmltools.convert import convert_lightgbm

infile = 'inclusive_1Mevents_allvars/models/bdt-photonid.txt'
outfile = 'bdt-photonid.onnx'

model = lgb.Booster(model_file=infile)
# https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.Booster.html
initial_types=[('X', FloatTensorType([None, model.num_feature()]))]
model_onnx = convert_lightgbm(model, initial_types=initial_types, zipmap=False,
                              split=100)
onnx.save(model_onnx, outfile)
