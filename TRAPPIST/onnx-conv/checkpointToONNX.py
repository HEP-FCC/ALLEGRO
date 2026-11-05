import argparse
import torch
import os
import sys
from pathlib import Path

script_dir = os.path.dirname(os.path.abspath(__file__))
working_dir = os.path.join(script_dir, "..")
working_dir = os.path.abspath(working_dir)
sys.path.append(os.path.join(working_dir, "training"))
sys.path.append(working_dir)

print(os.path.join(working_dir, "training"))

from onnxscript.function_libs.torch_lib.ops import nn as _nn
from onnxscript import opset18 as op

def _gelu_tanh_fix(x):
    # float constant as scalar with value_float
    three = op.Constant(value_float=3.0)
    coeff = op.Constant(value_float=0.044715)
    return op.Mul(op.Pow(x, three), coeff) + x

def _gelu_none_fix(x):
    sqrt2 = op.Constant(value_float=1.4142135)
    return op.Mul(op.Erf(op.Div(x, sqrt2)), x)

_nn._aten_gelu_approximate_tanh = _gelu_tanh_fix
_nn._aten_gelu_approximate_none = _gelu_none_fix


##################################################################################

from src.models.GATr.Gatr_onnx import ExampleWrapper as GravnetModel

parser = argparse.ArgumentParser()
#parser arguments
parser.add_argument("-c", "--checkpoint", type=str, default="", help="path to weight file")
parser.add_argument("-o", "--outputpath", type=str, default="", help="path to output directory")
parser.add_argument("-n", "--modelname", type=str, default="testconv.onnx", help="output model filename")

args = parser.parse_args()
filepath = args.outputpath + "/" + args.modelname
model_weights = args.checkpoint
torch._dynamo.config.verbose = True
model = GravnetModel.load_from_checkpoint(
                    model_weights,
                    args=args,
                    dev='cpu',  
                    map_location=torch.device("cpu")  
                )
model.eval()

pos_dummy = torch.randn((10, 3))
en_dummy = torch.randn((10, 1))
export_options = torch.onnx.ExportOptions(dynamic_shapes=True)


# Export
torch.onnx.export(
    model,
    (pos_dummy, en_dummy),
    filepath,
    input_names=["position", "energy"],
    output_names=["output"],
    opset_version=17,  
    do_constant_folding=True,
    dynamic_axes = {
        "position": {0: "n_hits"},
        "energy":   {0: "n_hits"},
    },
    verbose=True
)