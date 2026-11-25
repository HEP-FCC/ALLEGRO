import argparse
import torch
import os
import sys
import onnxruntime as ort
import numpy as np

this_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(this_dir, "..")
absolute = os.path.abspath(parent_dir)

final_path = os.path.join(absolute, "training")
sys.path.append(final_path)

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
parser = argparse.ArgumentParser()
parser.add_argument("-w","--weightfile",type=str, default="",help="path to weight file")
parser.add_argument("-o","--onnxfile",type=str, default="/",help="path to onnx file")   
args = parser.parse_args()

torch.manual_seed(42)
input = torch.randn((30, 7), dtype=torch.float32)
pos = torch.randn((200, 3),  dtype=torch.float32)
en = torch.randn((200, 1),  dtype=torch.float32)

# to run comparison with a parquet file
## import pandas as pd
## import numpy as np
## df = pd.read_parquet("reco_ALLEGRO_ZH240_gamma_10_004.parquet")
## print(df.shape)
## x = df["X_hit"].iloc[0] # X_hit
## first_event = np.stack(x[2]).astype(np.float32) #particular event
## 
## print("shape:", first_event.shape)
## 
## 
## pos = first_event[:, 1:4]
## pos = torch.tensor(pos, dtype=torch.float32, device='cpu')
## en = first_event[:, :1]
## en = torch.tensor(en, dtype=torch.float32, device='cpu')

# ONNX-friendly version of the model
from src.models.GATr.Gatr_onnx import ExampleWrapper as GravnetModel
model_weights = args.weightfile
torch._dynamo.config.verbose = True
model = GravnetModel.load_from_checkpoint(
    model_weights,
    args=args,
    dev='cpu',
    map_location=torch.device("cpu"))
model.eval()
with torch.no_grad():
    output_torch = model(pos, en)

# output from ONNX model
sess_options = ort.SessionOptions()
sess_options.log_severity_level = 3
onnx_model_path = args.onnxfile
session = ort.InferenceSession(onnx_model_path, sess_options)
position_name = session.get_inputs()[0].name
energy_name = session.get_inputs()[1].name

for inp in session.get_inputs():
    print(inp.name)
    print(" shape:", inp.shape)
    print(" type :", inp.type)

print("pos:", pos.shape)
print("en :", en.shape)

outputs = session.run(None, 
                      {position_name: pos.numpy().astype(np.float32), 
                       energy_name: en.numpy().astype(np.float32)},
                    )
output_onnx = outputs[0]
output_onnx = torch.tensor(output_onnx, dtype=torch.float32, device='cpu')

print("Output Pure Python (PyTorch):", output_torch)
print("Output ONNX Runtime:", output_onnx)

print("\nPyTorch Output:")
print("  Type:", type(output_torch))
print("  Dtype:", output_torch.dtype)
print("  Shape:", output_torch.shape)

print("\nONNX Output:")
print("  Type:", type(output_onnx))
print("  Dtype:", output_onnx.dtype)
print("  Shape:", output_onnx.shape)

if torch.is_tensor(output_onnx):
    output_onnx_np = output_onnx.detach().cpu().numpy()
else:
    output_onnx_np = np.array(output_onnx)

output_torch_np = output_torch.detach().cpu().numpy()

max_diff = np.max(np.abs(output_torch_np - output_onnx_np))
are_close = np.allclose(output_torch_np, output_onnx_np, rtol=1e-5, atol=1e-6)

print("\nComparison:")
print("  Max Absolute Difference:", max_diff)
print("  Are outputs close?:", are_close)

