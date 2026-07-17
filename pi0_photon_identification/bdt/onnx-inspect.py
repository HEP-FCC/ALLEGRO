import sys
import traceback

import onnx

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} /path/to/model.onnx")
    sys.exit(1)

model_path = sys.argv[1]
model = onnx.load(model_path)

input_all = {inp.name: inp for inp in model.graph.input}
print('all inputs count:', len(input_all))

initializer_names = set(init.name for init in model.graph.initializer)
feed_input_names = set(input_all) - initializer_names
print('required inputs:')
for inp_name in feed_input_names:
    print(input_all[inp_name])
print()

output = model.graph.output
print('outputs:', output, end="\n\n")

onnx.checker.check_model(model, full_check=True)
print("Model check successful!\n")

try:
    onnx.checker.check_graph(model.graph)
    print("Model graph check successful!\n")
except Exception:
    print("Model graph check failed with error:")
    print(traceback.format_exc())

try:
    onnx.utils.extract_model(
        model_path,
        "/tmp/extracted_model.onnx",
        input_names=list(feed_input_names),
        output_names=[outp.name for outp in model.graph.output],
    )
    print("Model extraction successful!\n")
except Exception:
    print("Model extraction failed with error:")
    print(traceback.format_exc())
