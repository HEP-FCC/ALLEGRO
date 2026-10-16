import pandas as pd

obj = pd.read_pickle("model_output_eval_logits_40epochs.pt")

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

print(type(obj))
print(obj)