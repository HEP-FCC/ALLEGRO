import torch
from src.models.EGNN_dgl import EGNN


class EGNNNetWrapper(torch.nn.Module):
    def __init__(self, args, dev, **kwargs) -> None:
        super().__init__()
        self.mod = EGNN(args, dev, separate_heads=kwargs.get("separate_heads", False))

    def forward(self, g, step_count):
        return self.mod(g, step_count)


def get_model(data_config, args, dev, **kwargs):

    # pf_features_dims = len(data_config.input_dicts['pf_features'])
    # num_classes = len(data_config.label_value)
    separate_heads = kwargs.get("separate_heads", False)
    model = EGNNNetWrapper(args, dev, separate_heads=separate_heads)

    model_info = {
        "input_names": list(data_config.input_names),
        "input_shapes": {
            k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()
        },
        "output_names": ["softmax"],
        "dynamic_axes": {
            **{k: {0: "N", 2: "n_" + k.split("_")[0]} for k in data_config.input_names},
            **{"softmax": {0: "N"}},
        },
    }

    return model, model_info


def get_loss(data_config, **kwargs):
    return torch.nn.CrossEntropyLoss()
