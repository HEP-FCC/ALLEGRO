import torch
from src.models.MixedModels import Mixed_EGNN


class MixedEGNNGATNetWrapper(torch.nn.Module):
    def __init__(self, dev, **kwargs) -> None:
        super().__init__()
        self.mod = Mixed_EGNN(dev)

    def forward(self, g):
        return self.mod(g)


def get_model(data_config, dev, **kwargs):

    # pf_features_dims = len(data_config.input_dicts['pf_features'])
    # num_classes = len(data_config.label_value)
    model = MixedEGNNGATNetWrapper(dev)

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
