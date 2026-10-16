import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    MLP Layer used after graph vector representation
"""

class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2): #L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [nn.Linear(input_dim, 20)]
        list_FC_layers += [ nn.Linear( 20, 20 , bias=True ) for l in range(L - 1) ]
        list_FC_layers.append(nn.Linear( 20 , output_dim , bias=True ))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.drop_out = nn.Dropout(0.1)
        self.L = L
        
    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
            y = self.drop_out(y)
        y = self.FC_layers[self.L](y)
        return y