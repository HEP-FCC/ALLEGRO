from dataclasses import dataclass
from typing import Any, List, Optional
import torch 




@dataclass
class Hits:
    hit_features: Any
    
    @classmethod
    def from_data(cls, output):
 
        X_hit = torch.tensor(output["X_hit"])
       
        # obtain the position of the hits and the energies and p
        hit_features = X_hit[:,:]
    
    
        
        return cls(
            hit_features = hit_features, 

        )



