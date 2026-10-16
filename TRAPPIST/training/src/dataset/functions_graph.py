import numpy as np
import torch
import dgl
from src.dataset.functions_data import (
    calculate_distance_to_boundary,
)
import time
from src.dataset.functions_particles import concatenate_Particles_GT, Particles_GT
from src.dataset.utils_hits import create_noise_label
from src.dataset.dataclasses import Hits





def create_graph(
    output,
    for_training =True 
):
    prediction = not for_training
    graph_empty = False
   
    hits = Hits.from_data(
    output,
    )

    g = dgl.graph(([], []))
    g.add_nodes(hits.hit_features.shape[0])
    g.ndata["h"] = hits.hit_features.float() 
    y = torch.max(torch.Tensor(output["y_event_type"])).view(-1)
    
    # added to have the info in testing output for binning on variables
    mc_energy    = float(output["z_mc_energy"][0])
    pi0_angle    = float(output["z_pi0_angle"][0])
    pi0_distance = float(output["z_pi0_distance"][0])

    return ([g, y], mc_energy, pi0_angle, pi0_distance, graph_empty)


def graph_batch_func(list_graphs):
    """collator function for graph dataloader

    Args:
        list_graphs (list): list of graphs from the iterable dataset

    Returns:
        batch dgl: dgl batch of graphs
    """
    ##list_graphs_g = [el[0] for el in list_graphs]
    ##list_y = [el[1] for el in list_graphs]
    list_graphs_g = [el[0][0] for el in list_graphs]
    list_y        = [el[0][1] for el in list_graphs]
    mc_energies   = [el[1] for el in list_graphs]
    pi0_angles    = [el[2] for el in list_graphs]
    pi0_distances = [el[3] for el in list_graphs]

    
    ys = torch.cat(list_y, dim=0)
    # ys = torch.reshape(ys, [-1, list_y[0].shape[1]])
    bg = dgl.batch(list_graphs_g)
    # reindex particle number
    
    mc_energies   = torch.tensor(mc_energies,   dtype=torch.float32)
    pi0_angles    = torch.tensor(pi0_angles,    dtype=torch.float32)
    pi0_distances = torch.tensor(pi0_distances, dtype=torch.float32)

    
    #return bg, ys
    return bg, ys, mc_energies, pi0_angles, pi0_distances

