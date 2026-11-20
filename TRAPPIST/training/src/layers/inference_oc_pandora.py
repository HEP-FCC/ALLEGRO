import dgl
import torch
import os

# from alembic.command import current
from sklearn.cluster import DBSCAN, HDBSCAN
from torch_scatter import scatter_max, scatter_add, scatter_mean
import numpy as np
from src.dataset.functions_data import CachedIndexList
from src.dataset.config_main.functions_data import spherical_to_cartesian
from src.dataset.utils_hits import CachedIndexList
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import pandas as pd
import wandb
from src.utils.inference.per_particle_metrics import plot_event
import random
import string

from src.layers.inference_oc import get_labels_pandora, match_showers, generate_showers_data_frame


def create_and_store_graph_output(
    batch_g,
    y,
    batch_idx,
    path_save,
    total_number_events,
):
    step = batch_idx
    graphs = dgl.unbatch(batch_g)
    batch_id = y.batch_number.view(-1)  # y[:, -1].view(-1)
    df_list_pandora = []
    local_rank = 0
    for i in range(0, len(graphs)):
        mask = batch_id == i
        dic = {}
        dic["graph"] = graphs[i]
        y1 = y.copy()
        y1.mask(mask)
        dic["part_true"] = y1  # y[mask]
        X = dic["graph"].ndata["pos_hits_xyz"] 
        labels_pandora = get_labels_pandora(True, dic, X.device)
        particle_ids = torch.unique(dic["graph"].ndata["particle_number"])
      
        (
            shower_p_unique_pandora,
            row_ind_pandora,
            col_ind_pandora,
            i_m_w_pandora,
            iou_m_pandora,
        ) = match_showers(
            labels_pandora,
            dic,
            particle_ids,
            X,
            local_rank,
            i,
            path_save,
            pandora=True,
            tracks=True,
        )

       
        if len(shower_p_unique_pandora) > 1:
                df_event_pandora = generate_showers_data_frame(
                    labels_pandora,
                    dic,
                    shower_p_unique_pandora,
                    particle_ids,
                    row_ind_pandora,
                    col_ind_pandora,
                    i_m_w_pandora,
                    pandora=True,
                    tracking=False,
                    step=step,
                    number_in_batch=total_number_events,
                    tracks=True,
                    save_plots_to_folder=False,
                )
                if df_event_pandora is not None and type(df_event_pandora) is not tuple:
                    df_list_pandora.append(df_event_pandora)
                else:
                    print("Not appending to df_list_pandora")
                total_number_events = total_number_events + 1

    
    df_batch_pandora = pd.concat(df_list_pandora)
    return df_batch_pandora,  total_number_events
  


def store_at_epoch_end(
    path_save,
    df_batch_pandora,
    local_rank=0,
    step=0,
):
 

    path_save_pandora = (
        path_save
        + "/"
        + str(local_rank)
        + "_"
        + str(step)
        + "_"
        + str(0)
        + "_pandora.pt"
    )
        
    df_batch_pandora.to_pickle(path_save_pandora)






