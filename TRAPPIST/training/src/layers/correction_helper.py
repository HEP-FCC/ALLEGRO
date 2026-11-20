from os import path
import sys

from time import time
from gatr import GATr, SelfAttentionConfig, MLPConfig
from gatr.interface import embed_point, extract_scalar, extract_point, embed_scalar
from torch_scatter import scatter_add, scatter_mean
import torch
import torch.nn as nn
from src.utils.save_features import save_features
import numpy as np
from typing import Tuple, Union, List
import dgl


from src.models.gravnet_3_L import obtain_clustering_for_matched_showers
from src.utils.post_clustering_features import (
    get_post_clustering_features,
    calculate_eta,
    calculate_phi,
)
from src.models.energy_correction_NN import (
    ECNetWrapper,
    ECNetWrapperGNN,
    ECNetWrapperGNNGlobalFeaturesSeparate,
    PickPAtDCA,)
import lightning as L

import os
import wandb


def clustering_and_global_features():
    time_matching_start = time()
    # Match graphs
    (
        graphs_new,
        true_new,
        sum_e,
        true_pid,
        e_true_corr_daughters,
        true_coords,
    ) = obtain_clustering_for_matched_showers(
        g,
        x,
        y,
        self.trainer.global_rank,
        use_gt_clusters=self.args.use_gt_clusters,
    )
    time_matching_end = time()
    wandb.log(
        {"time_clustering_matching": time_matching_end - time_matching_start}
    )