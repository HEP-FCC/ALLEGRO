import torch
import torch.nn as nn
import os
import lightning as L
from torch.optim.lr_scheduler import ReduceLROnPlateau

## from src.logger.logger_wandb import log_losses_wandb_tracking
## from src.layers.inference_oc_tracks import (
##     evaluate_efficiency_tracks,
##     store_at_batch_end,
##     store_at_batch_end_hits
## )
## from src.layers.losses import object_condensation_loss_tracking
## from src.layers.batch_operations import obtain_batch_numbers

from src.gatr_v111_onnx.nets.gatr import GATr
from src.gatr_v111_onnx.layers.attention.config import SelfAttentionConfig
from src.gatr_v111_onnx.layers.mlp.config import MLPConfig
from src.gatr_v111_onnx.interface import (
    embed_point,
    embed_scalar,
    embed_translation,
)
from src.gatr_v111_onnx.primitives.invariants import   compute_inner_product_mask
from src.gatr_v111_onnx.primitives.linear import _compute_pin_equi_linear_basis
from src.gatr_v111_onnx.primitives.attention import _build_dist_basis

#-----Jaco imports------

from os import path
import sys

# sys.path.append(
#     path.abspath("/afs/cern.ch/work/m/mgarciam/private/geometric-algebra-transformer/")
# )
# sys.path.append(path.abspath("/mnt/proj3/dd-23-91/cern/geometric-algebra-transformer/"))
from time import time

from torch_scatter import scatter_add, scatter_mean, scatter_max
import torch
import torch.nn as nn
## from src.utils.save_features import save_features
## from src.logger.plotting_tools import PlotCoordinates
import numpy as np
from typing import Tuple, Union, List
import dgl
## from src.logger.plotting_tools import PlotCoordinates
from src.models.mlp_readout_layer import MLPReadout

import lightning as L
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
## from src.layers.inference_oc import create_and_store_graph_output
## from xformers.ops.fmha import BlockDiagonalMask
import os

import torch.nn.functional as F
import pandas as pd

from sklearn.utils.class_weight import compute_class_weight

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

class ExampleWrapper(L.LightningModule):  # nn.Module L.LightningModule
    def __init__(
        self,
        args,
        dev,
    ):
        super().__init__()
        self.strict_loading = False
        blocks = 2
        hidden_mv_channels = 16
        hidden_s_channels = 16
        self.input_dim = 3
        self.output_dim = 8 # originally 8 blocks
        self.loss_final = 0
        self.number_b = 0
        self.args = args
        self.basis_gp = None
        self.basis_outer = None
        self.pin_basis = None
        self.basis_q = None
        self.basis_k = None
        self.basis_gp_mask = None
        self.ScaledGooeyBatchNorm2_1 = nn.BatchNorm1d(self.input_dim, momentum=0.1)
        # load_basis is the main difference wrt gatr_pf_e.py: we load basis for geometric and outer product here to avoid ONNX incompatibility issues
        self.load_basis()
        self.gatr = GATr(
            in_mv_channels=1,
            out_mv_channels=1,
            hidden_mv_channels=hidden_mv_channels,
            in_s_channels=None,
            out_s_channels=None,
            hidden_s_channels=hidden_s_channels,
            num_blocks=blocks,
            attention=SelfAttentionConfig(),
            mlp=MLPConfig(),
            basis_gp=self.basis_gp,
            basis_outer=self.basis_outer,
            basis_pin=self.pin_basis,
            basis_q=self.basis_q,
            basis_k=self.basis_k,
            basis_gp_mask = self.basis_gp_mask, 
        )
        # n_pi0_events = self.args.pi0_events
        # n_gamma_events = self.args.gamma_events
        n_pi0_events = 1
        n_gamma_events = 1
        # print(n_pi0_events, n_gamma_events)
        pos_weight = torch.tensor([n_pi0_events/n_gamma_events, n_gamma_events/n_pi0_events], dtype=torch.float32)
        self.loss_crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(dev))
        self.readout = "sum"   
        self.MLP_layer = MLPReadout(16, 2)   



    def load_basis(self):

        this_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.abspath(os.path.join(this_dir, "../../../"))
        
        filename = os.path.join(parent_dir, "gatr_utils/geometric_product.pt")
        sparse_basis = torch.load(filename).to(torch.float32)
        basis = sparse_basis.to_dense()
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.basis_gp = basis.to(device=device)
        
        filename = os.path.join(parent_dir, "gatr_utils/outer_product.pt")
        sparse_basis_outer = torch.load(filename).to(torch.float32)
        sparse_basis_outer = sparse_basis_outer.to_dense()
        self.basis_outer = sparse_basis_outer.to(device=device)

        self.pin_basis = _compute_pin_equi_linear_basis(
            device=self.basis_gp.device, dtype=basis.dtype
        )
        self.basis_q, self.basis_k = _build_dist_basis(
            device=self.basis_gp.device, dtype=basis.dtype
        )
        mask = compute_inner_product_mask(self.basis_gp, device=self.basis_gp.device)
        columns = torch.arange(0, 16).to(self.basis_gp.device)
        colums_take = columns[mask.bool()]
        self.basis_gp_mask = colums_take

    #def forward(self, g, step_count, mask, labels, eval="", return_train=False):
    def forward(self, position, energy):
        """Forward pass.

        Parameters
        ----------
        inputs : torch.Tensor with shape (*batch_dimensions, num_points, 3)
            Point cloud input data

        Returns
        -------
        outputs : torch.Tensor with shape (*batch_dimensions, 1)
            Model prediction: a single scalar for the whole point cloud.
        """
        # node features → transformer → node embeddings → sum pooling → hg → MLP → output

        # g is a graph object, .ndata to access the node features. every node has a feature vector h
        # tensor of hit position x,y,z 
        position = self.ScaledGooeyBatchNorm2_1(position)   
        embedded_inputs = embed_point(position) + embed_scalar(energy)

        #print("embedded_inputs: ", embedded_inputs)
        #print("size embedded_inputs: ", embedded_inputs.size())

        ## scalars = torch.zeros((position.shape[0], 1))
        #scalars = torch.zeros_like(position[:, :1])
        scalars = torch.zeros(
            (position.shape[0], 1),
            device=position.device,
            dtype=position.dtype
        )

        # dim: (batch_size*num_points, 1, 16)
        embedded_inputs = embedded_inputs.unsqueeze(-2) 

        #print("unsq_embedded_inputs: ", embedded_inputs)
        #print("size unsq_embedded_inputs: ", embedded_inputs.size())

        # Pass data through GATr, _ is the attention weight (ignored)
        embedded_outputs, _ = self.gatr(
            embedded_inputs, scalars=scalars)  # (..., num_points, 1, 16)
        
        # new embedding for each node, assigned to the new vector "h_"
        #dim: (batch_size*num_points, 16)
        nodes_output = embedded_outputs[:, 0, :]

        #print("g.ndata[h_]: ", g.ndata["h_"])
        #print("size: ", g.ndata["h_"].size())
        
        #dim: (batch_size, 16), after sum pooling of all the embeddings
        pooled_output = nodes_output.sum(dim=0, keepdim=True)  # sum pooling over all nodes

        #print("hg: ", hg)
        #print("size: ", hg.size())

        #Multi-layer perceptron, final decision head of the model
        out = self.MLP_layer(pooled_output)

        #print("all features: ", all_features)
        return out
    
    
    
    ##def build_attention_mask(self, g):
    ##    """Construct attention mask from pytorch geometric batch.
##
    ##    Parameters
    ##    ----------
    ##    inputs : torch_geometric.data.Batch
    ##        Data batch.
##
    ##    Returns
    ##    -------
    ##    attention_mask : xformers.ops.fmha.BlockDiagonalMask
    ##        Block-diagonal attention mask: within each sample, each token can attend to each other
    ##        token.
    ##    """
    ##    batch_numbers = obtain_batch_numbers(g)
##
    ##    return (
    ##        BlockDiagonalMask.from_seqlens(
    ##            torch.bincount(batch_numbers.long()).tolist()
    ##        ),
    ##        batch_numbers,
    ##    )
    

    def training_step(self, batch, batch_idx):
        # print(f"TRAIN: Rank {self.trainer.global_rank} / World Size {self.trainer.world_size} is running batch {batch_idx}")
        # batch = grouping several events 
        y = batch[1]        # labels
        batch_g = batch[0]  # graph input data
        
        # metadata unpacked but not used in training
        _mc_energy, _pi0_angle, _pi0_distance = batch[2], batch[3], batch[4]

        h = batch_g.ndata["h"]  # node features
        position = h[:, 1:].contiguous()  # node positions
        energy = h[:, 0].contiguous().view(-1, 1)
        
        
        model_output = self(position, energy)

        #! loss calculation for BCEWithLogitsLoss (onehot encoding still needed)
        loss = self.loss_crit(model_output, 
                    1.0 * F.one_hot(y.view(-1).long(), num_classes=2))
        
        acc = torch.mean(1.0 * (model_output.argmax(axis=1) == y.view(-1)))
        del model_output
        return loss



##    def validation_step(self, batch, batch_idx):
        
##        self.validation_step_outputs = [] # useless?
##        y = batch[1] #labels
##        batch_g = batch[0] # graph input data
##        
##        # metadata, not used in loss/forward
##        mc_energy    = batch[2]  
##        pi0_angle    = batch[3]
##        pi0_distance = batch[4]
##
##        # mask, labels = self.build_attention_mask(batch_g)
##
##        # forward step, why no batch_idx?
##        model_output_val = self(batch_g, 1, mask, labels)
##
##        # accumulate predictions (all ranks!)
##        self.val_logits.append(model_output_val.detach().cpu())
##        self.val_labels.append(y.view(-1).detach().cpu())        
##
##        #if y == 0 (photon) then [1,0], if y == 1 (pi0) then [0,1]
##        #! loss calculation for BCELoss (sigmoid needed)
##        #! loss = self.loss_crit(
##        #!     #torch.sigmoid(model_output), this was used, but I changed it for consistency with training step
##        #!     self.m(model_output_val),
##        #!     1.0 * F.one_hot(y.view(-1).long(), num_classes=2),
##        #! )
##
##        #! loss calculation for BCEWithLogitsLoss (onehot encoding still needed)
##        loss = self.loss_crit(model_output_val, 
##                    1.0 * F.one_hot(y.view(-1).long(), num_classes=2))
##
##        if self.args.predict:
##            d = {
##                "photon": model_output_val.detach().cpu()[:, 0].view(-1),
##                "pi0": model_output_val.detach().cpu()[:, 1].view(-1),
##                "labels_true": y.detach().cpu().view(-1),
##                # metadata
##                "mc_energy":    mc_energy.detach().cpu().view(-1),   
##                "pi0_angle":    pi0_angle.detach().cpu().view(-1),
##                "pi0_distance": pi0_distance.detach().cpu().view(-1),
##            }
##
##            df = pd.DataFrame(data=d)
##            #print("dataframe: ", df)
##            self.eval_df.append(df)
##
##        # if self.trainer.is_global_zero:
### 
##        #     # print(model_output)
##        #     # print(labels_true)
##        #     
##        #     # in this way, wandb is showing only the last loss/accuracy of the epoch
##        #     wandb.log({"loss_validation": loss.item()})
### 
##        #     acc = torch.mean(1.0 * (model_output_val.argmax(axis=1) == y.view(-1)))
##        #     # print(acc)
##        #     wandb.log({"accuracy_validation ": acc.item()})
##
##
##
##        ## if self.trainer.is_global_zero:
##        ##     # wandb.log({"loss": loss.item()})
#### 
##        ##     # accuracy computation: model_output.argmax(axis=1) picks the index of largest value per row (the predicted class)
##        ##     # then, this is compared with true labels, giving a boolean tensor, which is multiplied by one (true = 1, false = 0) and then averaged
##        ##     # wandb.log({"accuracy": acc.item()})
##        ##     wandb.log({"loss_val": loss.item()})
##        ##     wandb.log({"accuracy_val": acc.item()})
##        
##        acc = torch.mean(1.0 * (model_output_val.argmax(axis=1) == y.view(-1)))
##        # Log loss (for epoch-average)
##        self.log("loss_val", loss, on_step=False, on_epoch=True, sync_dist=True)
##        # Log accuracy (for epoch-average)
##        self.log("accuracy_val", acc, on_step=False, on_epoch=True, sync_dist=True)
##
##
##
##        del loss
##        del model_output_val

    # hook called at the end of every training epoch
    def on_train_epoch_end(self):
        # total loss across batches in the epoch / number of batches 
        ## self.log("train loss epoch", self.loss_final / self.number_b, sync_dist=True)
        pass 

    


    def on_train_epoch_start(self):
        # if self.trainer.is_global_zero and self.current_epoch == 0:
        #     self.stat_dict = {}
        self.make_mom_zero()

    def on_validation_epoch_start(self):
        self.make_mom_zero()
        self.eval_df = []

        self.val_labels = []
        self.val_logits = []

    def on_validation_epoch_end(self):
        # if prediction, concatenate all batches and output a file with the logits
        if self.args.predict:
            df_batch1 = pd.concat(self.eval_df)
            # why in model-prefix and not in predict-output (unused)
            df_batch1.to_pickle(self.args.model_prefix + "/model_output_eval_logits.pt") 

        # concatenate different batches 
        logits = torch.cat(self.val_logits)     # local
        labels = torch.cat(self.val_labels)     # local

        #print("logits: ", logits)
        #print("logitss size: ", logits.size())

        # gather from all ranks (= processes across different gpus)
        logits = self.all_gather(logits)
        labels = self.all_gather(labels)

        #print("logits then: ", logits)
        #print("logitss size then: ", logits.size())

        # only rank 0 (the main process) logs metrics
        if self.trainer.is_global_zero:

            # reshape after gather: (world_size, N, num_classes)
            logits = logits.reshape(-1, logits.shape[-1])
            labels = labels.reshape(-1)

            # Apply softmax to get probabilities, maybe should use sigmoid?
            probs = torch.softmax(logits, dim=1)

            # returns the index of the maximum value (the prediction)
            preds = probs.argmax(dim=1)

            #print("y_true ", labels.detach().cpu().numpy())
            #print("y_probas ", probs.detach().cpu().numpy())
            #print("y_probas[:, 0] ", probs.detach().cpu().numpy()[:, 0])
            #print("preds ", preds.detach().cpu().numpy())

            y_truth = labels.detach().cpu().numpy()
            y_probs_gamma_pi0 = probs.detach().cpu().numpy() #first column gamma, second column pi0
            y_preds = preds.detach().cpu().numpy()

            y_probs_pi0 = y_probs_gamma_pi0[:, 1]
            y_probs_gamma = y_probs_gamma_pi0[:, 0]

            #print("y_truth ", y_truth.shape)
            #print("y_probs_pi0 ", y_probs_pi0.shape)
            #print("y_probs_gamma ", y_probs_gamma.shape)

            label_map = {0: "photon", 1: "pi0"}
            y_truth_named = np.array([label_map[int(y)] for y in y_truth])





    def make_mom_zero(self):
        if self.current_epoch > 1 or self.args.predict:
            print("Making momentum 0")
            self.ScaledGooeyBatchNorm2_1.momentum = 0
            
    # Optimizer is Adam, scheduler is ReduceLROnPlateau
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.args.start_lr
        )
        print("Optimizer params:", filter(lambda p: p.requires_grad, self.parameters()))
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, patience=3),
                "interval": "epoch",
                "monitor": "train loss epoch",
                "frequency": 1
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }

# return a tensor containing numbers from0 to batch_dim
# to connect different hits (= graphs) in the same cluster ("number" in the batch)
def obtain_batch_numbers(g):
    graphs_eval = dgl.unbatch(g)
    #print("graphs_eval: ", graphs_eval)
    number_graphs = len(graphs_eval)
    batch_numbers = []
    for index in range(0, number_graphs):
        gj = graphs_eval[index]
        num_nodes = gj.number_of_nodes()
        batch_numbers.append(index * torch.ones(num_nodes))
        num_nodes = gj.number_of_nodes()
    batch = torch.cat(batch_numbers, dim=0)
    #print("batch: ", batch)
    #print("batch size: ", batch.size())

    return batch
