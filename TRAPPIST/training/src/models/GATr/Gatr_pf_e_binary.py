from os import path
import sys

# sys.path.append(
#     path.abspath("/afs/cern.ch/work/m/mgarciam/private/geometric-algebra-transformer/")
# )
# sys.path.append(path.abspath("/mnt/proj3/dd-23-91/cern/geometric-algebra-transformer/"))
from time import time
from gatr import GATr, SelfAttentionConfig, MLPConfig
from gatr.interface import embed_point, extract_scalar, extract_point, embed_scalar
from torch_scatter import scatter_add, scatter_mean, scatter_max
import torch
import torch.nn as nn
from src.utils.save_features import save_features
from src.logger.plotting_tools import PlotCoordinates
import numpy as np
from typing import Tuple, Union, List
import dgl
from src.logger.plotting_tools import PlotCoordinates
from src.models.mlp_readout_layer import MLPReadout

import lightning as L
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from src.layers.inference_oc import create_and_store_graph_output
from xformers.ops.fmha import BlockDiagonalMask
import os
import wandb

import torch.nn.functional as F
import pandas as pd

from sklearn.utils.class_weight import compute_class_weight

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


# This is not a wrapper: this is the model!!!
class ExampleWrapper(L.LightningModule):
    """Example wrapper around a GATr model.

    Expects input data that consists of a point cloud: one 3D point for each item in the data.
    Returns outputs that consists of one scalar number for the whole dataset.

    Parameters
    ----------
    blocks : int
        Number of transformer blocks
    hidden_mv_channels : int
        Number of hidden multivector channels
    hidden_s_channels : int
        Number of hidden scalar channels
    """

    def __init__(
        self,
        args,
        dev,
    ):
        super().__init__()
        self.strict_loading = False
        blocks = 8 # original implementation uses 8 blocks 
        hidden_mv_channels = 16
        hidden_s_channels = 16
        self.input_dim = 3       
        self.output_dim = 2    
        self.loss_final = 0
        self.number_b = 0
        self.args = args
        self.ScaledGooeyBatchNorm2_1 = nn.BatchNorm1d(self.input_dim, momentum = 0.1) #batch normalization
        self.gatr = GATr(
            in_mv_channels = 1,
            out_mv_channels = 1,
            #multivector channels: 16-components geometric algebra objects, to capture spatial infos
            hidden_mv_channels = hidden_mv_channels,
            # scalar channels: no geometric structure, should be used to capture non geometric properties
            in_s_channels = None,
            out_s_channels = None,
            hidden_s_channels = hidden_s_channels,
            num_blocks = blocks,
            attention = SelfAttentionConfig(),  # Use default parameters for attention
            mlp = MLPConfig(),  # Use default parameters for MLP
        )
        # To handle class imbalance weights (number of events per each class) can be set when launching then training script 
        n_pi0_events = getattr(self.args, "pi0_events", None)
        n_gamma_events = getattr(self.args, "gamma_events", None)
        print(n_pi0_events, n_gamma_events)
        if n_pi0_events is not None and n_gamma_events is not None:
            pos_weight = torch.tensor([n_pi0_events/n_gamma_events, n_gamma_events/n_pi0_events], dtype=torch.float32)
            self.loss_crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(dev))

        self.readout = "sum"   # sum pooling: add features over all points
        self.MLP_layer = MLPReadout(16, 2)   # gives the final output decision



    def forward(self, g, step_count, mask, labels, eval="", return_train=False):
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

        # g is a graph object
        # tensor of hit position x,y,z 
        position = g.ndata["h"][:,1:].contiguous()   
        # tensor of hit energy
        energy = g.ndata["h"][:,0].contiguous().view(-1, 1)  

        # batch normalization for spatial inputs (energy is not normalized)
        inputs = self.ScaledGooeyBatchNorm2_1(position) 
        embedded_inputs = embed_point(inputs) + embed_scalar(energy)
        # scalar channel is not used in this example
        scalars = torch.zeros((inputs.shape[0], 1))

        embedded_inputs = embedded_inputs.unsqueeze(-2).contiguous()  

        # Pass data through GATr, _ is the attention weight (ignored)
        embedded_outputs, _ = self.gatr(
            embedded_inputs, scalars = scalars, attention_mask=mask
        )  
        
        # new embedding for each node, assigned to the new vector "h_"
        g.ndata["h_"] = embedded_outputs[:, 0, :].contiguous()
        
        # sum pooling of all the embeddings
        hg = dgl.sum_nodes(g, "h_").contiguous()

        #Multi-layer perceptron, final decision head of the model
        all_features = self.MLP_layer(hg)

        return all_features


    def build_attention_mask(self, g):
        """Construct attention mask from pytorch geometric batch.

        Parameters
        ----------
        inputs : torch_geometric.data.Batch
            Data batch.

        Returns
        -------
        attention_mask : xformers.ops.fmha.BlockDiagonalMask
            Block-diagonal attention mask: within each sample, each token can attend to each other
            token.
        """
        batch_numbers = obtain_batch_numbers(g)

        return (
            BlockDiagonalMask.from_seqlens(
                torch.bincount(batch_numbers.long()).tolist()
            ),
            batch_numbers,
        )


    def training_step(self, batch, batch_idx):
        # batch = grouping several events 
        y = batch[1]        # labels
        batch_g = batch[0]  # graph input data
        
        # metadata unpacked but not used in training
        _mc_energy, _pi0_angle, _pi0_distance = batch[2], batch[3], batch[4]

        # mask = allows to connect different hits (since they are part of the same cluster)
        # block diagonal matrix telling the cumulative starting position of each graph's node sequence in the batch
        mask, labels = self.build_attention_mask(batch_g)

        # this is the forward step
        model_output = self(batch_g, batch_idx, mask, labels)

        # loss calculation for BCEWithLogitsLoss
        loss = self.loss_crit(model_output, 
                    1.0 * F.one_hot(y.view(-1).long(), num_classes=2))
        
        acc = torch.mean(1.0 * (model_output.argmax(axis=1) == y.view(-1)))
        self.log('train_loss_step', loss, on_step=True, on_epoch=False, sync_dist=True)
        self.log('train_loss_epoch', loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('train_acc_step', acc, on_step=True, on_epoch=False, sync_dist=True)
        self.log('train_acc_epoch', acc, on_step=False, on_epoch=True, sync_dist=True)
        
        del model_output
        return loss



    def validation_step(self, batch, batch_idx):
        
        self.validation_step_outputs = [] 
        y = batch[1] #labels
        batch_g = batch[0] # graph input data
        
        # metadata, not used in loss/forward
        mc_energy    = batch[2]  
        pi0_angle    = batch[3]
        pi0_distance = batch[4]

        mask, labels = self.build_attention_mask(batch_g)

        # forward step
        model_output_val = self(batch_g, 1, mask, labels)

        # accumulate predictions (all ranks!)
        self.val_logits.append(model_output_val.detach().cpu())
        self.val_labels.append(y.view(-1).detach().cpu())        

        #! loss calculation for BCEWithLogitsLoss 
        loss = self.loss_crit(model_output_val, 
                    1.0 * F.one_hot(y.view(-1).long(), num_classes=2))

        if self.args.predict:
            d = {
                "photon": model_output_val.detach().cpu()[:, 0].view(-1),
                "pi0": model_output_val.detach().cpu()[:, 1].view(-1),
                "labels_true": y.detach().cpu().view(-1),
                # metadata
                "mc_energy":    mc_energy.detach().cpu().view(-1),   
                "pi0_angle":    pi0_angle.detach().cpu().view(-1),
                "pi0_distance": pi0_distance.detach().cpu().view(-1),
            }

            df = pd.DataFrame(data=d)
            #print("dataframe: ", df)
            self.eval_df.append(df)

        acc = torch.mean(1.0 * (model_output_val.argmax(axis=1) == y.view(-1)))
        # Log loss (for epoch-average)
        self.log("loss_val_epoch", loss, on_step=False, on_epoch=True, sync_dist=True)
        # Log accuracy (for epoch-average)
        self.log("accuracy_val", acc, on_step=False, on_epoch=True, sync_dist=True)



        del loss
        del model_output_val

    # hook called at the end of every training epoch
    def on_train_epoch_end(self):
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
            # in model-prefix!
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

            # same score as the C++ inference: sigmoid(logit_pi0 - logit_gamma)
            logit_gamma = logits[:, 0]
            logit_pi0 = logits[:, 1]
            score_pi0 = torch.sigmoid(logit_pi0 - logit_gamma)

            # working point: score > 0.5  <=>  logit_pi0 > logit_gamma
            preds = (score_pi0 > 0.5).long()

            y_truth = labels.detach().cpu().numpy()   # 0 = photon, 1 = pi0
            y_score = score_pi0.detach().cpu().numpy()
            y_preds = preds.detach().cpu().numpy()

            # images logged under the same key every epoch -> step slider in wandb
            epoch = self.current_epoch
            class_names = ["photon", "pi0"]
            log_dict = {"epoch": epoch}

            # confusion matrix, rows normalized to the true class
            counts = np.zeros((2, 2), dtype=np.int64)
            np.add.at(counts, (y_truth.astype(int), y_preds.astype(int)), 1)
            row_sums = counts.sum(axis=1, keepdims=True)
            fractions = np.divide(counts, row_sums, out=np.zeros((2, 2)), where=row_sums > 0)

            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(fractions, vmin=0, vmax=1, cmap="Blues")
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{fractions[i, j]:.3f}\n({counts[i, j]})", ha="center", va="center",
                            color="white" if fractions[i, j] > 0.5 else "black")
            ax.set_xticks([0, 1])
            ax.set_xticklabels(class_names)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(class_names)
            ax.set_xlabel("predicted")
            ax.set_ylabel("true")
            ax.set_title(f"Confusion matrix (score > 0.5), epoch {epoch}")
            log_dict["confusion_matrix"] = wandb.Image(fig)
            plt.close(fig)

            # ROC/AUC need both classes, otherwise roc_auc_score raises
            if len(np.unique(y_truth)) == 2:
                # pi0 = positive class (label 1)
                fpr, tpr, _ = roc_curve(y_truth, y_score, pos_label=1)
                auc = roc_auc_score(y_truth, y_score)

                fig, ax = plt.subplots(figsize=(6, 6))
                ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}")
                ax.plot([0, 1], [0, 1], "k--", lw=1)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1.02)
                ax.set_xlabel("photon misID rate (fpr)")
                ax.set_ylabel("pi0 efficiency (tpr)")
                ax.set_title(f"ROC pi0 vs photon, epoch {epoch}")
                ax.legend(loc="lower right")
                ax.grid(True, alpha=0.3)
                log_dict["roc_curve"] = wandb.Image(fig)
                log_dict["auc"] = auc
                plt.close(fig)

            # one call per epoch, so ROC and confusion matrix share the same slider step
            wandb.log(log_dict)



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
                "monitor": "train_loss_epoch",
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

