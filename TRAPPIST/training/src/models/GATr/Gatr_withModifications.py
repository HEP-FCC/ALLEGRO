import torch
import torch.nn as nn
import os
import lightning as L
from torch.optim.lr_scheduler import ReduceLROnPlateau
import sys
import torch.distributed as dist
import numpy as np
# from src.logger.logger_wandb import log_losses_wandb_tracking
#from src.layers.inference_oc_tracks import (
#    evaluate_efficiency_tracks,
#    store_at_batch_end,
#    store_at_batch_end_hits,
#    get_clustering
#)
#from src.layers.losses import object_condensation_loss_tracking
#from src.layers.batch_operations import obtain_batch_numbers

import dgl
from src.models.mlp_readout_layer import MLPReadout
import torch.nn.functional as F

from src.gatr_v111.nets.gatr import GATr
from src.gatr_v111.layers.attention.config import SelfAttentionConfig
from src.gatr_v111.layers.mlp.config import MLPConfig
from src.gatr_v111.interface import (
    embed_point,
    embed_scalar,
    embed_translation,
)
from xformers.ops.fmha import BlockDiagonalMask
from src.gatr_v111.primitives.invariants import   compute_inner_product_mask
from src.gatr_v111.primitives.linear import _compute_pin_equi_linear_basis
from src.gatr_v111.primitives.attention import _build_dist_basis

import wandb
#from src.logger.plotting_tools import (PlotCoordinates, efficiency_purity_plot)
import pandas as pd

class ExampleWrapper(L.LightningModule): 
    def __init__(
        self,
        args,
        dev,
    ):
        super().__init__()
        blocks = 8
        hidden_mv_channels = 16
        hidden_s_channels = 16
        self.input_dim = 3
        self.output_dim = 2
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
        n_pi0_events = getattr(self.args, "pi0_events", None)
        n_gamma_events = getattr(self.args, "gamma_events", None)
        print(n_pi0_events, n_gamma_events)
        if n_pi0_events is not None and n_gamma_events is not None:
            pos_weight = torch.tensor([n_pi0_events/n_gamma_events, n_gamma_events/n_pi0_events], dtype=torch.float32)
            self.loss_crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(dev))


        self.readout = "sum"   # sum pooling: add features over all points
        self.MLP_layer = MLPReadout(16, 2)   # gives the final output decision, first column for index 0, second for index 1


    def load_basis(self):

        filename = "gatr_utils/geometric_product.pt"
        sparse_basis = torch.load(filename).to(torch.float32)
        basis = sparse_basis.to_dense()
        self.basis_gp = basis.to(device="cuda")
        filename = "gatr_utils/outer_product.pt"
        sparse_basis_outer = torch.load(filename).to(torch.float32)
        sparse_basis_outer = sparse_basis_outer.to_dense()
        self.basis_outer = sparse_basis_outer.to(device="cuda")

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

    def forward(self, g, step_count, mask, labels, eval="", return_train=False):  
        position = g.ndata["h"][:,1:].contiguous() 
        energy = g.ndata["h"][:,0].contiguous().view(-1, 1)      
        inputs = self.ScaledGooeyBatchNorm2_1(position)
        embedded_inputs = embed_point(inputs) + embed_scalar(energy)

        embedded_inputs = embedded_inputs.unsqueeze(-2).contiguous()
        scalars = torch.zeros((inputs.shape[0], 1))
        embedded_outputs, _ = self.gatr(
            embedded_inputs, scalars=scalars, attention_mask=mask
        )
        g.ndata["h_"] = embedded_outputs[:, 0, :].contiguous()
        hg = dgl.sum_nodes(g, "h_").contiguous()
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
            batch_numbers
        )

    def training_step(self, batch, batch_idx):
        y = batch[1]
        batch_g = batch[0]
        _mc_energy, _pi0_angle, _pi0_distance = batch[2], batch[3], batch[4]
        mask, labels = self.build_attention_mask(batch_g)
        if self.trainer.is_global_zero:
            model_output = self(batch_g, batch_idx, mask, labels)
        else:
            model_output = self(batch_g, 1, mask, labels)
        loss = self.loss_crit(model_output, 
                    1.0 * F.one_hot(y.view(-1).long(), num_classes=2))
        
        acc = torch.mean(1.0 * (model_output.argmax(axis=1) == y.view(-1)))
        self.log('loss', loss, on_step=True, on_epoch=False, sync_dist=True)
        # Log accuracy: on_step=False (per epoch average), on_epoch=True
        self.log('train_acc', acc, on_step=False, on_epoch=True, sync_dist=True)
        self.log('train loss epoch', loss, on_step=False, on_epoch=True, sync_dist=True)

 
        del model_output
        return loss

    def validation_step(self, batch, batch_idx):
        
        self.validation_step_outputs = [] # useless?
        y = batch[1] #labels
        batch_g = batch[0] # graph input data
        
        # metadata, not used in loss/forward
        mc_energy    = batch[2]  
        pi0_angle    = batch[3]
        pi0_distance = batch[4]

        mask, labels = self.build_attention_mask(batch_g)

        # forward step, why no batch_idx?
        model_output_val = self(batch_g, 1, mask, labels)

        # accumulate predictions (all ranks!)
        self.val_logits.append(model_output_val.detach().cpu())
        self.val_labels.append(y.view(-1).detach().cpu())        

        #if y == 0 (photon) then [1,0], if y == 1 (pi0) then [0,1]
        #! loss calculation for BCELoss (sigmoid needed)
        #! loss = self.loss_crit(
        #!     #torch.sigmoid(model_output), this was used, but I changed it for consistency with training step
        #!     self.m(model_output_val),
        #!     1.0 * F.one_hot(y.view(-1).long(), num_classes=2),
        #! )

        #! loss calculation for BCEWithLogitsLoss (onehot encoding still needed)
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

        # if self.trainer.is_global_zero:
# 
        #     # print(model_output)
        #     # print(labels_true)
        #     
        #     # in this way, wandb is showing only the last loss/accuracy of the epoch
        #     wandb.log({"loss_validation": loss.item()})
# 
        #     acc = torch.mean(1.0 * (model_output_val.argmax(axis=1) == y.view(-1)))
        #     # print(acc)
        #     wandb.log({"accuracy_validation ": acc.item()})



        ## if self.trainer.is_global_zero:
        ##     # wandb.log({"loss": loss.item()})
## 
        ##     # accuracy computation: model_output.argmax(axis=1) picks the index of largest value per row (the predicted class)
        ##     # then, this is compared with true labels, giving a boolean tensor, which is multiplied by one (true = 1, false = 0) and then averaged
        ##     # wandb.log({"accuracy": acc.item()})
        ##     wandb.log({"loss_val": loss.item()})
        ##     wandb.log({"accuracy_val": acc.item()})
        
        acc = torch.mean(1.0 * (model_output_val.argmax(axis=1) == y.view(-1)))
        # Log loss (for epoch-average)
        self.log("loss_val", loss, on_step=False, on_epoch=True, sync_dist=True)
        # Log accuracy (for epoch-average)
        self.log("accuracy_val", acc, on_step=False, on_epoch=True, sync_dist=True)



        del loss
        del model_output_val

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

            # this was in the validation step
            ## wandb.log({
            ##     "confusion_matrix": wandb.plot.confusion_matrix(
            ##         y_true=y_truth,         
            ##         preds=y_preds,
            ##         class_names=["photon", "pi0"]
            ##     )
            ## })
            ##if not self.args.predict:
            ##    # in y_true gamma = 0, pi0 = 1 
            ##    auc = roc_auc_score(y_truth, y_probs_pi0)  
            ##    wandb.log({"auc": auc})
##
            ##    wandb.log({
            ##        "roc_curve": wandb.plot.roc_curve(
            ##            y_true=y_truth_named,
            ##            y_probas=y_probs_gamma_pi0,    
            ##            labels=["photon", "pi0"]
            ##        )
            ##    })

                #fpr, tpr, thresholds = roc_curve(y_truth, y_probs_pi0)
#
#
            #    current_epoch = self.current_epoch 
#
            #    plt.figure(figsize=(12, 10))
            #    plt.plot(
            #        fpr,
            #        tpr,
            #        color='red',
            #        lw=2,
            #        label=f'ROC curve (AUC = {auc:.4f})',
            #    )
#
            #    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#
            #    plt.xlim([0.0, 1.05])
            #    plt.ylim([0.0, 1.05])
            #    plt.xlabel('False Positive Rate (FPR)')
            #    plt.ylabel('True Positive Rate (TPR)')
            #    plt.title(f'ROC Curve - Epoch {current_epoch}')
            #    plt.legend(loc='lower right', fontsize='small')
            #    plt.grid(True)
#
            #    filename = f"predict/roc.png"
            #    plt.savefig(filename)
            #    plt.close()


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