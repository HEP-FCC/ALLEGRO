import torch
import torch.nn as nn
from torch import Tensor
from torch_scatter import scatter_min, scatter_max, scatter_mean, scatter_add
from torch.nn import Parameter
from src.layers.GravNetConv3 import WeirdBatchNorm, knn_per_graph
import numpy as np
from typing import Tuple, Union, List
import dgl
from src.logger.plotting_tools import PlotCoordinates
from src.layers.object_cond import (
    calc_LV_Lbeta,
    get_clustering,
    calc_LV_Lbeta_inference,
)
from typing import Optional, Union
from src.layers.obj_cond_inf import calc_energy_loss
from src.layers.inference_oc import create_and_store_graph_output
from src.models.gravnet_calibration import (
    object_condensation_loss2,
    obtain_batch_numbers,
)
from torch_geometric.nn.conv import MessagePassing
import lightning as L
from src.utils.nn.tools import log_losses_wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import Tensor
from torch.nn import Linear
from torch_scatter import scatter
from torch_geometric.nn.conv import MessagePassing
import torch.nn as nn
import dgl
import dgl.function as fn
import numpy as np
from dgl.nn import EdgeWeightNorm
from torch_geometric.typing import OptTensor, PairTensor, PairOptTensor
from src.layers.graph_transformer_layer_pc1 import GraphTransformerLayer
from src.layers.mlp_readout_layer import MLPReadout
import os
from src.models.gravnet_3_L import GravNetBlock


class GraphT(L.LightningModule):
    def __init__(
        self,
        args,
        dev,
        input_dim: int = 4,
        output_dim: int = 4,
        n_postgn_dense_blocks: int = 3,
        n_gravnet_blocks: int = 4,
        clust_space_norm: str = "twonorm",
        k_gravnet: int = 7,
        activation: str = "elu",
        weird_batchnom=False,
    ):

        super(GraphT, self).__init__()
        self.dev = dev
        self.loss_final = 0
        self.number_b = 0
        self.df_showers = []
        self.df_showers_pandora = []
        self.df_showes_db = []

        self.args = args
        self.validation_step_outputs = []
        in_dim_node = 8  # node_dim (feat is an integer)
        hidden_dim = 64  # before 80
        out_dim = 64
        n_classes = 4
        num_heads = 1
        in_feat_dropout = 0.0
        dropout = 0.0
        n_layers = 4
        self.n_layers = n_layers
        self.layer_norm = False
        self.batch_norm = False
        self.residual = False
        self.dropout = dropout
        self.n_classes = n_classes
        self.lap_pos_enc = False
        self.wl_pos_enc = False
        max_wl_role_index = 100
        self.readout = "sum"
        self.output_dim = n_classes
        self.batchnorm1 = nn.BatchNorm1d(in_dim_node, momentum=0.5)
        self.batchnorm1.running_var = torch.Tensor(
            [10000, 10000, 10000, 1, 1, 1, 1, 1]
        ).to(self.batchnorm1.running_mean.device)
        self.embedding_h = nn.Linear(in_dim_node, hidden_dim)  # node feat is an integer
        self.embedding_h.weight.data.copy_(torch.eye(hidden_dim, in_dim_node))
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        self.n_gravnet_blocks = n_layers
        N_NEIGHBOURS = [16, 128, 16, 128]
        self.layers = nn.ModuleList(
            [
                GraphTransformerLayer(
                    N_NEIGHBOURS[i],
                    64 if i == 0 else (self.d_shape * i + 64),
                    self.d_shape,
                    num_heads,
                    dropout,
                    self.layer_norm,
                    self.batch_norm,
                    self.residual,
                )
                for i in range(n_layers - 1)
            ]
        )
        self.layers.append(
            GraphTransformerLayer(
                N_NEIGHBOURS[-1],
                self.d_shape * 4 + 64,
                self.d_shape,
                num_heads,
                dropout,
                self.layer_norm,
                self.batch_norm,
                self.residual,
            )
        )
        self.d_shape = 32
        self.MLP_layer = MLPReadout(self.d_shape * n_layers + 64, 64)
        self.clustering = nn.Linear(64, 4 - 1, bias=False)
        self.beta = nn.Linear(64, 1)

        # self.gravnet_blocks = nn.ModuleList(
        #     [
        #         GravNetBlock(
        #             64 if i == 0 else (self.d_shape * i + 64),
        #             k=N_NEIGHBOURS[i],
        #             weird_batchnom=weird_batchnom,
        #         )
        #         for i in range(self.n_gravnet_blocks)
        #     ]
        # )

    def forward(self, g, y, step_count, eval=""):
        original_coords = g.ndata["pos_hits_xyz"]
        batch = obtain_batch_numbers(original_coords, g)
        # print("original_coords", original_coords.shape)
        if self.trainer.is_global_zero and step_count % 100 == 0:
            g.ndata["original_coords"] = original_coords
            PlotCoordinates(
                g,
                path="input_coords",
                outdir=self.args.model_prefix,
                features_type="ones",
                predict=self.args.predict,
                epoch=str(self.current_epoch) + eval,
                step_count=step_count,
            )
        ############################## Embeddings #############################################
        h = g.ndata["h"]
        # input embedding
        h = self.batchnorm1(h)
        h = self.embedding_h(h)

        # GraphTransformer Layers
        gu = knn_per_graph(g, original_coords, 7)
        gu.ndata["h"] = h
        allfeat = []
        allfeat.append(h)
        x = h
        # for num_layer, gravnet_block in enumerate(self.gravnet_blocks):
        #! first time dim x is 64
        #! second time is 64+d
        # x, graph, loss_regularizing_neig_block, loss_ll_ = gravnet_block(
        #     gu,
        #     x,
        #     batch,
        #     original_coords,
        #     step_count,
        #     self.args.model_prefix,
        #     num_layer,
        # )
        for layer_n, conv in enumerate(self.layers):
            x = conv(gu, x)
            allfeat.append(h)
            if len(allfeat) > 1:
                x = torch.concatenate(allfeat, dim=1)
        x = torch.cat(allfeat, dim=-1)
        x = self.MLP_layer(x)
        x_cluster_coord = self.clustering(x)
        beta = self.beta(x)
        if self.args.tracks:
            mask = g.ndata["hit_type"] == 1
            beta[mask] = 9
        g.ndata["final_cluster"] = x_cluster_coord
        g.ndata["beta"] = beta.view(-1)
        if self.trainer.is_global_zero and step_count % 100 == 0:
            PlotCoordinates(
                g,
                path="final_clustering",
                outdir=self.args.model_prefix,
                predict=self.args.predict,
                epoch=str(self.current_epoch) + eval,
                step_count=step_count,
            )
        x = torch.cat((x_cluster_coord, beta.view(-1, 1)), dim=1)
        pred_energy_corr = torch.ones_like(beta.view(-1, 1))
        return x, pred_energy_corr, 0

    # def on_after_backward(self):
    #     for name, p in self.named_parameters():
    #         if p.grad is None:
    #             print(name)

    def training_step(self, batch, batch_idx):
        y = batch[1]
        batch_g = batch[0]
        # if self.trainer.is_global_zero and self.current_epoch == 0:
        #     self.stat_dict = obtain_statistics_graph(self.stat_dict, y, batch_g)
        if self.trainer.is_global_zero:
            model_output, e_cor, loss_ll = self(batch_g, y, batch_idx)
        else:
            model_output, e_cor, loss_ll = self(batch_g, y, 1)
            e_cor = torch.ones_like(model_output[:, 0].view(-1, 1))
        (loss, losses, loss_E, loss_E_frac_true,) = object_condensation_loss2(
            batch_g,
            model_output,
            e_cor,
            y,
            clust_loss_only=True,
            add_energy_loss=False,
            calc_e_frac_loss=False,
            q_min=self.args.qmin,
            frac_clustering_loss=self.args.frac_cluster_loss,
            attr_weight=self.args.L_attractive_weight,
            repul_weight=self.args.L_repulsive_weight,
            fill_loss_weight=self.args.fill_loss_weight,
            use_average_cc_pos=self.args.use_average_cc_pos,
            hgcalloss=self.args.hgcalloss,
        )
        loss = loss  # + 0.01 * loss_ll  # + 1 / 20 * loss_E  # add energy loss # loss +
        if self.trainer.is_global_zero:
            log_losses_wandb(True, batch_idx, 0, losses, loss, loss_ll)

        self.loss_final = loss + self.loss_final
        self.number_b = self.number_b + 1
        return loss

    def validation_step(self, batch, batch_idx):
        cluster_features_path = os.path.join(self.args.model_prefix, "cluster_features")
        show_df_eval_path = os.path.join(
            self.args.model_prefix, "showers_df_evaluation"
        )
        if not os.path.exists(show_df_eval_path):
            os.makedirs(show_df_eval_path)
        if not os.path.exists(cluster_features_path):
            os.makedirs(cluster_features_path)
        self.validation_step_outputs = []
        y = batch[1]
        batch_g = batch[0]
        if self.args.correction:
            (
                model_output,
                e_cor1,
                true_e,
                sum_e,
                new_graphs,
                batch_id,
                graph_level_features,
            ) = self(batch_g, y, 1)
            loss_ll = 0
            e_cor = torch.ones_like(model_output[:, 0].view(-1, 1))
        else:
            model_output, e_cor1, loss_ll = self(batch_g, y, 1)
            loss_ll = 0
            e_cor = torch.ones_like(model_output[:, 0].view(-1, 1))
        preds = model_output.squeeze()
        (loss, losses, loss_E, loss_E_frac_true,) = object_condensation_loss2(
            batch_g,
            model_output,
            e_cor,
            y,
            clust_loss_only=True,
            add_energy_loss=False,
            calc_e_frac_loss=False,
            q_min=self.args.qmin,
            frac_clustering_loss=self.args.frac_cluster_loss,
            attr_weight=self.args.L_attractive_weight,
            repul_weight=self.args.L_repulsive_weight,
            fill_loss_weight=self.args.fill_loss_weight,
            use_average_cc_pos=self.args.use_average_cc_pos,
            hgcalloss=self.args.hgcalloss,
        )
        loss_ec = 0

        print("starting validation step", batch_idx, loss)
        if self.trainer.is_global_zero:
            log_losses_wandb(
                True, batch_idx, 0, losses, loss, loss_ll, loss_ec, val=True
            )
        self.validation_step_outputs.append([model_output, e_cor, batch_g, y])
        if self.args.predict:
            model_output1 = torch.cat((model_output, e_cor.view(-1, 1)), dim=1)
            e_corr = None
            (df_batch_pandora, df_batch1, df_batch) = create_and_store_graph_output(
                batch_g,
                model_output1,
                y,
                0,
                batch_idx,
                0,
                path_save=show_df_eval_path,
                store=True,
                predict=True,
                e_corr=e_corr,
                tracks=self.args.tracks,
            )
            self.df_showers.append(df_batch)
            self.df_showers_pandora.append(df_batch_pandora)
            self.df_showes_db.append(df_batch1)
        print("end of validation step ")

    def on_train_epoch_end(self):

        self.log("train_loss_epoch", self.loss_final / self.number_b)

    def on_train_epoch_start(self):
        self.make_mom_zero()

    def on_validation_epoch_start(self):
        self.make_mom_zero()
        self.df_showers = []
        self.df_showers_pandora = []
        self.df_showes_db = []

    def make_mom_zero(self):
        if self.current_epoch > 1 or self.args.predict:
            self.ScaledGooeyBatchNorm2_1.momentum = 0

    def on_validation_epoch_end(self):
        if self.trainer.is_global_zero:
            if self.args.predict:
                from src.layers.inference_oc import store_at_batch_end
                import pandas as pd

                self.df_showers = pd.concat(self.df_showers)
                self.df_showers_pandora = pd.concat(self.df_showers_pandora)
                self.df_showes_db = pd.concat(self.df_showes_db)
                store_at_batch_end(
                    path_save=os.path.join(
                        self.args.model_prefix, "showers_df_evaluation"
                    ),
                    df_batch=self.df_showers,
                    df_batch_pandora=self.df_showers_pandora,
                    df_batch1=self.df_showes_db,
                    step=0,
                    predict=True,
                    store=True,
                )
            # else:
            #     model_output = self.validation_step_outputs[0][0]
            #     e_corr = self.validation_step_outputs[0][1]
            #     batch_g = self.validation_step_outputs[0][2]
            #     y = self.validation_step_outputs[0][3]
            #     model_output1 = torch.cat((model_output, e_corr.view(-1, 1)), dim=1)
            #     create_and_store_graph_output(
            #         batch_g,
            #         model_output1,
            #         y,
            #         0,
            #         0,
            #         0,
            #         path_save=os.path.join(
            #             self.args.model_prefix, "showers_df_evaluation"
            #         ),
            #         store=True,
            #         predict=False,
            #         tracks=self.args.tracks,
            #     )
        self.validation_step_outputs = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=1e-3
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
