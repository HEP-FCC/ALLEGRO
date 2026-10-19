
from lightning.pytorch.callbacks import BaseFinetuning
import torch
import torch.nn as nn
import dgl
from src.layers.inference_oc import (
    get_clustering,
)
from src.layers.inference_oc import hfdb_obtain_labels, clustering_obtain_labels
from src.layers.inference_oc import match_showers
## import torch_cmspepr
from src.layers.inference_oc import remove_bad_tracks_from_cluster
class FreezeClustering(BaseFinetuning):
    def __init__(
        self,
    ):
        super().__init__()
        # self._unfreeze_at_epoch = unfreeze_at_epoch

    def freeze_before_training(self, pl_module):
        print("freezing the following module:", pl_module)
        # freeze any module you want
        # Here, we are freezing `feature_extractor`

        self.freeze(pl_module.ScaledGooeyBatchNorm2_1)
        # self.freeze(pl_module.Dense_1)
        self.freeze(pl_module.gatr)
        # self.freeze(pl_module.postgn_dense)
        # self.freeze(pl_module.ScaledGooeyBatchNorm2_2)
        self.freeze(pl_module.clustering)
        self.freeze(pl_module.beta)

        print("CLUSTERING HAS BEEN FROOOZEN")

    def finetune_function(self, pl_module, current_epoch, optimizer):
        print("Not finetunning")
        # # When `current_epoch` is 10, feature_extractor will start training.
        # if current_epoch == self._unfreeze_at_epoch:
        #     self.unfreeze_and_add_param_group(
        #         modules=pl_module.feature_extractor,
        #         optimizer=optimizer,
        #         train_bn=True,
        #     )


def obtain_batch_numbers(x, g):
    dev = x.device
    graphs_eval = dgl.unbatch(g)
    number_graphs = len(graphs_eval)
    batch_numbers = []
    for index in range(0, number_graphs):
        gj = graphs_eval[index]
        num_nodes = gj.number_of_nodes()
        batch_numbers.append(index * torch.ones(num_nodes).to(dev))
        # num_nodes = gj.number_of_nodes()

    batch = torch.cat(batch_numbers, dim=0)
    return batch



## def obtain_clustering_for_matched_showers(
##     batch_g, model_output, y_all, local_rank, use_gt_clusters=False, add_fakes=True
## ):
##     if use_gt_clusters:
##         print("!!! Using GT clusters for Energy Correction !!!!")
##     graphs_showers_matched = []
##     graphs_showers_fakes = []
##     true_energy_showers = []
##     reco_energy_showers = []
##     reco_energy_showers_fakes = []
##     energy_true_daughters = []
##     y_pids_matched = []
##     y_coords_matched = []
##     batch_g.ndata["coords"] = model_output[:, 0:3]
##     batch_g.ndata["beta"] = model_output[:, 3]
##     graphs = dgl.unbatch(batch_g)
##     batch_id = y_all.batch_number
##     for i in range(0, len(graphs)):
##         mask = batch_id == i
##         dic = {}
##         dic["graph"] = graphs[i]
##         y = y_all.copy()
##         # if "unique_list_particles" in y.__dict__:
##         #    del y.unique_list_particles
##         y.mask(mask.flatten())
##         dic["part_true"] = y
##         betas = torch.sigmoid(dic["graph"].ndata["beta"])
##         X = dic["graph"].ndata["coords"]
##         clustering_mode = "dbscan"
##         if clustering_mode == "clustering_normal":
##             labels = clustering_obtain_labels( X,torch.sigmoid(betas.view(-1)), betas.device,  tbeta=0.2, td=0.05)
##         elif clustering_mode == "dbscan":
##             if use_gt_clusters:
##                 labels = dic["graph"].ndata["particle_number"].type(torch.int64)
##             else:
##                 labels = hfdb_obtain_labels(X, model_output.device)
##                 labels = remove_bad_tracks_from_cluster(dic["graph"], labels)
##                 # labels = clustering_obtain_labels( X,betas.view(-1), betas.device,  tbeta=0.7, td=0.3)
##                 #if labels.min() == 0 and labels.sum() == 0:
##                 #    labels += 1  # Quick hack
##         particle_ids = torch.unique(dic["graph"].ndata["particle_number"])
##         shower_p_unique = torch.unique(labels)
##         shower_p_unique, row_ind, col_ind, i_m_w, _ = match_showers(
##             labels, dic, particle_ids, model_output, local_rank, i, None
##         )
##         row_ind = torch.Tensor(row_ind).to(model_output.device).long()
##         col_ind = torch.Tensor(col_ind).to(model_output.device).long()
##         if torch.sum(particle_ids == 0) > 0:
##             row_ind_ = row_ind - 1
##         else:
##             # if there is no zero then index 0 corresponds to particle 1.
##             row_ind_ = row_ind
##         index_matches = col_ind + 1
##         index_matches = index_matches.to(model_output.device).long()
##         """
##                     ### Plot shapes of some showers, to debug what's wrong with the energies
##                     debug_showers = False
##                     if debug_showers:
##                         energy_true_part = dic["part_true"][:, 3].detach().cpu()
##                         from torch_scatter import scatter_sum
##                         energy_sum_hits = scatter_sum(dic["graph"].ndata["e_hits"], dic["graph"].ndata["particle_number"].type(torch.int64), dim=0).flatten().detach().cpu()
##                         energy_noise = str(round(energy_sum_hits[0].item(), 2))
##                         n_hits_noise = torch.sum(dic["graph"].ndata["particle_number"] == 0).detach().cpu().item()
##                         #frac_energy_sum = energy_sum_hits / energy_true_part[1:]
##                         import matplotlib.pyplot as plt
##                         n_particles = len(particle_ids)
##                         fig = plt.figure(figsize=(18, 4 * n_particles))
##                         for j in range(n_particles):
##                             mask = labels == j
##                             # make ax projection 3D
##                             #ax.scatter(X[mask, 0].detach().cpu(), X[mask, 1].detach().cpu(), c=dic["graph"].ndata["hit_type"][mask].detach().cpu())
##                             ax = fig.add_subplot(n_particles, 1, j+1, projection='3d')
##                             ax.scatter(X[mask, 0].detach().cpu(), X[mask, 1].detach().cpu(), X[mask, 2].detach().cpu(), c=dic["graph"].ndata["hit_type"][mask].detach().cpu())
##                             pnum = (particle_ids[j]-1).type(torch.int64).detach().cpu()
##                             part_xyz = dic["part_true"][pnum, [0,1,2]].detach().cpu()
##                             ax.scatter(part_xyz[0], part_xyz[1], part_xyz[2], c='r', s=100)
##                             ax.set_title(f"gr. {i}, E c.f. = {str(round(energy_true_part[pnum].item() / energy_sum_hits[1:][pnum].item() - 1, 2))}, Etrue = {round(energy_true_part[pnum].item(), 2)}, Esum_hits = {round(energy_sum_hits[1:][pnum].item(), 2)}, Nnoisehits = {n_hits_noise}, Enoise = {energy_noise}, eta={part_eta},phi={part_phi}")
##                         # log to wandb
##                         wandb.log({"showers": [wandb.Image(fig, caption="showers")]})
##         """
##         for j, unique_showers_label in enumerate(index_matches):
##             if torch.sum(unique_showers_label == index_matches) == 1:
##                 index_in_matched = torch.argmax(
##                     (unique_showers_label == index_matches) * 1
##                 )
##                 mask = labels == unique_showers_label
##                 # non_graph = torch.sum(mask)
##                 sls_graph = graphs[i].ndata["pos_hits_xyz"][mask][:, 0:3]
##                 k = 7
##                 edge_index = torch_cmspepr.knn_graph(sls_graph, k=k)
##                 g = dgl.graph(
##                     (edge_index[0], edge_index[1]), num_nodes=sls_graph.shape[0]
##                 )
##                 g = dgl.remove_self_loop(g)
##                 # g = dgl.DGLGraph().to(graphs[i].device)
##                 # g.add_nodes(non_graph.detach().cpu())
##                 g.ndata["h"] = torch.cat(
##                     (
##                         graphs[i].ndata["h"][mask],
##                         graphs[i].ndata["beta"][mask].view(-1, 1),
##                     ),
##                     dim=1,
##                 )
##                 if "pos_pxpypz" in graphs[i].ndata:
##                     g.ndata["pos_pxpypz"] = graphs[i].ndata["pos_pxpypz"][mask]
##                 if "pos_pxpypz_at_vertex" in graphs[i].ndata:
##                     g.ndata["pos_pxpypz_at_vertex"] = graphs[i].ndata[
##                         "pos_pxpypz_at_vertex"
##                     ][mask]
##                 g.ndata["chi_squared_tracks"] = graphs[i].ndata["chi_squared_tracks"][mask]
##                 energy_t = dic["part_true"].E.to(model_output.device)
##                 energy_t_corr_daughters = dic["part_true"].E_corrected.to(
##                     model_output.device
##                 )
##                 true_energy_shower = energy_t[row_ind_[j]]
##                 y_pids_matched.append(y.pid[row_ind_[j]].item())
##                 y_coords_matched.append(y.coord[row_ind_[j]].detach().cpu().numpy())
##                 energy_true_daughters.append(energy_t_corr_daughters[row_ind_[j]])
##                 reco_energy_shower = torch.sum(graphs[i].ndata["e_hits"][mask])
##                 graphs_showers_matched.append(g)
##                 true_energy_showers.append(true_energy_shower.view(-1))
##                 reco_energy_showers.append(reco_energy_shower.view(-1))
##         pred_showers = shower_p_unique
##         pred_showers[index_matches] = -1
##         pred_showers[
##             0
##         ] = (
##             -1
##         )
##         mask_fakes = pred_showers != -1
##         fakes_idx = torch.where(mask_fakes)[0]
##         if add_fakes:
##             for j in fakes_idx:
##                 mask = labels == j
##                 sls_graph = graphs[i].ndata["pos_hits_xyz"][mask][:, 0:3]
##                 k = 7
##                 edge_index = torch_cmspepr.knn_graph(sls_graph, k=k)
##                 g = dgl.graph(
##                     (edge_index[0], edge_index[1]), num_nodes=sls_graph.shape[0]
##                 )
##                 g = dgl.remove_self_loop(g)
##                 g.ndata["h"] = torch.cat(
##                     (
##                         graphs[i].ndata["h"][mask],
##                         graphs[i].ndata["beta"][mask].view(-1, 1),
##                     ),
##                     dim=1,
##                 )
##                 if "pos_pxpypz" in graphs[i].ndata:
##                     g.ndata["pos_pxpypz"] = graphs[i].ndata["pos_pxpypz"][mask]
##                 if "pos_pxpypz_at_vertex" in graphs[i].ndata:
##                     g.ndata["pos_pxpypz_at_vertex"] = graphs[i].ndata[
##                         "pos_pxpypz_at_vertex"
##                     ][mask]
##                 g.ndata["chi_squared_tracks"] = graphs[i].ndata["chi_squared_tracks"][mask]
##                 graphs_showers_fakes.append(g)
##                 reco_energy_shower = torch.sum(graphs[i].ndata["e_hits"][mask])
##                 reco_energy_showers_fakes.append(reco_energy_shower.view(-1))
##     graphs_showers_matched = dgl.batch(graphs_showers_matched + graphs_showers_fakes)
##     true_energy_showers = torch.cat(true_energy_showers, dim=0)
##     reco_energy_showers = torch.cat(reco_energy_showers + reco_energy_showers_fakes, dim=0)
##     e_true_corr_daughters = torch.cat(energy_true_daughters, dim=0)
##     number_of_fakes = len(reco_energy_showers_fakes)
##     return (
##         graphs_showers_matched,
##         true_energy_showers,
##         reco_energy_showers,
##         y_pids_matched,
##         e_true_corr_daughters,
##         y_coords_matched,
##         number_of_fakes,
##         fakes_idx
##     )
## 