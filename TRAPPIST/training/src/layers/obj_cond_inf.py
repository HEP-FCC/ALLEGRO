from typing import Tuple, Union
import numpy as np
import torch
from torch_scatter import scatter_max, scatter_add, scatter_mean
from src.layers.object_cond import assert_no_nans, scatter_count, batch_cluster_indices
import dgl


def calc_energy_loss(
    batch, cluster_space_coords, beta, beta_stabilizing="soft_q_scaling", qmin=0.1, radius=0.7,
    e_frac_loss_return_particles=False, y=None, select_centers_by_particle=True
):
    # select_centers_by_particle: if True, we pretend we know which hits belong to which particle...
    list_graphs = dgl.unbatch(batch)
    node_counter = 0
    if beta_stabilizing == "paper":
        q = beta.arctanh() ** 2 + qmin
    elif beta_stabilizing == "clip":
        beta = beta.clip(0.0, 1 - 1e-4)
        q = beta.arctanh() ** 2 + qmin
    elif beta_stabilizing == "soft_q_scaling":
        q = (beta.clip(0.0, 1 - 1e-4) / 1.002).arctanh() ** 2 + qmin
    else:
        raise ValueError(f"beta_stablizing mode {beta_stabilizing} is not known")
    loss_E_frac = []
    loss_E_frac_true = []
    particle_ids_all = []
    reco_count = {}  # per-PID count
    non_reco_count = {}
    total_count = {}
    for g in list_graphs:
        particle_id = g.ndata["particle_number"]
        number_of_objects = len(particle_id.unique())
        print("No. of objects", number_of_objects)
        non = g.number_of_nodes()
        q_g = q[node_counter : non + node_counter]
        betas = beta[node_counter : non + node_counter]
        sorted, indices = torch.sort(betas.view(-1), descending=False)
        selected_centers = indices[0:number_of_objects]
        _, selected_centers_particles = scatter_max(
            betas.flatten().cpu(), particle_id.cpu().long() - 1
        )
        assert selected_centers.shape[0] == number_of_objects
        if select_centers_by_particle:
            selected_centers = selected_centers_particles.to(g.device)
        all_particles = set((particle_id.unique()-1).long().tolist())
        reco_particles = set((particle_id[selected_centers]-1).long().tolist())
        non_reco_particles = all_particles - reco_particles
        part_pids = y[:, 6].long()
        for particle in all_particles:
            curr_pid = part_pids[particle].item()
            if curr_pid in total_count:
                total_count[curr_pid] += 1
            else:
                total_count[curr_pid] = 1
            if particle in reco_particles:
                if curr_pid in reco_count:
                    reco_count[curr_pid] += 1
                else:
                    reco_count[curr_pid] = 1
            else:
                if curr_pid in non_reco_count:
                    non_reco_count[curr_pid] += 1
                else:
                    non_reco_count[curr_pid] = 1
        X = cluster_space_coords[node_counter : non + node_counter]

        if radius == "dynamic":
            pick_ = torch.argsort(
                torch.cdist(X[selected_centers], X[selected_centers], p=2),
                dim=1)[:, 1]
            current_radius = torch.cdist(torch.Tensor(X[selected_centers]), torch.Tensor(X[selected_centers]), p=2).gather(1, pick_.view(-1, 1))
            current_radius = current_radius / 2
            current_radius = max(0.1, current_radius.flatten().min())
            print("Current radius", current_radius)
        else:
            print("Radius", radius)
            current_radius = radius
        clusterings = get_clustering(selected_centers, X, betas, td=current_radius)
        clusterings = clusterings.to(g.device)
        node_counter += non
        counter = 0
        frac_energy = []
        frac_energy_true = []
        particle_ids = []
        for alpha in selected_centers:
            id_particle = particle_id[alpha]
            true_mask_particle = particle_id == id_particle
            true_energy = torch.sum(g.ndata["e_hits"][true_mask_particle])
            mask_clustering_particle = clusterings == counter
            clustered_energy = torch.sum(g.ndata["e_hits"][mask_clustering_particle])
            clustered_energy_true = torch.sum(
                g.ndata["e_hits"][
                    mask_clustering_particle * true_mask_particle.flatten()
                ]
            )  # only consider how much has been correctly assigned
            counter += 1
            frac_energy.append(clustered_energy / (true_energy + 1e-7))
            frac_energy_true.append(clustered_energy_true / (true_energy + 1e-7))
            particle_ids.append(id_particle.cpu().long().item())
        frac_energy = torch.stack(frac_energy, dim=0)
        if not e_frac_loss_return_particles:
            frac_energy = torch.mean(frac_energy)
        frac_energy_true = torch.stack(frac_energy_true, dim=0)
        if not e_frac_loss_return_particles:
            frac_energy_true = torch.mean(frac_energy_true)
        loss_E_frac.append(frac_energy)
        loss_E_frac_true.append(frac_energy_true)
        particle_ids_all.append(particle_ids)
    if e_frac_loss_return_particles:
        return loss_E_frac, [loss_E_frac_true, particle_ids_all, reco_count, non_reco_count, total_count]
    loss_E_frac = torch.mean(torch.stack(loss_E_frac, dim=0))
    loss_E_frac_true = torch.mean(torch.stack(loss_E_frac_true, dim=0))
    return loss_E_frac, loss_E_frac_true


def get_clustering(index_alpha_i, X, betas, td=0.7):
    n_points = betas.size(0)
    unassigned = torch.arange(n_points).to(betas.device)
    clustering = -1 * torch.ones(n_points, dtype=torch.long)
    counter = 0
    for index_condpoint in index_alpha_i:
        d = torch.norm(X[unassigned] - X[index_condpoint], dim=-1)
        assigned_to_this_condpoint = unassigned[d < td]
        clustering[assigned_to_this_condpoint] = counter
        unassigned = unassigned[~(d < td)]
        counter = counter + 1
    counter = 0
    for index_condpoint in index_alpha_i:
        clustering[index_condpoint] = counter
        counter = counter + 1

    return clustering
