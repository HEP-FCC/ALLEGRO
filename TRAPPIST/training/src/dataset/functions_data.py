import numpy as np
import torch
import dgl
from torch_scatter import scatter_add
from src.dataset.utils_hits import CachedIndexList, get_number_of_daughters, get_ratios, get_number_hits, modify_index_link_for_gamma_e, get_e_reco
import math 

def cdist_wrap_angles(x1, x2=None, p=2, angle_indices=[0,1]):
    """
    Compute pairwise distances with wrap-around for multiple angular coordinates.
    
    Parameters
    ----------
    x1 : torch.Tensor, shape (N, D)
    x2 : torch.Tensor, shape (M, D) or None (defaults to x1)
    p  : float, norm degree (default 2)
    angle_indices : list or tuple of int
        Indices of dimensions that are angles to be wrapped in [-pi, pi].
        If None, no wrapping is done.
    """
    if x2 is None:
        x2 = x1

    diff = x1[:, None, :] - x2[None, :, :]

    if angle_indices is not None:
        for idx in angle_indices:
            diff[..., idx] = (diff[..., idx] + math.pi) % (2 * math.pi) - math.pi

    if p == 2:
        dist = torch.sqrt((diff ** 2).sum(dim=-1))
    else:
        dist = (diff.abs() ** p).sum(dim=-1) ** (1 / p)

    return dist

def calculate_delta_MC(y):
    y1 = y
    y_i = y1
    pseudorapidity = -torch.log(torch.tan(y_i.angle[:,0] / 2))
    phi = y_i.angle[:,1]
    x1 = torch.cat((pseudorapidity.view(-1, 1), phi.view(-1, 1)), dim=1)
    distance_matrix = cdist_wrap_angles(x1, x1, p=2)
    shape_d = distance_matrix.shape[0]
    values, _ = torch.sort(distance_matrix, dim=1)
    if shape_d>1:
        delta_MC = values[:, 1]
    else:
        delta_MC = torch.ones((shape_d,1)).view(-1)
    return delta_MC

def find_mask_no_energy_close_particles(
    hits,
    y,
    predict=False,
    is_Ks=False,
):
    
    list_p = np.unique(hits.hit_particle_link)
    list_remove = []
    delta_MC = calculate_delta_MC(y)
    for index, p in enumerate(list_p):
        mask = hits.hit_particle_link == p
        hit_types = np.unique(hits.hit_type_feature[mask])
        if (
            delta_MC[index]<0.19
        ):
            list_remove.append(p)
    if len(list_remove) > 0:
        mask = torch.tensor(np.full((len(hits.hit_particle_link)), False, dtype=bool))
        for p in list_remove:
            mask1 = hits.hit_particle_link == p
            mask = mask1 + mask

    else:
        mask = np.full((len(hits.hit_particle_link)), False, dtype=bool)

    if len(list_remove) > 0:
        mask_particles = np.full((len(list_p)), False, dtype=bool)
        for p in list_remove:
            mask_particles1 = list_p == p
            mask_particles = mask_particles1 + mask_particles

    else:
        mask_particles = np.full((len(list_p)), False, dtype=bool)
    return mask, mask_particles



def find_mask_no_energy(
    hit_particle_link,
    hit_type_a,
    hit_energies,
    y,
    daughters,
    predict=False,
    is_Ks=False,
):
    """This function remove particles with tracks only and remove particles with low fractions
    # Remove 2212 going to multiple particles without tracks for now
    # remove particles below energy cut
    # remove particles that decayed in the tracker
    # remove particles with two tracks (due to bad tracking)
    # remove particles with daughters for the moment

    Args:
        hit_particle_link (_type_): _description_
        hit_type_a (_type_): _description_
        hit_energies (_type_): _description_
        y (_type_): _description_

    Returns:
        _type_: _description_


    To use this function add the following code to create inputs from table:
                mask_hits, mask_particles = find_mask_no_energy(
                hits.hit_particle_link,
                hits.hit_type_feature,
                hits.e_hits,
                y_data_graph,
                hits.daughters,
                prediction,
                is_Ks=is_Ks,
            )
            # create mapping from links to number of particles in the event
            hits.hit_particle_link = hits.hit_particle_link[~mask_hits]
            hits.find_cluster_id()
            y_data_graph.mask(~mask_particles)
        
    """

    number_of_daughters = get_number_of_daughters(
        hit_type_a, hit_particle_link, daughters
    )
    list_p = np.unique(hit_particle_link)
    list_remove = []
    part_frac = torch.tensor(get_ratios(hit_energies, hit_particle_link, y))
    number_of_hits = get_number_hits(hit_energies, hit_particle_link)
    if predict:
        energy_cut = 0.1
        filt1 = (torch.where(part_frac >= energy_cut)[0] + 1).long().tolist()
    else:
        energy_cut = 0.01
        filt1 = (torch.where(part_frac >= energy_cut)[0] + 1).long().tolist()
    number_of_tracks = scatter_add(1 * (hit_type_a == 1), hit_particle_link.long())[1:]
    if is_Ks == False:
        for index, p in enumerate(list_p):
            mask = hit_particle_link == p
            hit_types = np.unique(hit_type_a[mask])

            if predict:
                if (
                    np.array_equal(hit_types, [0, 1])
                    or int(p) not in filt1
                    or (number_of_hits[index] < 2)
                    or (y.decayed_in_tracker[index] == 1)
                    or number_of_tracks[index] == 2
                    or number_of_daughters[index] > 1
                ):
                    list_remove.append(p)
            else:
                if (
                    np.array_equal(hit_types, [0, 1])
                    or int(p) not in filt1
                    or (number_of_hits[index] < 2)
                    or number_of_tracks[index] == 2
                    or number_of_daughters[index] > 1
                ):
                    list_remove.append(p)
    if len(list_remove) > 0:
        mask = torch.tensor(np.full((len(hit_particle_link)), False, dtype=bool))
        for p in list_remove:
            mask1 = hit_particle_link == p
            mask = mask1 + mask

    else:
        mask = np.full((len(hit_particle_link)), False, dtype=bool)

    if len(list_remove) > 0:
        mask_particles = np.full((len(list_p)), False, dtype=bool)
        for p in list_remove:
            mask_particles1 = list_p == p
            mask_particles = mask_particles1 + mask_particles

    else:
        mask_particles = np.full((len(list_p)), False, dtype=bool)
    return mask, mask_particles






def calculate_distance_to_boundary(g):
    r = 2150
    r_in_endcap = 2307
    mask_endcap = (torch.abs(g.ndata["pos_hits_xyz"][:, 2]) - r_in_endcap) > 0
    mask_barrer = ~mask_endcap
    weight = torch.ones_like(g.ndata["pos_hits_xyz"][:, 0])
    C = g.ndata["pos_hits_xyz"]
    A = torch.tensor([0, 0, 1], dtype=C.dtype, device=C.device)
    P = (
        r
        * 1
        / (torch.norm(torch.cross(A.view(1, -1), C, dim=-1), dim=1)).unsqueeze(1)
        * C
    )
    P1 = torch.abs(r_in_endcap / g.ndata["pos_hits_xyz"][:, 2].unsqueeze(1)) * C
    weight[mask_barrer] = torch.norm(P - C, dim=1)[mask_barrer]
    weight[mask_endcap] = torch.norm(P1[mask_endcap] - C[mask_endcap], dim=1)
    g.ndata["radial_distance"] = weight
    weight_ = torch.exp(-(weight / 1000))
    g.ndata["radial_distance_exp"] = weight_
    return g




def create_noise_label(hit_energies, hit_particle_link, y, cluster_id):
    """
    Creates a mask to identify noise hits in the event, these hits are assigned an index 0 so we don't attempt to reconstruct the particles they belong to.
    Args:
        hit_energies (torch.Tensor): Tensor containing the energies of the hits.
        hit_particle_link (torch.Tensor): Tensor containing the particle link for each hit.
        y (torch.Tensor): Tensor containing the particle information.
        cluster_id (torch.Tensor): Tensor containing the cluster IDs.
    Returns:
        tuple: A tuple containing two boolean masks. The first mask identifies the noise hits, and the second mask identifies the noise particles.
    """

    unique_p_numbers = torch.unique(cluster_id)
    number_of_hits = get_number_hits(hit_energies, cluster_id)
    e_reco = get_e_reco(hit_energies, cluster_id)
    mask_hits = torch.Tensor(number_of_hits) < 6
    mask_p = e_reco<0.10
    mask_all = mask_hits.view(-1) + mask_p.view(-1)
    list_remove = unique_p_numbers[mask_all.view(-1)]

    if len(list_remove) > 0:
        mask = torch.tensor(np.full((len(cluster_id)), False, dtype=bool))
        for p in list_remove:
            mask1 = cluster_id == p
            mask = mask1 + mask
    else:
        mask = torch.tensor(np.full((len(cluster_id)), False, dtype=bool))
    list_p = unique_p_numbers
    if len(list_remove) > 0:
        mask_particles = np.full((len(list_p)), False, dtype=bool)
        for p in list_remove:
            mask_particles1 = list_p == p
            mask_particles = mask_particles1 + mask_particles
    else:
        mask_particles = torch.tensor(np.full((len(list_p)), False, dtype=bool))
    return mask.to(bool), ~mask_particles.to(bool)