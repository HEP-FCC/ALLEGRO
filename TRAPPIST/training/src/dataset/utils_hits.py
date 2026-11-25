import numpy as np
import torch
import dgl
from torch_scatter import scatter_add, scatter_sum
from sklearn.preprocessing import StandardScaler
from torch_scatter import scatter_sum


def create_noise_tracks(index_bad_tracks, hit_particle_link, y, cluster_id):
    unique_p_numbers = torch.unique(cluster_id)
    list_remove = torch.unique(hit_particle_link[index_bad_tracks])

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

def create_noise_label(hits, y):
    cluster_id = hits.cluster_id
    hit_energies = hits.e_hits
    unique_p_numbers = torch.unique(cluster_id)
    number_of_hits = get_number_hits(hit_energies, cluster_id)
    e_reco = get_e_reco(hit_energies, cluster_id)
    mask_hits = torch.Tensor(number_of_hits) < 5
    mask_p = e_reco<0.1
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
    mask_loopers = mask.to(bool)
    mask_particles = ~mask_particles.to(bool)
    hits.hit_particle_link[mask_loopers] = -1
    y.mask(mask_particles)
    hits.find_cluster_id()


def get_ratios(e_hits, part_idx, y):
    """Obtain the percentage of energy of the particle present in the hits

    Args:
        e_hits (_type_): _description_
        part_idx (_type_): _description_
        y (_type_): _description_

    Returns:
        _type_: _description_
    """
    energy_from_showers = scatter_sum(e_hits, part_idx.long(), dim=0)
    # y_energy = y[:, 3]
    y_energy = y.E
    energy_from_showers = energy_from_showers[1:]
    assert len(energy_from_showers) > 0
    return (energy_from_showers.flatten() / y_energy).tolist()

def get_number_hits(e_hits, part_idx):
    number_of_hits = scatter_sum(torch.ones_like(e_hits), part_idx.long(), dim=0)
    return (number_of_hits[1:].flatten()).tolist()

def get_e_reco(e_hits, part_idx):
    number_of_hits = scatter_sum(e_hits, part_idx.long(), dim=0)
    return number_of_hits[1:].flatten()



def get_number_of_daughters(hit_type_feature, hit_particle_link, daughters):
    a = hit_particle_link
    b = daughters
    a_u = torch.unique(a)
    number_of_p = torch.zeros_like(a_u)
    for p, i in enumerate(a_u):
        mask2 = a == i
        number_of_p[p] = torch.sum(torch.unique(b[mask2]) != -1)
    return number_of_p

class CachedIndexList:
    def __init__(self, lst):
        self.lst = lst
        self.cache = {}

    def index(self, value):
        if value in self.cache:
            return self.cache[value]
        else:
            idx = self.lst.index(value)
            self.cache[value] = idx
            return idx
        

def scatter_count(input: torch.Tensor):
    return scatter_add(torch.ones_like(input, dtype=torch.long), input.long())



def modify_index_link_for_gamma_e(
    hit_type_feature, hit_particle_link, daughters, output, number_part, is_Ks=False
):
    """Split all particles that have daughters, mostly for brems and conversions but also for protons and neutrons

    Returns:
        hit_particle_link: new link
        hit_link_modified: bool for modified hits
    """
    hit_link_modified = torch.zeros_like(hit_particle_link).to(hit_particle_link.device)
    mask = hit_type_feature > 1
    a = hit_particle_link[mask]
    b = daughters[mask]
    a_u = torch.unique(a)
    number_of_p = torch.zeros_like(a_u)
    connections_list = []
    for p, i in enumerate(a_u):
        mask2 = a == i
        list_of_daugthers = torch.unique(b[mask2])
        number_of_p[p] = len(list_of_daugthers)
        if (number_of_p[p] > 1) and (torch.sum(list_of_daugthers == i) > 0):
            connections_list.append([i, torch.unique(b[mask2])])
    pid_particles = torch.tensor(output["pf_features"][6, 0:number_part])
    electron_photon_mask = (torch.abs(pid_particles[a_u.long()]) == 11) + (
        pid_particles[a_u.long()] == 22
    )
    electron_photon_mask = (
        electron_photon_mask * number_of_p > 1
    )  # electron_photon_mask *
  
    index_change = a_u  # [electron_photon_mask]
    # else:
    #     index_change = a_u[electron_photon_mask]
    for i in index_change:
        mask_n = mask * (hit_particle_link == i)
        hit_particle_link[mask_n] = daughters[mask_n]
        hit_link_modified[mask_n] = 1
    return hit_particle_link, hit_link_modified, connections_list

def standardize_coordinates(coord_cart_hits):
    if len(coord_cart_hits) == 0:
        return coord_cart_hits, None
    std_scaler = StandardScaler()
    coord_cart_hits = std_scaler.fit_transform(coord_cart_hits)
    return torch.tensor(coord_cart_hits).float(), std_scaler


def create_dif_interactions(i, j, pos, number_p):
    x_interactions = pos
    x_interactions = torch.reshape(x_interactions, [number_p, 1, 2])
    x_interactions = x_interactions.repeat(1, number_p, 1)
    xi = x_interactions[i, j, :]
    xj = x_interactions[j, i, :]
    x_interactions_m = xi - xj
    return x_interactions_m



# def theta_phi_to_pxpypz(pos_theta_phi, pt):
#     px = (pt.view(-1) * torch.cos(pos_theta_phi[:, 0])).view(-1, 1)
#     py = (pt.view(-1) * torch.sin(pos_theta_phi[:, 0])).view(-1, 1)
#     pz = (pt.view(-1) * torch.cos(pos_theta_phi[:, 1])).view(-1, 1)
#     pxpypz = torch.cat(
#         (pos_theta_phi[:, 0].view(-1, 1), pos_theta_phi[:, 1].view(-1, 1), pz), dim=1
#     )
#     return pxpypz