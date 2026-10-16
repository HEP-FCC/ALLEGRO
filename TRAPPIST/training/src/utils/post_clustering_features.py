import torch
from torch_scatter import scatter_sum, scatter_std

def calculate_phi(x, y, z=None):
    return torch.arctan2(y, x)

def calculate_eta(x, y, z):
    theta = torch.arctan2(torch.sqrt(x ** 2 + y ** 2), z)
    return -torch.log(torch.tan(theta / 2))

def get_post_clustering_features(graphs_new, sum_e, is_muons=False, add_hit_chis=False):
    '''
    Obtain graph-level qualitative features that can then be used to regress the energy corr. factor.
    :param graph_batch: Output from the previous step - clustered, matched showers
    :return:
    '''
    batch_num_nodes = graphs_new.batch_num_nodes()  # Num. of hits in each graph
    batch_idx = []
    batch_bounds = []
    for i, n in enumerate(batch_num_nodes):
        batch_idx.extend([i] * n)
        batch_bounds.append(n)
    batch_idx = torch.tensor(batch_idx).to(graphs_new.device)
    e_hits = graphs_new.ndata["h"][:, 8]
    if is_muons:
        muon_hits = graphs_new.ndata["h"][:, 7]
        filter_muon = torch.where(muon_hits)[0]
        per_graph_e_hits_muon = scatter_sum(e_hits[filter_muon], batch_idx[filter_muon], dim_size=batch_idx.max() + 1)
        per_graph_n_hits_muon = scatter_sum((e_hits[filter_muon] > 0).type(torch.int), batch_idx[filter_muon], dim_size=batch_idx.max() + 1)
    ecal_hits = graphs_new.ndata["h"][:, 5]
    filter_ecal = torch.where(ecal_hits)[0]
    hcal_hits = graphs_new.ndata["h"][:, 6]
    filter_hcal = torch.where(hcal_hits)[0]
    per_graph_e_hits_ecal = scatter_sum(e_hits[filter_ecal], batch_idx[filter_ecal], dim_size=batch_idx.max() + 1)
    per_graph_e_hits_ecal_dispersion = torch.zeros_like(per_graph_e_hits_ecal)
    per_graph_e_hits_ecal_dispersion = per_graph_e_hits_ecal_dispersion / batch_num_nodes
    # similar as above but with scatter_std
    per_graph_e_hits_ecal_dispersion = scatter_std(e_hits[filter_ecal], batch_idx[filter_ecal], dim_size=batch_idx.max() + 1) ** 2
    per_graph_e_hits_hcal = scatter_sum(e_hits[filter_hcal], batch_idx[filter_hcal], dim_size=batch_idx.max() + 1)
    # similar as above but with scatter_std -- !!!!! TODO: Retrain the base EC models using this definition !!!!!
    per_graph_e_hits_hcal_dispersion = scatter_std(e_hits[filter_hcal], batch_idx[filter_hcal], dim_size=batch_idx.max() + 1) ** 2
    # track_nodes =
    track_p = scatter_sum(graphs_new.ndata["h"][:, 9], batch_idx)
    chis_tracks = scatter_sum(graphs_new.ndata["chi_squared_tracks"], batch_idx)
    num_tracks = scatter_sum((graphs_new.ndata["h"][:, 9] > 0).type(torch.int), batch_idx)
    track_p = track_p / num_tracks
    track_p[num_tracks == 0] = 0.
    chis_tracks = chis_tracks / num_tracks
    num_hits = graphs_new.batch_num_nodes()
    # print shapes of the below things
    if add_hit_chis:
        if is_muons:
            return torch.nan_to_num(
                torch.stack([per_graph_e_hits_ecal / sum_e,
                                per_graph_e_hits_hcal / sum_e,
                                num_hits, track_p,
                                per_graph_e_hits_ecal_dispersion,
                                per_graph_e_hits_hcal_dispersion,
                                sum_e, num_tracks, torch.clamp(chis_tracks, -5, 5),
                                per_graph_e_hits_muon,
                                per_graph_n_hits_muon
                             ]).T
            )

        return torch.nan_to_num(
            torch.stack([per_graph_e_hits_ecal / sum_e,
                            per_graph_e_hits_hcal / sum_e,
                            num_hits, track_p,
                            per_graph_e_hits_ecal_dispersion,
                            per_graph_e_hits_hcal_dispersion,
                            sum_e, num_tracks, torch.clamp(chis_tracks, -5, 5)]).T
        )
    else:
        return torch.nan_to_num(
            torch.stack([per_graph_e_hits_ecal / sum_e,
                            per_graph_e_hits_hcal / sum_e,
                            num_hits, track_p,
                            per_graph_e_hits_ecal_dispersion,
                            per_graph_e_hits_hcal_dispersion,
                            sum_e, num_tracks]).T
        )  # nan_to_num due to division by zero when there is zero tracks


def get_extra_features(graphs_new, betas):
    '''
    Obtain extra graph-level features for debugging of the fakes
    '''
    batch_num_nodes = graphs_new.batch_num_nodes()  # Num. of hits in each graph
    batch_idx = []
    batch_bounds = []
    topk_highest_betas = []
    for i, n in enumerate(batch_num_nodes):
        batch_idx.extend([i] * n)
        batch_bounds.append(n)
    batch_idx = torch.tensor(batch_idx).to(graphs_new.device)
    n_highest_betas = 1
    for i in range(len(batch_num_nodes)):
        betas_i = betas[batch_idx == i]
        topk_betas = torch.topk(betas_i, n_highest_betas)
        if len(topk_betas.values) < n_highest_betas:
            topk_betas = torch.cat([topk_betas.values, torch.zeros(n_highest_betas - len(topk_betas.values))])
        topk_highest_betas.append(topk_betas.values)
    topk_highest_betas = torch.stack(topk_highest_betas)
    # Concat with batch_num_nodes
    features = torch.cat([batch_num_nodes.view(-1, 1), topk_highest_betas], dim=1)
    return features
