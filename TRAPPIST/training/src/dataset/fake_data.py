import torch
import dgl


def create_fake_graph():
    num_hits_per_particle = 15
    num_part = 3
    n_noise = 0
    # create a synthetic graph - random hits uniformly between -4 and 4, distribution of hits is gaussian
    y_coords = torch.zeros((num_part, 3)).float()
    # uniformly picked x,y,z coords saved in y_coords
    y_coords[:, 0] = torch.rand((num_part)).float() * 8 - 4
    y_coords[:, 1] = torch.rand((num_part)).float() * 8 - 4
    y_coords[:, 2] = torch.rand((num_part)).float() * 8 - 4
    graph_coordinates = torch.zeros((num_part * num_hits_per_particle, 3)).float()
    hit_type_one_hot = torch.zeros((num_part * num_hits_per_particle, 4)).float()
    e_hits = torch.zeros((num_part * num_hits_per_particle, 1)).float() + 1.0
    p_hits = (
        torch.zeros((num_part * num_hits_per_particle, 1)).float() + 1.0
    )  # to avoid nans
    for i in range(num_part):
        index = i * num_hits_per_particle
        graph_coordinates[index : index + num_hits_per_particle] = (
            torch.randn((num_hits_per_particle, 3)).float() * 0.1 + y_coords[i]
        )
        hit_type_one_hot[index : index + num_hits_per_particle, 3] = 1.0
    g = dgl.knn_graph(graph_coordinates, 7, exclude_self=True)

    i, j = g.edges()
    edge_attr = torch.norm(
        graph_coordinates[i] - graph_coordinates[j], p=2, dim=1
    ).view(-1, 1)
    hit_features_graph = torch.cat(
        (graph_coordinates, hit_type_one_hot, e_hits, p_hits), dim=1
    )
    hit_particle_link = torch.zeros((num_part * num_hits_per_particle, 1)).float()
    for i in range(num_part):
        index = i * num_hits_per_particle
        hit_particle_link[index : index + num_hits_per_particle] = (
            i + 1
        )  # 0 is for noise
    if n_noise > 0:
        noise = torch.zeros((p_hits.shape[0], n_noise)).float()
        noise.normal_(mean=0, std=1)
        hit_features_graph = torch.cat(
            (graph_coordinates, hit_type_one_hot, e_hits, p_hits, noise), dim=1
        )
    g.ndata["h"] = hit_features_graph
    g.ndata["pos_hits"] = graph_coordinates
    g.ndata["pos_hits_xyz"] = graph_coordinates
    g.ndata["pos_hits_norm"] = graph_coordinates
    g.ndata["hit_type"] = hit_type_one_hot
    g.ndata["p_hits"] = p_hits
    g.ndata["e_hits"] = e_hits
    g.ndata["particle_number"] = hit_particle_link
    g.ndata["particle_number_nomap"] = hit_particle_link
    g.edata["h"] = edge_attr

    y_data_graph = torch.cat(
        (
            y_coords,
            torch.zeros((num_part, 4)).float(),
        ),
        dim=1,
    )
    return [g, y_data_graph], False
