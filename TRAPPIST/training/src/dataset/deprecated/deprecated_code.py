        #     g = dgl.knn_graph(
        #         graph_coordinates,
        #         config.graph_config.get("k", 7),
        #         exclude_self=True,
        #     )
        #     if graph_coordinates.shape[0] < 10:
        #         print(graph_coordinates.shape)

        # i, j = g.edges()
        # edge_attr = torch.norm(
        #     graph_coordinates[i] - graph_coordinates[j], p=2, dim=1
        # ).view(-1, 1)
        # if n_noise > 0:
        #     noise = torch.zeros((p_hits.shape[0], n_noise)).float()
        #     noise.normal_(mean=0, std=1)
        #     hit_features_graph = torch.cat(
        #         (graph_coordinates, hit_type_one_hot, e_hits, p_hits, noise), dim=1
        #     )
        # else:


 # print("n hits:", number_hits, "number_part", number_part)
    # this builds fully connected graph
    # TODO build graph using the hit links (hit_particle_link) which assigns to each node the particle it belongs to
    # i, j = torch.tril_indices(number_hits, number_hits)
    # g = dgl.graph((i, j))
    # g = dgl.to_simple(g)
    # g = dgl.to_bidirected(g)


 # if config.graph_config.get("fully_connected", False):
        #     n_nodes = graph_coordinates.shape[0]
        #     if n_nodes > 1:
        #         i, j = torch.tril_indices(n_nodes, n_nodes, offset=-1)
        #         g = dgl.graph((i, j))  # create fully connected graph
        #         g = dgl.to_simple(g)  # remove repeated edges
        #         g = dgl.to_bidirected(g)
        #     else:
        #         g = dgl.knn_graph(graph_coordinates, 0, exclude_self=True)
        # else:


def create_graph_synthetic(config, n_noise=0, npart_min=3, npart_max=5):
    num_hits_per_particle_min, num_hits_per_particle_max = 5, 60
    num_part = np.random.randint(npart_min, npart_max)
    num_hits_per_particle = np.random.randint(
        num_hits_per_particle_min, num_hits_per_particle_max, size=(num_part,)
    )
    # create a synthetic graph - random hits uniformly between -4 and 4, distribution of hits is gaussian
    y_coords = torch.zeros((num_part, 3)).float()
    # uniformly picked x,y,z coords saved in y_coords
    y_coords[:, 0] = torch.rand((num_part)).float() * 8 - 4
    y_coords[:, 1] = torch.rand((num_part)).float() * 8 - 4
    y_coords[:, 2] = torch.rand((num_part)).float() * 8 - 4
    nh = np.sum(num_hits_per_particle)
    graph_coordinates = torch.zeros((nh, 3)).float()
    hit_type_one_hot = torch.zeros((nh, 4)).float()
    e_hits = torch.zeros((nh, 1)).float() + 1.0
    p_hits = torch.zeros((nh, 1)).float() + 1.0  # to avoid nans
    for i in range(num_part):
        index = np.sum(num_hits_per_particle[:i])
        graph_coordinates[index : index + num_hits_per_particle[i]] = (
            torch.randn((num_hits_per_particle[i], 3)).float()
            * torch.tensor([0.12, 0.5, 0.4])
            + y_coords[i]
        )
        hit_type_one_hot[index : index + num_hits_per_particle[i], 3] = 1.0
    g = dgl.knn_graph(
        graph_coordinates, config.graph_config.get("k", 7), exclude_self=True
    )
    i, j = g.edges()
    edge_attr = torch.norm(
        graph_coordinates[i] - graph_coordinates[j], p=2, dim=1
    ).view(-1, 1)
    hit_features_graph = torch.cat(
        (graph_coordinates, hit_type_one_hot, e_hits, p_hits), dim=1
    )
    hit_particle_link = torch.zeros((nh, 1)).float()
    for i in range(num_part):
        index = np.sum(num_hits_per_particle[:i])
        hit_particle_link[index : index + num_hits_per_particle[i]] = (
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


def to_hetero(g, all_hit_types=[2, 3]):
    # Convert the dgl graph object to a heterograph
    # We probably won't be using this
    hit_types = g.ndata["hit_type"]
    hit_types = torch.argmax(hit_types, dim=1)
    ht_idx = [all_hit_types.index(i) for i in hit_types]
    edges = g.edges()
    graph_data = {}
    for i in all_hit_types:
        for j in all_hit_types:
            edge_mask = hit_types[edges[0]] == i
            edge_mask = edge_mask & (hit_types[edges[1]] == j)
            graph_data[(str(i), "-", str(j))] = (
                edges[0][edge_mask],
                edges[1][edge_mask],
            )
    old_g = g
    g = dgl.heterograph(graph_data)
    g.nodes["2"].data = {key: old_g.ndata[key][ht_idx == 2] for key in old_g.ndata}
    g.nodes["3"].data = {key: old_g.ndata[key][ht_idx == 3] for key in old_g.ndata}
    return g
