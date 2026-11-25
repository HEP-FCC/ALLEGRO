import numpy as np
import torch
import dgl
from torch_scatter import scatter_add, scatter_sum
from sklearn.preprocessing import StandardScaler
from torch_scatter import scatter_sum
from src.dataset.functions_data import (
    get_ratios,
    find_mask_no_energy,
    find_cluster_id,
    get_particle_features,
    get_hit_features,
    calculate_distance_to_boundary,
    concatenate_Particles_GT,
)


def create_inputs_from_table(
    output, hits_only, prediction=False, hit_chis=False, pos_pxpy=False, is_Ks=False
):
    """Used by graph creation to get nodes and edge features

    Args:
        output (_type_): input from the root reading
        hits_only (_type_): reading only hits or also tracks
        prediction (bool, optional): if running in eval mode. Defaults to False.

    Returns:
        _type_: all information to construct a graph
    """
    number_hits = np.int32(np.sum(output["pf_mask"][0]))
    number_part = np.int32(np.sum(output["pf_mask"][1]))
    # t0 = time.time()
    (
        pos_xyz_hits,
        pos_pxpypz,
        p_hits,
        e_hits,
        hit_particle_link,
        pandora_cluster,
        pandora_cluster_energy,
        pfo_energy,
        pandora_mom,
        pandora_ref_point,
        unique_list_particles,
        cluster_id,
        hit_type_feature,
        pandora_pfo_link,
        daughters,
        hit_link_modified,
        connection_list,
        chi_squared_tracks,
    ) = get_hit_features(
        output,
        number_hits,
        prediction,
        number_part,
        hit_chis=hit_chis,
        pos_pxpy=pos_pxpy,
        is_Ks=is_Ks,
    )
    # t1 = time.time()
    # features particles
    y_data_graph = get_particle_features(
        unique_list_particles, output, prediction, connection_list
    )
    # t2 = time.time()
    # wandb.log({"time_get_hit_features": t1 - t0, "time_get_particle_features": t2 - t1})
    assert len(y_data_graph) == len(unique_list_particles)
    # remove particles that have no energy, no hits or only track hits
    mask_hits, mask_particles = find_mask_no_energy(
        cluster_id,
        hit_type_feature,
        e_hits,
        y_data_graph,
        daughters,
        prediction,
        is_Ks=is_Ks,
    )
    # create mapping from links to number of particles in the event
    cluster_id, unique_list_particles = find_cluster_id(hit_particle_link[~mask_hits])
    y_data_graph.mask(~mask_particles)

    if prediction:
        if is_Ks:
            result = [
                y_data_graph,  # y_data_graph[~mask_particles],
                p_hits[~mask_hits],
                e_hits[~mask_hits],
                cluster_id,
                hit_particle_link[~mask_hits],
                pos_xyz_hits[~mask_hits],
                pos_pxpypz[~mask_hits],
                pandora_cluster[~mask_hits],
                pandora_cluster_energy[~mask_hits],
                pandora_mom[~mask_hits],
                pandora_ref_point[~mask_hits],
                pfo_energy[~mask_hits],
                pandora_pfo_link[~mask_hits],
                hit_type_feature[~mask_hits],
                hit_link_modified[~mask_hits],
                daughters[~mask_hits]
            ]
        else:
            result = [
                y_data_graph,  # y_data_graph[~mask_particles],
                p_hits[~mask_hits],
                e_hits[~mask_hits],
                cluster_id,
                hit_particle_link[~mask_hits],
                pos_xyz_hits[~mask_hits],
                pos_pxpypz[~mask_hits],
                pandora_cluster[~mask_hits],
                pandora_cluster_energy[~mask_hits],
                pandora_mom,
                pandora_ref_point,
                pfo_energy[~mask_hits],
                pandora_pfo_link[~mask_hits],
                hit_type_feature[~mask_hits],
                hit_link_modified[~mask_hits],
            ]
    else:
        result = [
            y_data_graph,  # y_data_graph[~mask_particles],
            p_hits[~mask_hits],
            e_hits[~mask_hits],
            cluster_id,
            hit_particle_link[~mask_hits],
            pos_xyz_hits[~mask_hits],
            pos_pxpypz[~mask_hits],
            pandora_cluster,
            pandora_cluster_energy,
            pandora_mom,
            pandora_ref_point,
            pfo_energy,
            pandora_pfo_link,
            hit_type_feature[~mask_hits],
            hit_link_modified[~mask_hits], 
            daughters[~mask_hits]
        ]
    if hit_chis:
        result.append(
            chi_squared_tracks[~mask_hits],
        )
    else:
        result.append(None)
    hit_type = hit_type_feature[~mask_hits]
    # if hits only remove tracks, otherwise leave tracks
    if hits_only:
        hit_mask = (hit_type == 0) | (hit_type == 1)
        hit_mask = ~hit_mask
        for i in range(1, len(result)):
            if result[i] is not None:
                result[i] = result[i][hit_mask]
        hit_type_one_hot = torch.nn.functional.one_hot(
            hit_type_feature[~mask_hits][hit_mask] - 2, num_classes=2
        )

    else:
        # if we want the tracks keep only 1 track hit per charged particle.
        hit_mask = hit_type == 10
        hit_mask = ~hit_mask
        for i in range(1, len(result)):
            if result[i] is not None:
                # if len(result[i].shape) == 2 and result[i].shape[0] == 3:
                #     result[i] = result[i][:, hit_mask]
                # else:
                #     result[i] = result[i][hit_mask]
                result[i] = result[i][hit_mask]
        hit_type_one_hot = torch.nn.functional.one_hot(
            hit_type_feature[~mask_hits][hit_mask], num_classes=5
        )
    result.append(hit_type_one_hot)
    result.append(connection_list)
    return result

def remove_hittype0(graph):
    filt = graph.ndata["hit_type"] == 0
    # graph.ndata["hit_type"] -= 1
    return dgl.remove_nodes(graph, torch.where(filt)[0])

def store_track_at_vertex_at_track_at_calo(graph):
    # To make it compatible with clustering, remove the 0 hit type nodes and store them as pos_pxpypz_at_vertex
    tracks_at_calo = graph.ndata["hit_type"] == 1
    tracks_at_vertex = graph.ndata["hit_type"] == 0
    part = graph.ndata["particle_number"].long()
    assert (part[tracks_at_calo] == part[tracks_at_vertex]).all()
    graph.ndata["pos_pxpypz_at_vertex"] = torch.zeros_like(graph.ndata["pos_pxpypz"])
    graph.ndata["pos_pxpypz_at_vertex"][tracks_at_calo] = graph.ndata["pos_pxpypz"][tracks_at_vertex]
    return remove_hittype0(graph)

def create_graph(
    output,
    config=None,
    n_noise=0,
):
    ks_dataset = np.float32(np.sum(output["pf_mask"][2]))
    hits_only = config.graph_config.get(
        "only_hits", False
    )  # Whether to only include hits in the graph
    # standardize_coords = config.graph_config.get("standardize_coords", False)
    extended_coords = config.graph_config.get("extended_coords", False)
    prediction = config.graph_config.get("prediction", False)
    hit_chis = config.graph_config.get("hit_chis_track", False)
    pos_pxpy = config.graph_config.get("pos_pxpy", False)
    is_Ks = (torch.sum(torch.Tensor([ks_dataset])))!=0 #config.graph_config.get("ks", False)
    (
        y_data_graph,
        p_hits,
        e_hits,
        cluster_id,
        hit_particle_link,
        pos_xyz_hits,
        pos_pxpypz,
        pandora_cluster,
        pandora_cluster_energy,
        pandora_mom,
        pandora_ref_point,
        pandora_pfo_energy,
        pandora_pfo_link,
        hit_type,
        hit_link_modified,
        daugthers, 
        chi_squared_tracks,
        hit_type_one_hot,
        connections_list,
    ) = create_inputs_from_table(
        output,
        hits_only=hits_only,
        prediction=prediction,
        hit_chis=hit_chis,
        pos_pxpy=pos_pxpy,
        is_Ks=is_Ks,
    )
    graph_coordinates = pos_xyz_hits  # / 3330  # divide by detector size
    if pos_xyz_hits.shape[0] > 0:
        graph_empty = False
        g = dgl.graph(([], []))
        g.add_nodes(graph_coordinates.shape[0])
        if hits_only == False:
            hit_features_graph = torch.cat(
                (graph_coordinates, hit_type_one_hot, e_hits, p_hits), dim=1
            )  # dims = 8
        else:
            hit_features_graph = torch.cat(
                (graph_coordinates, hit_type_one_hot, e_hits, p_hits), dim=1
            )  # dims = 9

        g.ndata["h"] = hit_features_graph
        g.ndata["pos_hits_xyz"] = pos_xyz_hits
        g.ndata["pos_pxpypz"] = pos_pxpypz
        g = calculate_distance_to_boundary(g)
        g.ndata["hit_type"] = hit_type
        g.ndata[
            "e_hits"
        ] = e_hits  # if no tracks this is e and if there are tracks this fills the tracks e values with p
        if hit_chis:
            g.ndata["chi_squared_tracks"] = chi_squared_tracks
        g.ndata["particle_number"] = cluster_id
        g.ndata["hit_link_modified"] = hit_link_modified
        g.ndata["daugthers"] = daugthers
        g.ndata["particle_number_nomap"] = hit_particle_link
        if prediction:
            g.ndata["pandora_cluster"] = pandora_cluster
            g.ndata["pandora_pfo"] = pandora_pfo_link
            g.ndata["pandora_cluster_energy"] = pandora_cluster_energy
            g.ndata["pandora_pfo_energy"] = pandora_pfo_energy
            if is_Ks:
                g.ndata["pandora_momentum"] = pandora_mom
                g.ndata["pandora_reference_point"] = pandora_ref_point
        y_data_graph.calculate_corrected_E(g, connections_list)
        #if ks_dataset>0:  #is_Ks == True:
        if is_Ks == True:
            if y_data_graph.pid.flatten().shape[0] == 4 and np.count_nonzero(y_data_graph.pid.flatten() == 22) == 4:
                graph_empty = False
            else:
                graph_empty = True
            graph_empty = False
            if g.ndata["h"].shape[0] < 10 or (set(g.ndata["hit_type"].unique().tolist()) == set([0, 1]) and g.ndata["hit_type"][g.ndata["hit_type"] == 1].shape[0] < 10):
                graph_empty = True  # less than 10 hits
        if is_Ks == False:
            if len(y_data_graph) < 4:
                graph_empty = True
    else:
        graph_empty = True
        g = 0
        y_data_graph = 0
    if pos_xyz_hits.shape[0] < 10:
        graph_empty = True
    print("graph_empty", graph_empty, pos_xyz_hits.shape[0])
    if graph_empty:
        return [g, y_data_graph], graph_empty
    
    return [store_track_at_vertex_at_track_at_calo(g), y_data_graph], graph_empty


def graph_batch_func(list_graphs):
    """collator function for graph dataloader

    Args:
        list_graphs (list): list of graphs from the iterable dataset

    Returns:
        batch dgl: dgl batch of graphs
    """
    list_graphs_g = [el[0] for el in list_graphs]
    # list_y = add_batch_number(list_graphs)
    # ys = torch.cat(list_y, dim=0)
    # ys = torch.reshape(ys, [-1, list_y[0].shape[1]])
    ys = concatenate_Particles_GT(list_graphs)
    bg = dgl.batch(list_graphs_g)
    # reindex particle number
    return bg, ys
