import dgl
import torch
import os

# from alembic.command import current
from sklearn.cluster import DBSCAN, HDBSCAN
from torch_scatter import scatter_max, scatter_add, scatter_mean
import numpy as np
from src.dataset.functions_data import CachedIndexList
from src.dataset.config_main.functions_data import spherical_to_cartesian
from src.dataset.utils_hits import CachedIndexList
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import pandas as pd
import wandb
from src.utils.inference.per_particle_metrics import plot_event
import random
import string
# import hdbscan
def remove_bad_tracks_from_cluster(g, labels_hdb):
    mask_hit_type_t1 = g.ndata["hit_type"]==2
    mask_hit_type_t2 = g.ndata["hit_type"]==1
    labels_hdb_corrected_tracks = labels_hdb.clone()
    # check each cluster
    for i in range(0, torch.max(labels_hdb)+1):
        mask_labels_i = labels_hdb == i
        if torch.sum(mask_hit_type_t2[mask_labels_i])>0 and i>0:
            pos_track = g.ndata["pos_hits_xyz"][mask_labels_i*mask_hit_type_t2]
            mean_pos_cluster = torch.mean(g.ndata["pos_hits_xyz"][mask_labels_i*mask_hit_type_t1], dim=0)
            angles = torch.sum(mean_pos_cluster*g.ndata["pos_hits_xyz"][mask_labels_i*mask_hit_type_t2], dim=1)
            norms = torch.norm(mean_pos_cluster)*torch.norm(g.ndata["pos_hits_xyz"][mask_labels_i*mask_hit_type_t2], dim=1)
            angles_tracks = angles/norms
            distance_track_cluster = torch.norm(mean_pos_cluster-pos_track,dim=1)/1000
            bad_tracks = ((distance_track_cluster>0.24)+(angles_tracks<0.999))
            bad_tracks_ = torch.nonzero(bad_tracks).view(-1)
            cluster_t2_nodes = torch.nonzero(mask_labels_i & mask_hit_type_t2).view(-1)
            bad_tracks_nodes = cluster_t2_nodes[bad_tracks]
            labels_hdb_corrected_tracks[bad_tracks_nodes] = 0
    return labels_hdb_corrected_tracks


def generate_random_string(length):
    letters = string.ascii_letters + string.digits
    return "".join(random.choice(letters) for i in range(length))


def create_and_store_graph_output(
    batch_g,
    model_output,
    y,
    local_rank,
    step,
    epoch,
    path_save,
    store=False,
    predict=False,
    tracking=False,
    e_corr=None,
    shap_vals=None,
    ec_x=None,  # ec_x: "global" features (what gets inputted into the final deep neural network head) for energy correction
    tracks=False,
    store_epoch=False,
    total_number_events=0,
    pred_pos=None,
    pred_ref_pt=None,
    use_gt_clusters=False,
    pids_neutral=None,
    pids_charged=None,
    pred_pid=None,
    pred_xyz_track=None,
    number_of_fakes=None,
    extra_features=None,
    fakes_labels=None
):
    number_of_showers_total = 0
    number_of_showers_total1 = 0
    number_of_fake_showers_total1 = 0
    batch_g.ndata["coords"] = model_output[:, 0:3]
    batch_g.ndata["beta"] = model_output[:, 3]
    if not tracking:
        if e_corr is None:
            batch_g.ndata["correction"] = model_output[:, 4]
    graphs = dgl.unbatch(batch_g)
    batch_id = y.batch_number.view(-1)  # y[:, -1].view(-1)
    df_list = []
    df_list1 = []
    df_list_pandora = []
    total_number_candidates = 0
    for i in range(0, len(graphs)):
        mask = batch_id == i
        dic = {}
        dic["graph"] = graphs[i]
        y1 = y.copy()
        y1.mask(mask)
        dic["part_true"] = y1  # y[mask]
        X = dic["graph"].ndata["coords"]
        # if shap_vals is not None:
        #    dic["shap_values"] = shap_vals
        # if ec_x is not None:
        #    dic["ec_x"] = ec_x  ## ? No mask ?!?
        # if predict:
            # labels_clustering = clustering_obtain_labels(
            #     X, dic["graph"].ndata["beta"].view(-1), model_output.device, tbeta=0.2, td=0.05
            # )
        if use_gt_clusters:
            labels_hdb = dic["graph"].ndata["particle_number"].type(torch.int64)
        else:
            labels_hdb = hfdb_obtain_labels(X, model_output.device)
            #labels_hdb = labels_clustering 
            # betas = torch.sigmoid(dic["graph"].ndata["beta"])
            # labels_clustering = clustering_obtain_labels(
            #     X, betas.view(-1), model_output.device,  tbeta=0.7, td=0.3
            # )
            # labels_hdb = labels_clustering

            num_clusters = len(labels_hdb.unique())
            #if labels_hdb.min() == 0 and labels_hdb.sum() == 0:
            #    labels_hdb += 1  # Quick hack
            #    raise Exception("!!!! Labels==0 !!!!")
            labels_hdb = remove_bad_tracks_from_cluster(dic["graph"], labels_hdb)
        if predict:
            labels_pandora = get_labels_pandora(tracks, dic, model_output.device)
            num_clusters_pandora = len(labels_pandora.unique())
        particle_ids = torch.unique(dic["graph"].ndata["particle_number"])
        
        ## add function to remove tracks if they are bad
        shower_p_unique_hdb, row_ind_hdb, col_ind_hdb, i_m_w_hdb, iou_m = match_showers(
            labels_hdb,
            dic,
            particle_ids,
            model_output,
            local_rank,
            i,
            path_save,
            tracks=tracks,
            hdbscan=True,
        )
        if predict:
            (
                shower_p_unique_pandora,
                row_ind_pandora,
                col_ind_pandora,
                i_m_w_pandora,
                iou_m_pandora,
            ) = match_showers(
                labels_pandora,
                dic,
                particle_ids,
                model_output,
                local_rank,
                i,
                path_save,
                pandora=True,
                tracks=tracks,
            )

        # # if len(row_ind_hdb) < len(dic["part_true"]):
        # print(len(row_ind_hdb), len(dic["part_true"]))
        # print("storing  event", local_rank, step, i)
        # path_graphs_all_comparing = os.path.join(path_save, "graphs_all_comparing")
        # if not os.path.exists(path_graphs_all_comparing):
        #    os.makedirs(path_graphs_all_comparing)
        '''torch.save(
            dic,
            path_save
            + "/graphs_all_comparing_Gregor/"
            + str(local_rank)
            + "_"
            + str(step)
            + "_"
            + str(i)
            + ".pt",
        #  )'''
        # torch.save(
        #     dic,
        #     path_save
        #     + "/graphs_3M/"
        #     + str(local_rank)
        #     + "_"
        #     + str(step)
        #     + "_"
        #     + str(i)
        #     + ".pt",
        #  )
        
        if len(shower_p_unique_hdb) > 1:
            # df_event, number_of_showers_total = generate_showers_data_frame(
            #     labels_clustering,
            #     labels_clustering,
            #     dic,
            #     shower_p_unique,
            #     particle_ids,
            #     row_ind,
            #     col_ind,
            #     i_m_w,
            #     e_corr=e_corr,
            #     number_of_showers_total=number_of_showers_total,
            #     step=step,
            #     number_in_batch=i,
            #     tracks=tracks,
            # )
            # if pred_pos is not None:
            # Apply temporary correction
            import math
            # phi = math.atan2(pred_pos[:, 1], pred_pos[:, 0])
            # phi = torch.atan2(pred_pos[:, 1], pred_pos[:, 0])
            # theta = torch.acos(pred_pos[:, 2] / torch.norm(pred_pos, dim=1))
            # pred_pos = spherical_to_cartesian(theta, phi, torch.norm(pred_pos, dim=1), normalized=True)
            # pred_pos= pred_pos.to(model_output.device)
            df_event1, number_of_showers_total1, number_of_fake_showers_total1 = generate_showers_data_frame(
                labels_hdb,
                dic,
                shower_p_unique_hdb,
                particle_ids,
                row_ind_hdb,
                col_ind_hdb,
                i_m_w_hdb,
                e_corr=e_corr,
                number_of_showers_total=number_of_showers_total1,
                step=step,
                number_in_batch=total_number_events,
                tracks=tracks,
                ec_x=ec_x,
                shap_vals=shap_vals,
                pred_pos=pred_pos,
                pred_ref_pt=pred_ref_pt,
                pred_pid=pred_pid,
                save_plots_to_folder=path_save + "/ML_",
                number_of_fakes=number_of_fakes,
                number_of_fake_showers_total=number_of_fake_showers_total1,
                extra_features=extra_features, # To help with the debugging of the fakes
                #fakes_labels=fakes_labels
            )
            if len(df_event1) > 1:
                df_list1.append(df_event1)
            if predict:
                df_event_pandora = generate_showers_data_frame(
                    labels_pandora,
                    dic,
                    shower_p_unique_pandora,
                    particle_ids,
                    row_ind_pandora,
                    col_ind_pandora,
                    i_m_w_pandora,
                    pandora=True,
                    tracking=tracking,
                    step=step,
                    number_in_batch=total_number_events,
                    tracks=tracks,
                    save_plots_to_folder=path_save + "/Pandora",
                )
                if df_event_pandora is not None and type(df_event_pandora) is not tuple:
                    df_list_pandora.append(df_event_pandora)
                else:
                    print("Not appending to df_list_pandora")
            total_number_events = total_number_events + 1
        # print("number of showers total", number_of_showers_total)
        # number_of_showers_total = number_of_showers_total + len(shower_p_unique_hdb)
        # print("number of showers total", number_of_showers_total)

    df_batch1 = pd.concat(df_list1)
    if predict:
        df_batch_pandora = pd.concat(df_list_pandora)
    else:
        df_batch = []
        df_batch_pandora = []
    #
    if store:
        store_at_batch_end(
            path_save,
            df_batch1,
            df_batch_pandora,
            # df_batch,
            local_rank,
            step,
            epoch,
            predict=predict,
            store=store_epoch,
        )
    if predict:
        return df_batch_pandora, df_batch1, total_number_events
    else:
        return df_batch1


def store_at_batch_end(
    path_save,
    df_batch1,
    df_batch_pandora,
    # df_batch,
    local_rank=0,
    step=0,
    epoch=None,
    predict=False,
    store=False,
):
    if predict:
        path_save_ = (
            path_save
            + str(local_rank)
            + "_"
            + str(step)
            + "_"
            + str(epoch)
            + ".pt"
        )
        # if store and predict:
        #     df_batch.to_pickle(path_save_)
        # log_efficiency(df_batch, clustering=True)
    path_save_ = (
        path_save
        + str(local_rank)
        + "_"
        + str(step)
        + "_"
        + str(epoch)
        + ".pt"
    )
    if store and predict:
        df_batch1.to_pickle(path_save_)
    if predict:
        path_save_pandora = (
            path_save
            + str(local_rank)
            + "_"
            + str(step)
            + "_"
            + str(epoch)
            + "_pandora.pt"
        )
        if store and predict:
            df_batch_pandora.to_pickle(path_save_pandora)
    log_efficiency(df_batch1)
    if predict:
        log_efficiency(df_batch_pandora, pandora=True)


def log_efficiency(df, pandora=False, clustering=False):
    mask = ~np.isnan(df["reco_showers_E"])
    eff = np.sum(~np.isnan(df["pred_showers_E"][mask].values)) / len(
        df["pred_showers_E"][mask].values
    )
    if pandora:
        wandb.log({"efficiency validation pandora": eff})
    elif clustering:
        wandb.log({"efficiency validation clustering": eff})
    else:
        wandb.log({"efficiency validation": eff})


def remove_labels_of_double_showers(labels, g):
    is_track_per_shower = scatter_add(1*(g.ndata["hit_type"] == 1), labels).int()
    e_hits_sum = scatter_add(g.ndata["e_hits"].view(-1), labels.view(-1).long()).int()
    mask_tracks = g.ndata["hit_type"]==1
    for i, label_i in enumerate(torch.unique(labels)):
        if is_track_per_shower[label_i]==2:
            if label_i>0:
            #if there are two tracks
                sum_pred_2 = e_hits_sum[label_i]
                mask_labels_i = labels == label_i
                mask_label_i_and_is_track = mask_labels_i * mask_tracks
                if not mask_label_i_and_is_track.sum()==2:
                    print("Error")
                    print(mask_label_i_and_is_track.tolist(), mask_label_i_and_is_track.sum())
                    print(label_i)
                    print(sum_pred_2)
                assert mask_label_i_and_is_track.sum()==2
                tracks_E = g.ndata['h'][:,-1][mask_label_i_and_is_track]
                chi_tracks = g.ndata['chi_squared_tracks'][mask_label_i_and_is_track]
                ind_min_E = torch.argmax(torch.abs(tracks_E - sum_pred_2))
                ind_min_chi = torch.argmax(chi_tracks)
                # Calc distance track cluster:
                mask_hit_type_t1 = g.ndata["hit_type"][mask_labels_i]==2
                mask_hit_type_t2 = g.ndata["hit_type"][mask_labels_i]==1
                mask_all = mask_hit_type_t1
                # the other error could come from no hits in the ECAL for a cluster
                index_sorted = torch.argsort(g.ndata["radial_distance"][mask_labels_i][mask_hit_type_t1])
                mask_sorted_ind = index_sorted<10
                mean_pos_cluster = torch.mean(g.ndata["pos_hits_xyz"][mask_labels_i][mask_all][mask_sorted_ind], dim=0)

                pos_track = g.ndata["pos_hits_xyz"][mask_labels_i][mask_hit_type_t2]
                distance_track_cluster = torch.norm(pos_track-mean_pos_cluster, dim=1)/1000
                ind_max_dtc = torch.argmax(distance_track_cluster)
                if torch.min(distance_track_cluster)<0.4:
                    ind_min = ind_max_dtc
                elif ind_min_E == ind_min_chi:
                    ind_min = ind_min_E
                # if the chi tracks are very similar pick the lowest E diff
                elif torch.max(chi_tracks-torch.min(chi_tracks))<2:
                    ind_min = ind_min_E
                else:
                    ind_min = ind_min_chi
                ind_change = torch.argwhere(mask_label_i_and_is_track)[ind_min]
                labels[ind_change]=0
    return labels


def generate_showers_data_frame(
    labels,
    dic,
    shower_p_unique,
    particle_ids,
    row_ind,
    col_ind,
    i_m_w,
    pandora=False,
    tracking=False,
    e_corr=None,
    number_of_showers_total=None,
    step=0,
    number_in_batch=0,
    tracks=False,
    shap_vals=None,
    ec_x=None,
    pred_pos=None,
    pred_pid=None,
    save_plots_to_folder="",
    pred_ref_pt=None,
    number_of_fake_showers_total=None,
    number_of_fakes=None,
    extra_features=None,
):
    shap = shap_vals is not None
    e_pred_showers = scatter_add(dic["graph"].ndata["e_hits"].view(-1), labels)
    e_pred_showers_ecal = scatter_add(1*(dic["graph"].ndata["hit_type"].view(-1)==2), labels)
    e_pred_showers_hcal = scatter_add(1*(dic["graph"].ndata["hit_type"].view(-1)==3), labels)
    if pandora:
        e_pred_showers_cali = scatter_mean(
            dic["graph"].ndata["pandora_pfo_energy"].view(-1), labels
        )
        e_pred_showers_pfo = scatter_mean(
            dic["graph"].ndata["pandora_pfo_energy"].view(-1), labels
        )
        # px_pred_pfo = scatter_mean(dic["graph"].ndata["hit_px"], labels)
        # py_pred_pfo = scatter_mean(dic["graph"].ndata["hit_py"], labels)
        # pz_pred_pfo = scatter_mean(dic["graph"].ndata["hit_pz"], labels)
        # p_pred_pfo = scatter_mean(dic["graph"].ndata["pos_pxpypz"], labels) # FIX THIS: the shape of pos_pxpypz is [-1, 3]
        calc_pandora_momentum = "pandora_momentum" in dic["graph"].ndata
        if calc_pandora_momentum:
            px_pred_pfo = scatter_mean(
                dic["graph"].ndata["pandora_momentum"][:, 0], labels
            )
            py_pred_pfo = scatter_mean(
                dic["graph"].ndata["pandora_momentum"][:, 1], labels
            )
            pz_pred_pfo = scatter_mean(
                dic["graph"].ndata["pandora_momentum"][:, 2], labels
            )
            ref_pt_px_pred_pfo = scatter_mean(
                dic["graph"].ndata["pandora_reference_point"][:, 0], labels
            )
            ref_pt_py_pred_pfo = scatter_mean(
                dic["graph"].ndata["pandora_reference_point"][:, 1], labels
            )
            
            ref_pt_pz_pred_pfo = scatter_mean(
                dic["graph"].ndata["pandora_reference_point"][:, 2], labels
            )
            pandora_pid = scatter_mean(
                dic["graph"].ndata["pandora_pid"], labels
            )
            ref_pt_pred_pfo = torch.stack(
                (ref_pt_px_pred_pfo, ref_pt_py_pred_pfo, ref_pt_pz_pred_pfo), dim=1
            )
            # p_pred_pandora = scatter_mean(dic["graph"].ndata["pandora_momentum"], labels)
            p_pred_pandora = torch.stack((px_pred_pfo, py_pred_pfo, pz_pred_pfo), dim=1)
            p_size_pandora = torch.norm(p_pred_pandora, dim=1)
            pxyz_pred_pfo = (
                p_pred_pandora  # / torch.norm(p_pred_pandora, dim=1).view(-1, 1)
            )
    else:
        if e_corr is None:
            corrections_per_shower = get_correction_per_shower(labels, dic)
            e_pred_showers_cali = e_pred_showers * corrections_per_shower
        else:
            corrections_per_shower = e_corr.view(-1)
            if number_of_fakes > 0:
                corrections_per_shower_fakes = corrections_per_shower[-number_of_fakes:]
                corrections_per_shower = corrections_per_shower[:-number_of_fakes]

    e_reco_showers = scatter_add(
        dic["graph"].ndata["e_hits"].view(-1),
        dic["graph"].ndata["particle_number"].long(),
    )
    is_track_in_MC = scatter_add(
        1*(dic["graph"].ndata["hit_type"].view(-1)==1),
        dic["graph"].ndata["particle_number"].long(),
    )
    distance_to_cluster_all = distance_to_cluster_track(dic, is_track_in_MC)
    distances, number_of_tracks = distance_to_true_cluster_of_track(dic, labels)

    row_ind = torch.Tensor(row_ind).to(e_pred_showers.device).long()
    col_ind = torch.Tensor(col_ind).to(e_pred_showers.device).long()

    if torch.sum(particle_ids == 0) > 0:
        # particle id can be 0 because there is noise
        # then row ind 0 in any case corresponds to particle 1.
        # if there is particle_id 0 then row_ind should be +1?
        row_ind_ = row_ind - 1
    else:
        # if there is no zero then index 0 corresponds to particle 1.
        row_ind_ = row_ind

    pred_showers = shower_p_unique
    energy_t = (
        dic["part_true"].E_corrected.view(-1).to(e_pred_showers.device)
    ).float()  # dic["part_true"][:, 3].to(e_pred_showers.device)
    vertex = dic["part_true"].vertex.to(e_pred_showers.device)
    pos_t = dic["part_true"].coord.to(e_pred_showers.device)
    pid_t = dic["part_true"].pid.to(e_pred_showers.device)
    if not pandora:
        labels = remove_labels_of_double_showers(labels, dic["graph"])
    is_track_per_shower = scatter_add(1*(dic["graph"].ndata["hit_type"] == 1), labels).int()
    is_track = torch.zeros(energy_t.shape).to(e_pred_showers.device)
    if shap:
        matched_shap_vals = torch.zeros((energy_t.shape[0], ec_x.shape[1])) * (
            torch.nan
        )
        matched_shap_vals = matched_shap_vals.numpy()
        matched_ec_x = torch.zeros((energy_t.shape[0], ec_x.shape[1])) * (torch.nan)
        matched_ec_x = matched_ec_x.numpy()
    index_matches = col_ind + 1
    index_matches = index_matches.to(e_pred_showers.device).long()
    matched_es = torch.zeros_like(energy_t) * (torch.nan)
    matched_ECAL = torch.zeros_like(energy_t) * (torch.nan)
    matched_HCAL = torch.zeros_like(energy_t) * (torch.nan)
    matched_positions = torch.zeros((energy_t.shape[0], 3)) * (torch.nan)
    matched_positions = matched_positions.to(e_pred_showers.device)
    matched_ref_pt = torch.zeros((energy_t.shape[0], 3)) * (torch.nan)
    matched_ref_pt = matched_ref_pt.to(e_pred_showers.device)
    matched_pid = torch.zeros_like(energy_t) * (torch.nan)
    matched_pid = matched_pid.to(e_pred_showers.device).long()
    matched_positions_pfo = torch.zeros((energy_t.shape[0], 3)) * (torch.nan)
    matched_positions_pfo = matched_positions_pfo.to(e_pred_showers.device)
    matched_pandora_pid = (torch.zeros((energy_t.shape[0])) * (torch.nan)).to(e_pred_showers.device)
    matched_ref_pts_pfo =   torch.zeros((energy_t.shape[0], 3)) * (torch.nan)
    matched_ref_pts_pfo = matched_ref_pts_pfo.to(e_pred_showers.device)
    matched_es = matched_es.to(e_pred_showers.device)
    matched_es[row_ind_] = e_pred_showers[index_matches]

    matched_ECAL = matched_ECAL.to(e_pred_showers.device)
    matched_ECAL[row_ind_] = 1.0*e_pred_showers_ecal[index_matches]
    matched_HCAL = matched_HCAL.to(e_pred_showers.device)
    matched_HCAL[row_ind_] = 1.0*e_pred_showers_hcal[index_matches]


    n_extra_features = 7 # n nodes, 1 highest betas
    matched_extra_features = torch.zeros((energy_t.shape[0], n_extra_features)) * (torch.nan)
    #matched_extra_features = matched_extra_features.to(e_pred_showers.device)
    if pandora:
        matched_es_cali = matched_es.clone()
        matched_es_cali[row_ind_] = e_pred_showers_cali[index_matches]
        matched_es_cali_pfo = matched_es.clone()
        matched_es_cali_pfo[row_ind_] = e_pred_showers_pfo[index_matches]
        matched_pandora_pid[row_ind_] = pandora_pid[index_matches]
        if calc_pandora_momentum:
            matched_positions_pfo[row_ind_] = pxyz_pred_pfo[index_matches]
            matched_ref_pts_pfo[row_ind_] = ref_pt_pred_pfo[index_matches]
        is_track[row_ind_] = is_track_per_shower[index_matches].float()
    else:
        if e_corr is None:
            matched_es_cali = matched_es.clone()
            matched_es_cali[row_ind_] = e_pred_showers_cali[index_matches]
            calibration_per_shower = matched_es.clone()
            calibration_per_shower[row_ind_] = corrections_per_shower[index_matches]
        else:
            matched_es_cali = matched_es.clone()
            number_of_showers = e_pred_showers[index_matches].shape[0] # DOESN'T INCLUDE THE FAKE SHOWERS
            #number_of_fake_showers = e_pred_showers.shape[0] - number_of_showers
            matched_es_cali[row_ind_] = (
                corrections_per_shower[
                    number_of_showers_total : number_of_showers_total
                    + number_of_showers
                ]
                #* e_pred_showers[index_matches]
            )
            # if len(row_ind) and len(index_matches):
            #     assert row_ind.max() < len(is_track)
            #     assert index_matches.max() < len(is_track_per_shower)
            if pred_pos is not None:
                matched_positions[row_ind_] = pred_pos[number_of_showers_total : number_of_showers_total
                    + number_of_showers]
                matched_ref_pt[row_ind_] = pred_ref_pt[number_of_showers_total : number_of_showers_total + number_of_showers]
                matched_pid[row_ind_] = pred_pid[number_of_showers_total : number_of_showers_total + number_of_showers]
                if not pandora:
                    matched_extra_features[row_ind_] = torch.tensor(extra_features[number_of_showers_total : number_of_showers_total + number_of_showers])
            if shap:
                matched_shap_vals[row_ind_.cpu()] = shap_vals[index_matches.cpu()]
                matched_ec_x[row_ind_.cpu()] = ec_x[index_matches.cpu()]
            calibration_per_shower = matched_es.clone()
            calibration_per_shower[row_ind_] = corrections_per_shower[
                number_of_showers_total : number_of_showers_total + number_of_showers
            ]
            number_of_showers_total = number_of_showers_total + number_of_showers
        is_track[row_ind_] = is_track_per_shower[index_matches].float()

    # match the tracks to the particle
    dic["graph"].ndata["particle_number_u"]= dic["graph"].ndata["particle_number"].clone()
    dic["graph"].ndata["particle_number_u"][dic["graph"].ndata["particle_number_u"]==0]=100
    tracks_label = scatter_max((dic["graph"].ndata["hit_type"] == 1)*(dic["graph"].ndata["particle_number_u"]), labels)[0].int()
    tracks_label = tracks_label-1
    tracks_label[tracks_label<0]=0
    matched_es_tracks = torch.zeros_like(energy_t) * (torch.nan)
    matched_es_tracks_1 = torch.zeros_like(energy_t) * (torch.nan)
    matched_es_tracks[row_ind_]=row_ind_.float()
    matched_es_tracks_1[row_ind_]=tracks_label[index_matches].float()
    matched_es_tracks_1 = 1.0*(matched_es_tracks==matched_es_tracks_1)
    matched_es_tracks_1 = matched_es_tracks_1*is_track


    intersection_E = torch.zeros_like(energy_t) * (torch.nan)
    if len(col_ind) > 0:
        ie_e = obtain_intersection_values(i_m_w, row_ind, col_ind, dic)
        intersection_E[row_ind_] = ie_e.to(e_pred_showers.device)
        pred_showers[index_matches] = -1
        pred_showers[
            0
        ] = (
            -1
        )  # This takes into account that the class 0 for pandora and for dbscan is noise
        mask = pred_showers != -1
        fakes_in_event = mask.sum()
        fake_showers_e = e_pred_showers[mask]
        fake_showers_e_hcal = e_pred_showers_hcal[mask]
        fake_showers_e_ecal = e_pred_showers_ecal[mask]
        number_of_fake_showers = mask.sum()
        # mask is the labels which have not been matched
        all_labels = labels.unique().to(e_pred_showers.device)
        number_of_fake_showers = mask.sum()
        fakes_labels = torch.where(mask)[0].to(e_pred_showers.device)
        fake_showers_distance_to_cluster = distances[fakes_labels.cpu()]
        fake_showers_num_tracks = number_of_tracks[fakes_labels.cpu()]

        if e_corr is None or pandora:
            fake_showers_e_cali = e_pred_showers_cali[mask]
            # fakes_positions = dic["graph"].ndata["coords"][mask]
        else:
            #fake_showers_e_cali = corrections_per_shower[number_of_showers_total:number_of_showers_total+number_of_showers][mask]# * (torch.nan)
            #fakes_positions = torch.zeros((fake_showers_e.shape[0], 3)) * (torch.nan)
            #fake_showers_e_cali = fake_showers_e
            #fakes_pid_pred = torch.zeros((fake_showers_e.shape[0])) * (torch.nan) # just for now for debugigng
            #fakes_positions = fakes_positions.to(e_pred_showers.device)
            #fakes_pid_pred = fakes_pid_pred.to(e_pred_showers.device)
            fakes_positions = pred_pos[-number_of_fakes:][number_of_fake_showers_total:number_of_fake_showers_total+number_of_fake_showers]
            fake_showers_e_cali = e_corr[-number_of_fakes:][number_of_fake_showers_total:number_of_fake_showers_total+number_of_fake_showers]
            fakes_pid_pred = pred_pid[-number_of_fakes:][number_of_fake_showers_total:number_of_fake_showers_total+number_of_fake_showers]
            fake_showers_e_reco = e_reco_showers[-number_of_fakes:][number_of_fake_showers_total:number_of_fake_showers_total+number_of_fake_showers]
            fakes_positions = fakes_positions.to(e_pred_showers.device)
            fakes_extra_features = extra_features[-number_of_fakes:][number_of_fake_showers_total:number_of_fake_showers_total+number_of_fake_showers]
            fake_showers_e_cali = fake_showers_e_cali.to(e_pred_showers.device)
            fakes_pid_pred = fakes_pid_pred.to(e_pred_showers.device)
            fake_showers_e_reco = fake_showers_e_reco.to(e_pred_showers.device)
            #fakes_pid_pred = pred_pid[number_of_showers_total:number_of_showers_total+number_of_showers][mask]
            #fakes_positions = fakes_positions.to(e_pred_showers.device)
        if pandora:
            fake_pandora_pid = (torch.zeros((fake_showers_e.shape[0], 3)) * (torch.nan)).to(e_pred_showers.device)
            fake_pandora_pid = pandora_pid[mask]
            if calc_pandora_momentum:
                fake_positions_pfo = torch.zeros((fake_showers_e.shape[0], 3)) * (torch.nan)
                fake_positions_pfo = fake_positions_pfo.to(e_pred_showers.device)
                fake_positions_pfo = pxyz_pred_pfo[mask]
                fakes_positions_ref = (torch.zeros((fake_showers_e.shape[0], 3)) * (torch.nan)).to(e_pred_showers.device)
                fakes_positions_ref = ref_pt_pred_pfo[mask]
        if not pandora:
            if e_corr is None:
                fake_showers_e_cali_factor = corrections_per_shower[mask]
            else:
                fake_showers_e_cali_factor = fake_showers_e_cali
        fake_showers_showers_e_truw = torch.zeros((fake_showers_e.shape[0])) * (
            torch.nan
        )
        fake_showers_vertex = torch.zeros((fake_showers_e.shape[0], 3)) * (torch.nan)
        fakes_is_track = (torch.zeros((fake_showers_e.shape[0])) * (torch.nan)).to(e_pred_showers.device)
        fakes_is_track = is_track_per_shower[mask]
        fakes_positions_t = torch.zeros((fake_showers_e.shape[0], 3)) * (torch.nan)
        if not pandora:
            number_of_fake_showers_total = number_of_fake_showers_total + number_of_fake_showers
        # """if shap:
        #     fake_showers_shap_vals = torch.zeros((fake_showers_e.shape[0], shap_vals_t.shape[1])) * (
        #         torch.nan
        #     )
        #     fake_showers_ec_x_t = torch.zeros((fake_showers_e.shape[0], ec_x_t.shape[1])) * (
        #         torch.nan
        #     )
        #     #fake_showers_shap_vals = fake_showers_shap_vals.to(e_pred_showers.device)
        #     #fake_showers_ec_x_t = fake_showers_ec_x_t.to(e_pred_showers.device)
        #     shap_vals_t = torch.cat((torch.tensor(shap_vals_t), fake_showers_shap_vals), dim=0)
        #     ec_x_t = torch.cat((torch.tensor(ec_x_t), fake_showers_ec_x_t), dim=0)
        # """

        fake_showers_showers_e_truw = fake_showers_showers_e_truw.to(
            e_pred_showers.device
        )
        fakes_positions_t = fakes_positions_t.to(e_pred_showers.device)
        fake_showers_vertex = fake_showers_vertex.to(e_pred_showers.device)
        energy_t = torch.cat(
            (energy_t, fake_showers_showers_e_truw),
            dim=0,
        )
        vertex = torch.cat((vertex, fake_showers_vertex), dim=0)
        pid_t = torch.cat(
            (pid_t.view(-1), fake_showers_showers_e_truw),
            dim=0,
        )
        pos_t = torch.cat(
            (pos_t, fakes_positions_t),
            dim=0,
        )
        e_reco = torch.cat((e_reco_showers[1:], fake_showers_showers_e_truw), dim=0)
        is_track_in_MC = torch.cat((is_track_in_MC[1:], fake_showers_num_tracks.to(e_reco.device)), dim=0)
        distance_to_cluster_MC = torch.cat((distance_to_cluster_all[1:], fake_showers_distance_to_cluster.to(e_reco.device)), dim=0)
        e_pred = torch.cat((matched_es, fake_showers_e), dim=0)
        e_pred_ECAL =  torch.cat((matched_ECAL, fake_showers_e_ecal), dim=0)
        e_pred_HCAL =  torch.cat((matched_HCAL, fake_showers_e_hcal), dim=0)
        e_pred_cali = torch.cat((matched_es_cali, fake_showers_e_cali), dim=0)
        if pred_pos is not None:
            e_pred_pos = torch.cat((matched_positions, fakes_positions), dim=0)
            e_pred_pid = torch.cat((matched_pid, fakes_pid_pred), dim=0)
            e_pred_ref_pt = torch.cat((matched_ref_pt, fakes_positions), dim=0)
            extra_features_all = torch.cat((matched_extra_features, torch.tensor(fakes_extra_features)), dim=0)
        if pandora:
            e_pred_cali_pfo = torch.cat(
                (matched_es_cali_pfo, fake_showers_e_cali), dim=0
            )
            positions_pfo = torch.cat((matched_positions_pfo, fake_positions_pfo), dim=0)
            pandora_pid = torch.cat((matched_pandora_pid, fake_pandora_pid), dim=0)
            ref_pts_pfo =   torch.cat((matched_ref_pts_pfo, fakes_positions_ref), dim=0)
        if not pandora:
            calibration_factor = torch.cat(
                (calibration_per_shower, fake_showers_e_cali_factor), dim=0
            )
        if shap:
            # pad
            matched_shap_vals = torch.cat(
                (
                    torch.tensor(matched_shap_vals),
                    torch.zeros((fake_showers_e.shape[0], shap_vals.shape[1])),
                ),
                dim=0,
            )
            matched_ec_x = torch.cat(
                (
                    torch.tensor(matched_ec_x),
                    torch.zeros((fake_showers_e.shape[0], ec_x.shape[1])),
                ),
                dim=0,
            )

        e_pred_t = torch.cat(
            (
                intersection_E,
                torch.zeros_like(fake_showers_e) * (torch.nan),
            ),
            dim=0,
        )
        # e_pred_t_pandora = torch.cat(
        #     (
        #         intersection_E,
        #         torch.zeros_like(fake_showers_e) * (-200),
        #         torch.zeros_like(fake_showers_e_pandora) * (-100),
        #     ),
        #     dim=0,
        # )
        is_track = torch.cat((is_track, fakes_is_track.to(is_track.device)), dim=0)
        matched_es_tracks_1 = torch.cat((matched_es_tracks_1, 0*fakes_is_track.to(is_track.device)), dim=0)
        if pandora:
            d = {
                "true_showers_E": energy_t.detach().cpu(),
                "reco_showers_E": e_reco.detach().cpu(),
                "pred_showers_E": e_pred.detach().cpu(),
                "e_pred_and_truth": e_pred_t.detach().cpu(),
                "pandora_calibrated_E": e_pred_cali.detach().cpu(),
                "pandora_calibrated_pfo": e_pred_cali_pfo.detach().cpu(),
                "pandora_calibrated_pos": positions_pfo.detach().cpu().tolist(),
                "pandora_ref_pt": ref_pts_pfo.detach().cpu().tolist(),
                "pid": pid_t.detach().cpu(),
                "pandora_pid":pandora_pid.detach().cpu(),
                "step": torch.ones_like(energy_t.detach().cpu()) * step,
                "number_batch": torch.ones_like(energy_t.detach().cpu())
                * number_in_batch,
                "is_track_in_cluster": is_track.detach().cpu(),
                "is_track_correct":matched_es_tracks_1.detach().cpu(),
                "is_track_in_MC": is_track_in_MC.detach().cpu(),
                "distance_to_cluster_MC":distance_to_cluster_MC.detach().cpu(),
                "vertex": vertex.detach().cpu().tolist(), 
                "ECAL_hits": e_pred_ECAL.detach().cpu(),
                "HCAL_hits": e_pred_HCAL.detach().cpu(),
            }
        else:
            d = {
                "true_showers_E": energy_t.detach().cpu(),
                "reco_showers_E": e_reco.detach().cpu(),
                "pred_showers_E": e_pred.detach().cpu(),
                "e_pred_and_truth": e_pred_t.detach().cpu(),
                "pid": pid_t.detach().cpu(),
                "calibration_factor": calibration_factor.detach().cpu(),
                "calibrated_E": e_pred_cali.detach().cpu(),
                "step": torch.ones_like(energy_t.detach().cpu()) * step,
                "number_batch": torch.ones_like(energy_t.detach().cpu())
                * number_in_batch,
                "is_track_in_cluster": is_track.detach().cpu(),
                "is_track_correct":matched_es_tracks_1.detach().cpu(),
                "is_track_in_MC": is_track_in_MC.detach().cpu(),
                "distance_to_cluster_MC":distance_to_cluster_MC.detach().cpu(),
                "vertex": vertex.detach().cpu().tolist(), 
                "ECAL_hits": e_pred_ECAL.detach().cpu(),
                "HCAL_hits": e_pred_HCAL.detach().cpu(),
            }
            if pred_pos is not None:
                pred_pos1 = e_pred_pos.detach().cpu()
                pred_pid1  = e_pred_pid.detach().cpu()
                pred_ref_pt1 = e_pred_ref_pt.detach().cpu()
                d["pred_pos_matched"] = (
                    pred_pos1.tolist()
                )  # Otherwise it doesn't work nicely with Pandas DataFrames
                d["pred_pid_matched"] = pred_pid1.tolist()
                d["pred_ref_pt_matched"] = pred_ref_pt1.tolist()
                d["matched_extra_features"] = extra_features_all.detach().cpu().tolist()
        """if shap:
            print("Adding ec_x and shap_values to the DataFrame")
            d["ec_x"] = ec_x_t
            d["shap_values"] = shap_vals_t"""
        if shap:
            d["shap_values"] = matched_shap_vals.tolist()
            d["ec_x"] = matched_ec_x.tolist()
        d["true_pos"] = pos_t.detach().cpu().tolist()
        df = pd.DataFrame(data=d)
        #event_list = [40]   # Fill with the list of selected events that we want to investigate
        # if save_plots_to_folder:
        #     event_numbers = np.unique(df.number_batch)
        #     for evt in event_numbers:
        #         #if evt contains muons
        #         if 13 in dic["part_true"].pid.flatten().abs():
        #             print("Event contains muons, plotting!")
        #             # Random string
        #             if not pandora:
        #                 plot_event(
        #                     df[df.number_batch == evt],
        #                     pandora,
        #                     save_plots_to_folder + "GT_" + str(evt),
        #                     graph=dic["graph"].to("cpu"),
        #                     y=dic["part_true"],
        #                     labels=dic["graph"].ndata["particle_number"].long(),
        #                     is_track_in_cluster=df.is_track_in_cluster,
        #                     pid_filter=[13, -13]
        #                 )
                    #plot_event(
                    #    df[df.number_batch == evt],
                    #    pandora,
                    #   save_plots_to_folder + "clustering_" + str(evt),
                    #    graph=dic["graph"].to("cpu"),
                    #    y=dic["part_true"],
                    #    labels=labels.detach().cpu(),
                    #    is_track_in_cluster=df.is_track_in_cluster
                    #)

        if number_of_showers_total is None:
            return df
        else:
            return df, number_of_showers_total, number_of_fake_showers_total
    else:
        return [], 0, 0


def get_correction_per_shower(labels, dic):
    unique_labels = torch.unique(labels)
    list_corr = []
    for ii, pred_label in enumerate(unique_labels):
        if ii == 0:
            if pred_label != 0:
                list_corr.append(dic["graph"].ndata["correction"][0].view(-1) * 0)
        mask = labels == pred_label
        corrections_E_label = dic["graph"].ndata["correction"][mask]
        betas_label_indmax = torch.argmax(dic["graph"].ndata["beta"][mask])
        list_corr.append(corrections_E_label[betas_label_indmax].view(-1))

    corrections = torch.cat(list_corr, dim=0)
    return corrections


def obtain_intersection_matrix(shower_p_unique, particle_ids, labels, dic, e_hits):
    len_pred_showers = len(shower_p_unique)
    intersection_matrix = torch.zeros((len_pred_showers, len(particle_ids))).to(
        shower_p_unique.device
    )
    intersection_matrix_w = torch.zeros((len_pred_showers, len(particle_ids))).to(
        shower_p_unique.device
    )

    for index, id in enumerate(particle_ids):
        counts = torch.zeros_like(labels)
        mask_p = dic["graph"].ndata["particle_number"] == id
        h_hits = e_hits.clone()
        counts[mask_p] = 1
        h_hits[~mask_p] = 0
        intersection_matrix[:, index] = scatter_add(counts, labels)
        intersection_matrix_w[:, index] = scatter_add(h_hits, labels.to(h_hits.device))
    return intersection_matrix, intersection_matrix_w


def obtain_union_matrix(shower_p_unique, particle_ids, labels, dic):
    len_pred_showers = len(shower_p_unique)
    union_matrix = torch.zeros((len_pred_showers, len(particle_ids)))

    for index, id in enumerate(particle_ids):
        counts = torch.zeros_like(labels)
        mask_p = dic["graph"].ndata["particle_number"] == id
        for index_pred, id_pred in enumerate(shower_p_unique):
            mask_pred_p = labels == id_pred
            mask_union = mask_pred_p + mask_p
            union_matrix[index_pred, index] = torch.sum(mask_union)

    return union_matrix


def get_clustering(betas: torch.Tensor, X: torch.Tensor, tbeta=0.7, td=0.2):
    """
    Returns a clustering of hits -> cluster_index, based on the GravNet model
    output (predicted betas and cluster space coordinates) and the clustering
    parameters tbeta and td.
    Takes torch.Tensors as input.
    """
    n_points = betas.size(0)
    select_condpoints = betas > tbeta
    # Get indices passing the threshold
    indices_condpoints = select_condpoints.nonzero()
    # Order them by decreasing beta value
    indices_condpoints = indices_condpoints[(-betas[select_condpoints]).argsort()]
    # Assign points to condensation points
    # Only assign previously unassigned points (no overwriting)
    # Points unassigned at the end are bkg (-1)
    unassigned = torch.arange(n_points).to(betas.device)
    clustering = -1 * torch.ones(n_points, dtype=torch.long).to(betas.device)
    while len(indices_condpoints) > 0 and len(unassigned) > 0:
        index_condpoint = indices_condpoints[0]
        d = torch.norm(X[unassigned] - X[index_condpoint][0], dim=-1).to(betas.device)
        indices = d < td
        indices = indices.cpu()
        assigned_to_this_condpoint = unassigned[indices]
        clustering[assigned_to_this_condpoint] = index_condpoint[0]
        unassigned = unassigned[~(indices)]
        # calculate indices_codpoints again
        indices_condpoints = find_condpoints(betas, unassigned, tbeta)
    return clustering


def find_condpoints(betas, unassigned, tbeta):
    n_points = betas.size(0)
    select_condpoints = betas > tbeta
    device = betas.device
    mask_unassigned = torch.zeros(n_points).to(device)
    mask_unassigned[unassigned] = True
    select_condpoints = mask_unassigned.to(bool) * select_condpoints
    # Get indices passing the threshold
    indices_condpoints = select_condpoints.nonzero()
    # Order them by decreasing beta value
    indices_condpoints = indices_condpoints[(-betas[select_condpoints]).argsort()]
    return indices_condpoints


def obtain_intersection_values(intersection_matrix_w, row_ind, col_ind, dic):
    list_intersection_E = []
    # intersection_matrix_w = intersection_matrix_w
    particle_ids = torch.unique(dic["graph"].ndata["particle_number"])
    if torch.sum(particle_ids == 0) > 0:
        # removing also the MC particle corresponding to noise
        intersection_matrix_wt = torch.transpose(intersection_matrix_w[1:, 1:], 1, 0)
        row_ind = row_ind - 1
    else:
        intersection_matrix_wt = torch.transpose(intersection_matrix_w[1:, :], 1, 0)
    for i in range(0, len(col_ind)):
        list_intersection_E.append(
            intersection_matrix_wt[row_ind[i], col_ind[i]].view(-1)
        )
    if len(list_intersection_E) > 0:
        return torch.cat(list_intersection_E, dim=0)
    else:
        return 0


def plot_iou_matrix(iou_matrix, image_path, hdbscan=False):
    iou_matrix = torch.transpose(iou_matrix[1:, :], 1, 0)
    fig, ax = plt.subplots()
    iou_matrix = iou_matrix.detach().cpu().numpy()
    ax.matshow(iou_matrix, cmap=plt.cm.Blues)
    for i in range(0, iou_matrix.shape[1]):
        for j in range(0, iou_matrix.shape[0]):
            c = np.round(iou_matrix[j, i], 1)
            ax.text(i, j, str(c), va="center", ha="center")
    fig.savefig(image_path, bbox_inches="tight")
    if hdbscan:
        wandb.log({"iou_matrix_hdbscan": wandb.Image(image_path)})
    else:
        wandb.log({"iou_matrix": wandb.Image(image_path)})


def match_showers(
    labels,
    dic,
    particle_ids,
    model_output,
    local_rank,
    i,
    path_save,
    pandora=False,
    tracks=False,
    hdbscan=False,
):
    iou_threshold = 0.25
    shower_p_unique = torch.unique(labels)
    if torch.sum(labels == 0) == 0:
        shower_p_unique = torch.cat(
            (
                torch.Tensor([0]).to(shower_p_unique.device).view(-1),
                shower_p_unique.view(-1),
            ),
            dim=0,
        )
    e_hits = dic["graph"].ndata["e_hits"].view(-1)
    i_m, i_m_w = obtain_intersection_matrix(
        shower_p_unique, particle_ids, labels, dic, e_hits
    )
    i_m = i_m.to(model_output.device)
    i_m_w = i_m_w.to(model_output.device)
    u_m = obtain_union_matrix(shower_p_unique, particle_ids, labels, dic)
    u_m = u_m.to(model_output.device)
    iou_matrix = i_m / u_m
    if torch.sum(particle_ids == 0) > 0:
        # removing also the MC particle corresponding to noise
        iou_matrix_num = (
            torch.transpose(iou_matrix[1:, 1:], 1, 0).clone().detach().cpu().numpy()
        )
    else:
        iou_matrix_num = (
            torch.transpose(iou_matrix[1:, :], 1, 0).clone().detach().cpu().numpy()
        )
    iou_matrix_num[iou_matrix_num < iou_threshold] = 0
    row_ind, col_ind = linear_sum_assignment(-iou_matrix_num)
    # Next three lines remove solutions where there is a shower that is not associated and iou it's zero (or less than threshold)
    mask_matching_matrix = iou_matrix_num[row_ind, col_ind] > 0
    row_ind = row_ind[mask_matching_matrix]
    col_ind = col_ind[mask_matching_matrix]
    if torch.sum(particle_ids == 0) > 0:
        row_ind = row_ind + 1
    if i == 0 and local_rank == 0:
        if path_save is not None:
            if pandora:
                image_path = path_save + "/example_1_clustering_pandora.png"
            else:
                image_path = path_save + "/example_1_clustering.png"
            # plot_iou_matrix(iou_matrix, image_path, hdbscan)
    # row_ind are particles that are matched and col_ind the ind of preds they are matched to
    return shower_p_unique, row_ind, col_ind, i_m_w, iou_matrix


def clustering_obtain_labels(X, betas, device,  tbeta=0.7, td=0.2):
    clustering = get_clustering(betas, X,  tbeta=tbeta, td=td)
    map_from = list(np.unique(clustering.detach().cpu()))
    cluster_id = map(lambda x: map_from.index(x), clustering.detach().cpu())
    clustering_ordered = torch.Tensor(list(cluster_id)).long()
    if torch.unique(clustering)[0] != -1:
        clustering = clustering_ordered + 1
    else:
        clustering = clustering_ordered
    clustering = clustering.view(-1).long().to(device)
    return clustering


def hfdb_obtain_labels(X, device, eps=0.05):
    hdb = hdbscan.HDBSCAN(min_cluster_size=8, min_samples=8, cluster_selection_epsilon=eps).fit(
        X.detach().cpu()
    )
    labels_hdb = hdb.labels_ + 1
    labels_hdb = np.reshape(labels_hdb, (-1))
    labels_hdb = torch.Tensor(labels_hdb).long().to(device)
    return labels_hdb


def dbscan_obtain_labels(X, device):
    distance_scale = (
        (torch.min(torch.abs(torch.min(X, dim=0)[0] - torch.max(X, dim=0)[0])) / 30)
        .view(-1)
        .detach()
        .cpu()
        .numpy()[0]
    )

    db = DBSCAN(eps=distance_scale, min_samples=15).fit(X.detach().cpu())
    # DBSCAN has clustering labels -1,0,.., our cluster 0 is noise so we add 1
    labels = db.labels_ + 1
    labels = np.reshape(labels, (-1))
    labels = torch.Tensor(labels).long().to(device)
    return labels


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


def get_labels_pandora(tracks, dic, device):
    if tracks:
        labels_pandora = dic["graph"].ndata["pandora_pfo"].long()
    else:
        labels_pandora = dic["graph"].ndata["pandora_cluster"].long()
    labels_pandora = labels_pandora + 1
    map_from = list(np.unique(labels_pandora.detach().cpu()))
    map_from = CachedIndexList(map_from)
    cluster_id = map(lambda x: map_from.index(x), labels_pandora.detach().cpu().numpy())
    labels_pandora = torch.Tensor(list(cluster_id)).long().to(device)
    return labels_pandora

def distance_to_true_cluster_of_track(dic, labels):
    # For each cluster, get distance from the track to the cluster of the true MC particle that the track otherwise belongs to.
    # Also returns the number of tracks in the MC cluster that the track belongs to.
    g = dic["graph"]
    mask_hit_type_t2 = g.ndata["hit_type"] == 1
    if torch.sum(labels.unique()==0) == 0:
        distances = torch.zeros(len(labels.unique())+1).float().to(labels.device)
        number_of_tracks = torch.zeros(len(labels.unique())+1).int()
    else:
        distances = torch.zeros(len(labels.unique())).float().to(labels.device)
        number_of_tracks = torch.zeros(len(labels.unique())).int()
    # labels should be a list of labels for each particle
    for i, label in enumerate(labels.unique()):
        mask_labels_i = labels == label
        mask  = mask_labels_i * mask_hit_type_t2
        if mask.sum() == 0:
            continue
        pos_track = g.ndata["pos_hits_xyz"][mask][0]
        if pos_track.shape[0] == 0:
            continue
        true_part_idx_track = g.ndata["particle_number"][mask_labels_i * mask_hit_type_t2][0].int()
        mask_labels_i_true = g.ndata["particle_number"] == true_part_idx_track
        mean_pos_cluster_true = torch.mean(
            g.ndata["pos_hits_xyz"][mask_labels_i_true], dim=0
        )
        number_of_tracks[label] = torch.sum(mask_labels_i_true * mask_hit_type_t2)
        distances[label] = torch.norm(mean_pos_cluster_true - pos_track) / 3300
    return distances, number_of_tracks

def distance_to_cluster_track(dic, is_track_in_MC):

    g = dic["graph"]
    mask_hit_type_t1 = g.ndata["hit_type"]==2
    mask_hit_type_t2 = g.ndata["hit_type"]==1
    pos_track = g.ndata["pos_hits_xyz"][mask_hit_type_t2]
    particle_track = g.ndata["particle_number"][mask_hit_type_t2]
    if len(particle_track)>0:
        mean_pos_cluster_all = []
        for i in particle_track:
            if i ==0:
                mean_pos_cluster_all.append(torch.zeros((1,3)).view(-1,3).to(particle_track.device))
            else:
                mask_labels_i = g.ndata["particle_number"] ==i
                mean_pos_cluster = torch.mean(g.ndata["pos_hits_xyz"][mask_labels_i*mask_hit_type_t1], dim=0)
                mean_pos_cluster_all.append(mean_pos_cluster.view(-1,3))
        mean_pos_cluster_all = torch.cat(mean_pos_cluster_all, dim=0)
        # if  torch.sum(g.ndata["particle_number"] == 0)==0:
        #     #then index 1 is at 0
        #     mean_pos_cluster = mean_pos_cluster[1:,:]
        #     particle_track = particle_track-1
        # if mean_pos_cluster.shape[0]> torch.max(particle_track):
        #     distance_track_cluster = torch.norm(mean_pos_cluster[particle_track.long()]-pos_track,dim=1)/1000
        distance_track_cluster = torch.norm(mean_pos_cluster_all-pos_track,dim=1)/1000
        if len(particle_track)>len(torch.unique(particle_track)):
            distance_track_cluster_unique =[]
            for i in torch.unique(particle_track):
                mask_tracks = particle_track == i
                distance_track_cluster_unique.append(torch.min(distance_track_cluster[mask_tracks]).view(-1))
            distance_track_cluster_unique= torch.cat(distance_track_cluster_unique, dim=0)
            unique_particle_track = torch.unique(particle_track)
        else:
            distance_track_cluster_unique = distance_track_cluster
            unique_particle_track = particle_track
        distance_to_cluster_all = is_track_in_MC.clone().float()
        distance_to_cluster_all[unique_particle_track.long()] = distance_track_cluster_unique
        return distance_to_cluster_all
    else:
        return is_track_in_MC.clone().float()