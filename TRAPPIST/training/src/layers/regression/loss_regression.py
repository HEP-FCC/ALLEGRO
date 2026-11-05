
import torch 
import torch.nn as nn
import numpy as np 




def loss_score_func(e_cor):
    score_object = e_cor["score_object"]
    idx_fakes = e_cor["fakes_idx"]
    score_true = torch.ones_like(idx_fakes)
    score_true[idx_fakes]=0
    loss_fn = nn.BCEWithLogitsLoss()
    loss_score = loss_fn(score_object, score_true)
    return loss_score

def obtain_PID_charged(dic,pid_true_matched, pids_charged, args, pid_conversion_dict):
    charged_PID_pred = dic["charged_PID_pred"]
    charged_PID_true = np.array(pid_true_matched)[dic["charged_idx"].cpu().tolist()]
    # one-hot encoded
    charged_PID_true_onehot = torch.zeros(
        len(charged_PID_true), len(pids_charged)
    ).to(charged_PID_pred.device)
    mask_charged = torch.ones(len(charged_PID_true))
    if not args.PID_4_class:
        for i in range(len(charged_PID_true)):
            if charged_PID_true[i] in pids_charged:
                charged_PID_true_onehot[i, pids_charged.index(charged_PID_true[i])] = 1
            else:
                charged_PID_true_onehot[i, -1] = 1
    else:
        for i in range(len(charged_PID_true)):
            true_idx = pid_conversion_dict.get(charged_PID_true[i], 3)
            if true_idx not in pids_charged:
                # Nonsense example - don't train on this one
                mask_charged[i] = 0
            else:
                charged_PID_true_onehot[i, pids_charged.index(true_idx)] = 1
            if charged_PID_true[i] not in pid_conversion_dict:
                print("Unknown PID", charged_PID_true[i])
    return charged_PID_pred, charged_PID_true_onehot, mask_charged

def loss_position(part_coords_matched, pred_pos, args, neutral_idx, charged_idx):
    true_pos = torch.tensor(part_coords_matched).to(pred_pos.device)
    if args.regress_unit_p:
        true_pos = (true_pos / torch.norm(true_pos, dim=1).view(-1, 1)).clone()
        pred_pos = (pred_pos / torch.norm(pred_pos, dim=1).view(-1, 1)).clone()
    # loss_pos = torch.nn.L1Loss()(pred_pos, true_pos)
    loss_pos = 1 - ((torch.nn.CosineSimilarity()(pred_pos, true_pos)).mean())
    loss_pos_neutrals = torch.nn.L1Loss()(
        pred_pos[neutral_idx], true_pos[neutral_idx]
    )
    loss_pos_charged = torch.nn.L1Loss()(
        pred_pos[charged_idx], true_pos[charged_idx]
    ) 
    return loss_pos, loss_pos_neutrals, loss_pos_charged







def obtain_PID_neutral(dic,pid_true_matched,pids_neutral, args, pid_conversion_dict):
    neutral_PID_pred = dic["neutral_PID_pred"]
    neutral_idx = dic["neutrals_idx"]
    neutral_PID_true = np.array(pid_true_matched)[neutral_idx.cpu()]
    if type(neutral_PID_true) == np.float64:
        neutral_PID_true = [neutral_PID_true]
    # One-hot encoded
    neutral_PID_true_onehot = torch.zeros(
        len(neutral_PID_true), len(pids_neutral)
    ).to(neutral_PID_pred.device)
    mask_neutral = torch.ones(len(neutral_PID_true))

    # convert from true PID to int list PID
    if not args.PID_4_class:
        for i in range(len(neutral_PID_true)):
            if neutral_PID_true[i] in pids_neutral:
                neutral_PID_true_onehot[i, pids_neutral.index(neutral_PID_true[i])] = 1
            else:
                neutral_PID_true_onehot[i, -1] = 1
    else:
        for i in range(len(neutral_PID_true)):
            true_idx = pid_conversion_dict.get(neutral_PID_true[i], 3)
            if true_idx not in pids_neutral:
                mask_neutral[i] = 0
            else:
                neutral_PID_true_onehot[i, pids_neutral.index(true_idx)] = 1
            if neutral_PID_true[i] not in pid_conversion_dict:
                print("Unknown PID", neutral_PID_true[i])
    neutral_PID_true_onehot = neutral_PID_true_onehot.to(neutral_idx.device)
    return neutral_PID_pred, neutral_PID_true_onehot, mask_neutral
    


def save_features_func(args, graph_level_features, e_true, e_cor,e_sum_hits, e_true_corr_daughters, part_coords_matched, pid_true_matched):
    cluster_features_path = os.path.join(
        args.model_prefix, "cluster_features"
    )
    if not os.path.exists(cluster_features_path):
        os.makedirs(cluster_features_path)
    save_features(
        cluster_features_path,
        {
            "x": graph_level_features.detach().cpu(),
            # """ "xyz_covariance_matrix": covariances.cpu(),"""
            "e_true": e_true.detach().cpu(),
            "e_reco": e_cor.detach().cpu(),
            "true_e_corr": (e_true / e_sum_hits - 1).detach().cpu(),
            "e_true_corrected_daughters": e_true_corr_daughters.detach().cpu(),
            # "node_features_avg": scatter_mean(
            #    batch_g.ndata["h"], batch_idx, dim=0
            # ),  # graph-averaged node features
            "coords_y": part_coords_matched,
            "pid_y": pid_true_matched,
        },
    )

