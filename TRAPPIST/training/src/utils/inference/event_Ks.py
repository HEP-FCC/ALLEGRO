import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.rc("font", size=35)
import matplotlib.pyplot as plt
import torch
from torch_scatter import scatter_sum, scatter_mean


def calculate_event_energy_resolution(df, pandora=False, full_vector=False):
    true_e = torch.Tensor(df.true_showers_E.values)
    mask_nan_true = np.isnan(df.true_showers_E.values)
    true_e[mask_nan_true] = 0
    batch_idx = df.number_batch
    if pandora:
        pred_E = df.pandora_calibrated_pfo.values
        nan_mask = np.isnan(df.pandora_calibrated_pfo.values)
        pred_E[nan_mask] = 0
        pred_e1 = torch.tensor(pred_E).unsqueeze(1).repeat(1, 3)
        pred_vect = torch.tensor(np.array(df.pandora_calibrated_pos.values.tolist()))
        pred_vect[nan_mask] = 0
        true_vect = torch.tensor(np.array(df.true_pos.values.tolist()))
        true_vect[mask_nan_true] = 0
    else:
        pred_E = df.calibrated_E.values
        nan_mask = np.isnan(df.calibrated_E.values)
        # print(np.sum(nan_mask))
        pred_E[nan_mask] = 0
        pred_e1 = torch.tensor(pred_E).unsqueeze(1).repeat(1, 3)
        pred_vect = torch.tensor(np.array(df.pred_pos_matched.values.tolist()))
        pred_vect[nan_mask] = 0
        true_vect = torch.tensor(np.array(df.true_pos.values.tolist()))
        true_vect[mask_nan_true] = 0

    batch_idx = torch.tensor(batch_idx.values).long()
    pred_E = torch.tensor(pred_E)

    true_jet_vect = scatter_sum(true_vect, batch_idx, dim=0)
    pred_jet_vect = scatter_sum(pred_vect, batch_idx, dim=0)
    true_E_jet = scatter_sum(torch.tensor(true_e), batch_idx)
    pred_E_jet = scatter_sum(torch.tensor(pred_E), batch_idx)
    true_jet_p = torch.norm(
        true_jet_vect, dim=1
    )  # This is actually momentum resolution
    pred_jet_p = torch.norm(pred_jet_vect, dim=1)

    mass_true = torch.sqrt(torch.abs(true_E_jet**2 - true_jet_p**2))
    mass_pred = torch.sqrt(torch.abs(pred_E_jet**2 - pred_jet_p**2))

    mass_over_true = mass_pred / mass_true

    return mass_over_true


def get_response_for_event_energy(matched_pandora, matched_):
    mass_over_true_pandora = calculate_event_energy_resolution(
        matched_pandora, True, True
    )
    decay_type = get_decay_type(matched_pandora)
    mass_over_true_model = calculate_event_energy_resolution(matched_, False, True)
    dic = {}
    dic["mass_over_true_model"] = mass_over_true_model
    dic["mass_over_true_pandora"] = mass_over_true_pandora
    dic["decay_type"] = decay_type
    return dic


def get_decay_type(sd_hgb1):
    batch_number = sd_hgb1.number_batch.values
    decay_type_list = []
    for batch_id in range(0, int(np.max(batch_number)) + 1):
        decay_type = determine_decay_type(sd_hgb1, batch_id)
        decay_type_list.append(decay_type)
    return torch.cat(decay_type_list)


def determine_decay_type(sd_hgb1, i):
    pid_values = np.abs(sd_hgb1[sd_hgb1.number_batch == i].pid.values)
    if len(pid_values) == 2:
        decay_type = 0
        charged = np.prod(pid_values == [211.0, 211])
    elif len(pid_values) == 4 and np.count_nonzero(pid_values == 22.0) == 4:
        decay_type = 1
        neutral = np.prod(pid_values == [22.0, 22.0, 22.0, 22.0])
    else:
        decay_type = 2
    return torch.Tensor([decay_type])


def plot_mass_resolution(event_res_dic, PATH_store):
    mask_decay_charged = event_res_dic["decay_type"] == 0
    fig, ax = plt.subplots()
    ax.set_xlabel("M_pred/M_true")
    ax.hist(
        event_res_dic["mass_over_true_model"][mask_decay_charged],
        bins=100,
        histtype="step",
        label="ML",
        color="red",
        density=True,
    )

    ax.hist(
        event_res_dic["mass_over_true_pandora"][mask_decay_charged],
        bins=100,
        histtype="step",
        label="Pandora",
        color="blue",
        density=True,
    )
    ax.grid()
    ax.legend()
    ax.set_xlim([0, 10])
    fig.tight_layout()
    fig.savefig(PATH_store + "mass_resolution_charged.pdf", bbox_inches="tight")

    mask_decay_neutral = event_res_dic["decay_type"] == 1
    fig, ax = plt.subplots()
    ax.set_xlabel("M_pred/M_true")

    ax.hist(
        event_res_dic["mass_over_true_model"][mask_decay_neutral],
        bins=100,
        histtype="step",
        label="ML",
        color="red",
        density=True,
    )

    ax.hist(
        event_res_dic["mass_over_true_pandora"][mask_decay_neutral],
        bins=100,
        histtype="step",
        label="Pandora",
        color="blue",
        density=True,
    )
    ax.grid()
    ax.legend()
    ax.set_xlim([0, 10])
    fig.tight_layout()

    fig.savefig(PATH_store + "mass_resolution_neutral.pdf", bbox_inches="tight")


def mass_Ks(matched_pandora, matched_, PATH_store):
    event_res_dic = get_response_for_event_energy(matched_pandora, matched_)
    plot_mass_resolution(event_res_dic, PATH_store)
