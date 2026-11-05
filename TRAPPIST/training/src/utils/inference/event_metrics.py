import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib

matplotlib.rc("font", size=35)
import matplotlib.pyplot as plt
import torch
from src.utils.inference.inference_metrics import get_sigma_gaussian
from torch_scatter import scatter_sum, scatter_mean
import os
from src.utils.pid_conversion import our_to_pandora_mapping, pandora_to_our_mapping


def plot_per_event_metrics(sd, sd_pandora, PATH_store=None):
    (
        calibrated_list,
        calibrated_list_pandora,
        reco_list,
        reco_list_pandora,
    ) = calculate_energy_per_event(sd, sd_pandora)
    plot_per_event_energy_distribution(
        calibrated_list,
        calibrated_list_pandora,
        reco_list,
        reco_list_pandora,
        PATH_store,
    )


def calculate_energy_per_event(
    sd,
    sd_pandora,
):
    sd = sd.reset_index(drop=True)
    sd_pandora = sd_pandora.reset_index(drop=True)
    corrected_list = []
    reco_list = []
    reco_list_pandora = []
    corrected_list_pandora = []
    for i in range(0, int(np.max(sd.number_batch))):
        mask = sd.number_batch == i
        event_E_total_reco = np.nansum(sd.reco_showers_E[mask])
        event_E_total_true = np.nansum(sd.true_showers_E[mask])
        event_E_total_reco_corrected = np.nansum(sd.calibrated_E[mask])
        event_ML_total_reco = np.nansum(sd.pred_showers_E[mask])
        mask_p = sd_pandora.number_batch == i
        event_E_total_reco_p = np.nansum(sd_pandora.reco_showers_E[mask_p])
        event_E_total_true_p = np.nansum(sd_pandora.true_showers_E[mask_p])
        event_ML_total_reco_p = np.nansum(sd_pandora.pred_showers_E[mask_p])
        event_ML_total_reco_p_corrected = np.nansum(
            sd_pandora.pandora_calibrated_pfo[mask_p]
        )

        reco_list.append(event_ML_total_reco / event_E_total_reco)
        corrected_list.append(event_E_total_reco_corrected / event_E_total_true)
        reco_list_pandora.append(event_ML_total_reco_p / event_E_total_reco_p)
        corrected_list_pandora.append(
            event_ML_total_reco_p_corrected / event_E_total_true_p
        )
    return corrected_list, corrected_list_pandora, reco_list, reco_list_pandora


def plot_per_event_energy_distribution(
    calibrated_list, calibrated_list_pandora, reco_list, reco_list_pandora, PATH_store
):
    fig = plt.figure(figsize=(8, 8))
    sns.histplot(
        data=np.array(calibrated_list),  # + 1 - np.mean(calibrated_list)
        stat="percent",
        binwidth=0.01,
        label="MLPF",
        # element="step",
        # fill=False,
        color="red",
        # linewidth=2,
    )
    sns.histplot(
        data=calibrated_list_pandora,
        stat="percent",
        color="blue",
        binwidth=0.01,
        label="Pandora",
        # element="step",
        # fill=False,
        # linewidth=2,
    )
    plt.ylabel("Percent of events")
    plt.xlabel("$E_{corrected}/E_{total}$")
    # plt.yscale("log")
    plt.legend()
    plt.xlim([0, 2])
    fig.savefig(
        PATH_store + "per_event_E.png",
        bbox_inches="tight",
    )
    fig = plt.figure(figsize=(8, 8))
    sns.histplot(data=reco_list, stat="percent", binwidth=0.01, label="MLPF")
    sns.histplot(
        data=reco_list_pandora,
        stat="percent",
        color="orange",
        binwidth=0.01,
        label="Pandora",
    )
    plt.ylabel("Percent of events")
    plt.xlabel("$E_{recoML}/E_{reco}$")
    plt.legend()
    plt.xlim([0.5, 1.5])
    # plt.yscale("log")
    fig.savefig(
        PATH_store + "per_event_E_reco.png",
        bbox_inches="tight",
    )

particle_masses = {0: 0, 22: 0, 11: 0.00511, 211: 0.13957, 130: 0.493677, 2212: 0.938272, 2112: 0.939565}
particle_masses_4_class = {0: 0.00511, 1: 0.13957, 2: 0.939565, 3: 0.0, 4: 0.10566} # electron, CH, NH, photon, muon

def safeint(x, default_val=0):
    if np.isnan(x):
        return default_val
    return int(x)



def calculate_event_mass_resolution(df, pandora, perfect_pid=False, mass_zero=False, ML_pid=False, fake=False):
    # reco showers> 0 does not consider showers that are in the event but do not contribute energy to the total in the event
    #df = df[((df.reco_showers_E>0) | (pd.isna(df.pid)))]
    dic = {}
    df = df[df.reco_showers_E != 0.0]
    true_e = torch.Tensor(df.true_showers_E.values)
    reco_true_e = torch.Tensor(df.reco_showers_E.values)
    mask_nan_true = np.isnan(df.true_showers_E.values)
    true_e[mask_nan_true] = 0
    reco_true_e[mask_nan_true] = 0
    batch_idx = df.number_batch
    if pandora:
        pred_E = df.pandora_calibrated_pfo.values
        pred_E_reco = df.pred_showers_E.values
        nan_mask = np.isnan(df.pandora_calibrated_pfo.values)
        pred_E_reco[nan_mask] = 0
        pred_E[nan_mask] = 0

        pred_vect = torch.tensor(np.array(df.pandora_calibrated_pos.values.tolist()))
        nan_mask_p = torch.isnan(pred_vect).any(dim=1)
        pred_vect[nan_mask_p] = 0
    else:
        pred_E = df.calibrated_E.values
        pred_E_reco = df.pred_showers_E.values
        nan_mask = np.isnan(df.calibrated_E.values)
        pred_E[nan_mask] = 0
        pred_E_reco[nan_mask] = 0
        pred_vect = torch.tensor(
            np.array(df.pred_pos_matched.values.tolist())
        )
        pred_vect[nan_mask] = 0
    true_vect = torch.tensor(np.array(df.true_pos.values.tolist()))
    true_vect[mask_nan_true] = 0
    if fake:
        p_true_norm = true_vect/ np.linalg.norm(true_vect, axis=1).reshape(-1, 1)
        m = np.array([particle_masses.get(abs(safeint(i)), 0) for i in df.pid])
        p_squared_true = (true_e ** 2 - m ** 2)
        true_vect = np.sqrt(p_squared_true).reshape(-1, 1) * np.array(p_true_norm)
    if perfect_pid or mass_zero or ML_pid:
        if len(pred_vect) > 0:
            pred_vect /= np.linalg.norm(pred_vect, axis=1).reshape(-1, 1)
            pred_vect[torch.isnan(pred_vect)] = 0
        if ML_pid:
            if pandora:
                m = np.array([particle_masses.get(abs(safeint(i)), 0) for i in df.pandora_pid.values])
            else:
                m = np.array([particle_masses_4_class.get(safeint(i), 0) for i in df.pred_pid_matched.values])
        else:
            m = np.array([particle_masses.get(abs(safeint(i)), 0) for i in df.pid])
        if mass_zero:
            m = np.array([0 for _ in m])
        p_squared = (pred_E ** 2 - m ** 2)
        p_squared[p_squared < 0] = 0 # they are always like of order -1e-8
        pred_vect = np.sqrt(p_squared).reshape(-1, 1) * np.array(pred_vect)
    batch_idx = torch.tensor(batch_idx.values).long()
    pred_E = torch.tensor(pred_E)
    pred_E_reco = torch.tensor(pred_E_reco)
    true_jet_vect = scatter_sum(true_vect, batch_idx, dim=0)
    pred_jet_vect = scatter_sum(torch.tensor(pred_vect), batch_idx, dim=0)
    true_E_jet = scatter_sum(torch.tensor(true_e), batch_idx)
    true_E_jet_reco = scatter_sum(torch.tensor(reco_true_e), batch_idx)
    pred_E_jet = scatter_sum(torch.tensor(pred_E), batch_idx)
    pred_E_jet_reco = scatter_sum(torch.tensor(pred_E_reco), batch_idx)
    true_jet_p = torch.norm(true_jet_vect, dim=1)  # This is actually momentum resolution
    pred_jet_p = torch.norm(pred_jet_vect, dim=1)
    mass_true = torch.sqrt((true_E_jet ** 2).abs() - true_jet_p ** 2)
    mass_pred_p = torch.sqrt(torch.abs(pred_E_jet ** 2) - pred_jet_p ** 2)
    # replace nans in these with 0
    dic["mass_over_true_p"] = mass_pred_p / mass_true
    dic["E_over_true"] = pred_E_jet / true_E_jet
    dic["E_over_true_reco"] = pred_E_jet_reco/true_E_jet_reco
    dic["p_over_true"] = pred_jet_p / true_jet_p
    p_jet_pandora = pred_jet_p
    (
        mean_mass,
        var_mass,
        _,
        _,
    ) = get_sigma_gaussian(dic["mass_over_true_p"], np.linspace(0, 2, 400), epsilon=0.005)
    dic["mean_mass"] = mean_mass
    dic["var_mass"] = var_mass
    dic["mass_true"] = mass_true
    dic["true_jet_p"]=true_jet_p
    dic["mass_pred_p"] = mass_pred_p
    dic["E_pred"] = pred_E_jet
    dic["E_true"] = true_E_jet
    return dic


def calculate_event_energy_resolution(df, pandora=False, full_vector=False):
    if full_vector and pandora:
        assert "pandora_calibrated_pos" in df.columns
    bins = [0, 700]
    binsx = []
    mean = []
    variance = []
    distributions = []
    distr_baseline = []
    mean_baseline = []
    variance_baseline = []
    mass_list = []
    binning = 1e-2
    bins_per_binned_E = np.arange(0, 2, binning)
    for i in range(len(bins) - 1):
        bin_i = bins[i]
        bin_i1 = bins[i + 1]
        binsx.append(0.5 * (bin_i + bin_i1))
        true_e = df.true_showers_E.values
        batch_idx = df.number_batch
        if pandora:
            pred_e = df.pandora_calibrated_pfo.values
            pred_e1 = torch.tensor(pred_e).unsqueeze(1).repeat(1, 3)
            if full_vector:
                pred_vect = (
                    np.array(df.pandora_calibrated_pos.values.tolist())
                    # * pred_e1.numpy()
                )
                true_vect = (
                    np.array(df.true_pos.values.tolist())
                    # * torch.tensor(true_e).unsqueeze(1).repeat(1, 3).numpy()
                )
                pred_vect = torch.tensor(pred_vect)
                true_vect = torch.tensor(true_vect)
        else:
            pred_e = df.calibrated_E.values
            pred_e1 = torch.tensor(pred_e).unsqueeze(1).repeat(1, 3)
            if full_vector:
                pred_vect = (
                    np.array(df.pred_pos_matched.values.tolist()) * pred_e1.numpy()
                )
                true_vect = (
                    np.array(df.true_pos.values.tolist())
                    # * torch.tensor(true_e).unsqueeze(1).repeat(1, 3).numpy()
                )
                pred_vect = torch.tensor(pred_vect)
                true_vect = torch.tensor(true_vect)
        true_rec = df.reco_showers_E
        # pred_e_nocor = df.pred_showers_E[mask]
        true_e = torch.tensor(true_e)
        batch_idx = torch.tensor(batch_idx.values).long()
        pred_e = torch.tensor(pred_e)
        true_rec = torch.tensor(true_rec.values)
        if full_vector:
            true_p_vect = scatter_sum(true_vect, batch_idx, dim=0)
            pred_p_vect = scatter_sum(pred_vect, batch_idx, dim=0)
            true_e1 = scatter_sum(torch.tensor(true_e), batch_idx)
            pred_e1 = scatter_sum(torch.tensor(pred_e), batch_idx)
            true_e = torch.norm(
                true_p_vect, dim=1
            )  # This is actually momentum resolution
            pred_e = torch.norm(pred_p_vect, dim=1)
        else:
            true_e = scatter_sum(true_e, batch_idx)
            pred_e = scatter_sum(pred_e, batch_idx)
        true_rec = scatter_sum(true_rec, batch_idx)
        mask_above = true_e <= bin_i1
        mask_below = true_e > bin_i
        mask_check = true_e > 0
        mask = mask_below * mask_above * mask_check
        true_e = true_e[mask]
        true_rec = true_rec[mask]
        pred_e = pred_e[mask]
        if torch.sum(mask) > 0:  # if the bin is not empty
            e_over_true = pred_e / true_e
            e_over_reco = true_rec / true_e
            distributions.append(e_over_true)
            distr_baseline.append(e_over_reco)
            (
                mean_predtotrue,
                var_predtotrue,
                err_mean_predtotrue,
                err_var_predtotrue,
            ) = get_sigma_gaussian(e_over_true, bins_per_binned_E)
            if full_vector:
                mass_true = torch.sqrt(true_e1[mask] ** 2 - true_e**2)
                mass_pred = torch.sqrt(pred_e1[mask] ** 2 - pred_e**2)
                print(pandora, len(mass_true), len(mass_pred))
                mass_over_true = mass_pred / mass_true

                (
                    mean_mass,
                    var_mass,
                    _,
                    _,
                ) = get_sigma_gaussian(mass_over_true, bins_per_binned_E)
                mass_list.append(mass_over_true)
            (
                mean_reco_true,
                var_reco_true,
                err_mean_reco_true,
                err_var_reco_true,
            ) = get_sigma_gaussian(e_over_reco, bins_per_binned_E)
            mean.append(mean_predtotrue)
            variance.append(np.abs(var_predtotrue))
            mean_baseline.append(mean_reco_true)
            variance_baseline.append(np.abs(var_reco_true))
    if full_vector:
        mass_list = torch.cat(mass_list)
    ret = [
        mean,
        variance,
        distributions,
        binsx,
        mean_baseline,
        variance_baseline,
        distr_baseline,
    ]
    if full_vector:
        ret += [mass_list]
    else:
        ret += [None]
    return ret

def get_mass_contribution_per_category(matched_pandora, matched_, matched_gt, perfect_pid=False, mass_zero=False, ML_pid=True, energy_bins=None):
    # PID_categories: Report in terms of categories e, CH, NH, gamma
    mass_over_true = {}
    mass_over_true_gt = {}
    mass_over_true_pandora = {}
    pid_model_over_pred = {}
    pid_pandora_over_pred = {}
    pid_model_over_true = {}
    pid_pandora_over_true = {}
    pid_true_over_true = {}
    E_over_true = {}
    E_over_true_pandora = {}
    pid_groups = [[0,1],[2,3], [0,1]]
    for pid_indx, pid_group in enumerate(pid_groups):
        filt_pandora_pred = (matched_pandora.pid_4_class_true == pid_group[0])+(matched_pandora.pid_4_class_true == pid_group[1]) #matched_pandora.pid.isin(our_to_pandora_mapping[pid_group[0]])+ matched_pandora.pid.isin(our_to_pandora_mapping[pid_group[1]])
        filt_model_pred = (matched_.pid_4_class_true == pid_group[0])+(matched_.pid_4_class_true == pid_group[1])
        filt_model_pred_GT = (matched_gt.pid_4_class_true == pid_group[0])+(matched_gt.pid_4_class_true == pid_group[1])
        filt_model_true = ((matched_.pid_4_class_true == pid_group[0]))+(matched_.pid_4_class_true == pid_group[1])

        if pid_indx==0:
            filt_model_pred = filt_model_pred*(~np.isnan(matched_.pred_showers_E))*((matched_.pred_pid_matched == pid_group[0])+(matched_.pred_pid_matched == pid_group[1]))
            filt_pandora_pred = filt_pandora_pred*(~np.isnan(matched_pandora.pred_showers_E))*(matched_pandora.pandora_pid.isin(our_to_pandora_mapping[pid_group[0]])+ matched_pandora.pandora_pid.isin(our_to_pandora_mapping[pid_group[1]]))
            filt_model_true = filt_model_true*(matched_.is_track_in_MC ==1)
            filt_model_pred_GT =filt_model_pred_GT*(matched_gt.is_track_in_MC ==1)
            filt_model_pred = filt_model_pred*(matched_.is_track_in_MC ==1)
            filt_pandora_pred = filt_pandora_pred*(matched_pandora.is_track_in_MC ==1)
        if pid_indx==2:
            filt_model_pred = filt_model_pred*(~np.isnan(matched_.pred_showers_E))*((matched_.pred_pid_matched == pid_groups[1][0])+(matched_.pred_pid_matched == pid_groups[1][1]))
            filt_pandora_pred = filt_pandora_pred*(~np.isnan(matched_pandora.pred_showers_E))*(matched_pandora.pandora_pid.isin(our_to_pandora_mapping[pid_groups[1][0]])+ matched_pandora.pandora_pid.isin(our_to_pandora_mapping[pid_groups[1][1]]))
            filt_model_true = filt_model_true*(matched_.is_track_in_MC ==0)
            filt_model_pred_GT =filt_model_pred_GT*(matched_gt.is_track_in_MC ==0)
            filt_model_pred = filt_model_pred*(matched_.is_track_in_MC ==0)
            filt_pandora_pred = filt_pandora_pred*(matched_pandora.is_track_in_MC ==0)
 

        

        # filt_pandora_true = matched_pandora.pid_4_class_true == pid
        # if energy_bins is not None:
        #     filt_pandora_pred = filt_pandora_pred & (matched_pandora.pandora_calibrated_pfo > energy_bins[0]) & (matched_pandora.pandora_calibrated_pfo < energy_bins[1])
        #     filt_model_pred = filt_model_pred & (matched_.calibrated_E > energy_bins[0]) & (matched_.calibrated_E < energy_bins[1])
        #     filt_model_true = filt_model_true & (matched_.true_showers_E > energy_bins[0]) & (matched_.true_showers_E < energy_bins[1])
        #     filt_pandora_true = filt_pandora_true & (matched_pandora.true_showers_E > energy_bins[0]) & (matched_pandora.true_showers_E < energy_bins[1])
            
        dic_pandora= calculate_event_mass_resolution(matched_pandora[filt_pandora_pred],
                                                                        True,
                                                                        perfect_pid=perfect_pid,
                                                                        mass_zero=mass_zero,
                                                                        ML_pid=ML_pid)
       
        if filt_model_pred.sum() > 0:
            dic_pred = calculate_event_mass_resolution(matched_[filt_model_pred],
                                                                           False,
                                                                            perfect_pid=perfect_pid,
                                                                            mass_zero=mass_zero,
                                                                            ML_pid=ML_pid)
        else:
            distr_mass = torch.tensor([])
        
        # the best that we can compare to for charged is perfect clustering (perfect track to cluster link) and perfect PID
        dic_gt = calculate_event_mass_resolution(matched_gt[filt_model_pred_GT],
                                                                                False,
                                                                                    perfect_pid=True,
                                                                                    mass_zero=False,
                                                                                ML_pid=False)


        
        dimsize = int(matched_.number_batch.max() + 1)
        assert dimsize == matched_pandora.number_batch.max() + 1
        E_model = torch.nan_to_num(torch.tensor(matched_.calibrated_E.values))
        E_pandora = torch.nan_to_num(torch.tensor(matched_pandora.pandora_calibrated_pfo.values))
        E_true = torch.nan_to_num(torch.tensor(matched_.true_showers_E.values))
        E_model_PID = torch.nan_to_num(torch.tensor(matched_[filt_model_pred].calibrated_E.values))
        E_pandora_PID = torch.nan_to_num(torch.tensor(matched_pandora[filt_pandora_pred].pandora_calibrated_pfo.values))
        E_PID_true = torch.nan_to_num(torch.tensor(matched_[filt_model_true].true_showers_E.values))
        event_energy_PID_true = scatter_sum(E_PID_true, torch.tensor(matched_[filt_model_true].number_batch.values).long(), dim_size=int(dimsize))
        event_energy_true = scatter_sum(E_true, torch.tensor(matched_.number_batch.values).long(), dim_size=int(dimsize))
        event_energy_pred = scatter_sum(E_model, torch.tensor(matched_.number_batch.values).long(), dim_size=int(dimsize))
        event_energy_pred_pandora = scatter_sum(E_pandora, torch.tensor(matched_pandora.number_batch.values).long(), dim_size=dimsize)
        event_energy_pred_PID = scatter_sum(E_model_PID, torch.tensor(matched_[filt_model_pred].number_batch.values).long(), dim_size=dimsize)
        event_energy_pred_pandora_PID = scatter_sum(E_pandora_PID, torch.tensor(matched_pandora[filt_pandora_pred].number_batch.values).long(), dim_size=dimsize)
        # event_energy_pred_PID / event_energy_pred, event_energy_pred_pandora_PID / event_energy_pred_pandora, event_energy_pred_PID / event_energy_true, event_energy_pred_pandora_PID / event_energy_true
        pid_model_over_pred_result = event_energy_pred_PID / event_energy_pred
        pid_pandora_over_pred_result = event_energy_pred_pandora_PID / event_energy_pred_pandora
        pid_model_over_true_result = event_energy_pred_PID / event_energy_true
        pid_pandora_over_true_result = event_energy_pred_pandora_PID / event_energy_true
        pid = pid_indx
        mass_over_true[pid] = dic_pred["mass_over_true_p"]
        mass_over_true_gt[pid] = dic_gt["mass_over_true_p"]
        mass_over_true_pandora[pid] = dic_pandora["mass_over_true_p"]
        pid_model_over_pred[pid] = pid_model_over_pred_result
        pid_pandora_over_pred[pid] = pid_pandora_over_pred_result
        pid_model_over_true[pid] = pid_model_over_true_result
        pid_pandora_over_true[pid] = pid_pandora_over_true_result
        pid_true_over_true[pid] = event_energy_PID_true / event_energy_true
        E_over_true_pandora[pid] = event_energy_pred_pandora_PID / event_energy_PID_true
        E_over_true[pid] = event_energy_pred_PID / event_energy_PID_true
    return mass_over_true, mass_over_true_pandora, E_over_true, E_over_true_pandora, pid_model_over_true, pid_pandora_over_true, pid_true_over_true, mass_over_true_gt

def get_mass_contribution_per_PID(matched_pandora, matched_, perfect_pid=False, mass_zero=False, ML_pid=True):
    # PID_categories: whether to report in terms of categories e, CH, NH, gamma
    # get the mass contributions to event energy and event mass per PID
    mass_over_true = {}
    mass_over_true_pandora = {}
    E_over_true = {}
    E_over_true_pandora = {}
    pid_model_over_pred = {}
    pid_pandora_over_pred = {}
    pid_model_over_true = {}
    pid_pandora_over_true = {}
    pid_true_over_true = {}
    for pid in [11, 130, 2112, 22, 2212, 211]:
        filt_pandora = matched_pandora.pid==pid
        filt_model = matched_.pid==pid
        if filt_pandora.sum() == 0:
            continue
        dic_pandora  = calculate_event_mass_resolution(matched_pandora[filt_pandora],
                                                                                                                        True,
                                                                                                                        perfect_pid=perfect_pid,
                                                                                                                        mass_zero=mass_zero,
                                                                                                                        ML_pid=ML_pid)
        dic_pred = calculate_event_mass_resolution(matched_[filt_model], False,
                                                                                                        perfect_pid=perfect_pid,
                                                                                                        mass_zero=mass_zero,
                                                                                                        ML_pid=ML_pid)
        dimsize = int(matched_.number_batch.max() + 1)
        assert dimsize == matched_pandora.number_batch.max() + 1
        E_model = torch.nan_to_num(torch.tensor(matched_.calibrated_E.values))
        E_pandora = torch.nan_to_num(torch.tensor(matched_pandora.pandora_calibrated_pfo.values))
        E_true = torch.nan_to_num(torch.tensor(matched_.true_showers_E.values))
        E_model_PID = torch.nan_to_num(torch.tensor(matched_[matched_.pid==pid].calibrated_E.values))
        E_pandora_PID = torch.nan_to_num(torch.tensor(matched_pandora[matched_pandora.pid==pid].pandora_calibrated_pfo.values))
        E_PID_true = torch.nan_to_num(torch.tensor(matched_[matched_.pid==pid].true_showers_E.values))
        event_energy_PID_true = scatter_sum(E_PID_true, torch.tensor(matched_[matched_.pid==pid].number_batch.values).long(), dim_size=int(dimsize))
        event_energy_true = scatter_sum(E_true, torch.tensor(matched_.number_batch.values).long(), dim_size=int(dimsize))
        event_energy_pred = scatter_sum(E_model, torch.tensor(matched_.number_batch.values).long(), dim_size=int(dimsize))
        event_energy_pred_pandora = scatter_sum(E_pandora, torch.tensor(matched_pandora.number_batch.values).long(), dim_size=dimsize)
        event_energy_pred_PID = scatter_sum(E_model_PID, torch.tensor(matched_[matched_.pid==pid].number_batch.values).long(), dim_size=dimsize)
        event_energy_pred_pandora_PID = scatter_sum(E_pandora_PID, torch.tensor(matched_pandora[matched_pandora.pid==pid].number_batch.values).long(), dim_size=dimsize)
        # event_energy_pred_PID / event_energy_pred, event_energy_pred_pandora_PID / event_energy_pred_pandora, event_energy_pred_PID / event_energy_true, event_energy_pred_pandora_PID / event_energy_true
        pid_model_over_pred_result = event_energy_pred_PID / event_energy_pred
        pid_pandora_over_pred_result = event_energy_pred_pandora_PID / event_energy_pred_pandora
        pid_model_over_true_result = event_energy_pred_PID / event_energy_true
        pid_pandora_over_true_result = event_energy_pred_pandora_PID / event_energy_true
        mass_over_true[pid] = dic_pred["mass_over_true_p"]
        mass_over_true_pandora[pid] = dic_pandora["mass_over_true_p"]
        E_over_true[pid] = dic_pred["E_over_true"]
        E_over_true_pandora[pid] = dic_pandora["E_over_true"] 
        pid_model_over_pred[pid] = pid_model_over_pred_result
        pid_pandora_over_pred[pid] = pid_pandora_over_pred_result
        pid_model_over_true[pid] = pid_model_over_true_result
        pid_pandora_over_true[pid] = pid_pandora_over_true_result
        pid_true_over_true[pid] = event_energy_PID_true / event_energy_true
    return mass_over_true, mass_over_true_pandora, E_over_true, E_over_true_pandora, pid_model_over_pred, pid_pandora_over_pred, pid_model_over_true, pid_pandora_over_true, pid_true_over_true

def get_response_for_event_energy(matched_pandora, matched_, perfect_pid=False, mass_zero=False, ML_pid=False):
    (
        mean_p,
        variance_om_p,
        distr_p,
        x_p,
        _,
        _,
        _,
        mass_over_true_pandora,
    ) = calculate_event_energy_resolution(matched_pandora, True, False)
    (
        mean,
        variance_om,
        distr,
        x,
        mean_baseline,
        variance_om_baseline,
        _,
        mass_over_true_model,
    ) = calculate_event_energy_resolution(matched_, False, False)

    dic_pandora = calculate_event_mass_resolution(matched_pandora, True, perfect_pid=perfect_pid, mass_zero=mass_zero, ML_pid=ML_pid, fake=False)
    
    dic_model = calculate_event_mass_resolution(matched_, False, perfect_pid=perfect_pid, mass_zero=mass_zero, ML_pid=ML_pid, fake=False)
    # mean_mass_perfect_PID, var_mass_perfect_PID, distr_mass_perfect_PID, mass_true_perfect_PID, _, _, E_over_true_perfect_PID, E_over_true_reco_perfect_PID = calculate_event_mass_resolution(matched_, False, perfect_pid=True, mass_zero=False, ML_pid=False,  fake=True)
    # matched_.calibrated_E = matched_.pred_showers_E
    df_copy = matched_.copy(deep=True)
    df_copy.true_showers_E = matched_.reco_showers_E
    # matched_pandora.pandora_calibrated_pfo = matched_pandora.pred_showers_E
    # matched_.pred_pos_matched = matched_.true_pos
    dic_perfect_E = calculate_event_mass_resolution(df_copy, False, perfect_pid=False, mass_zero=False, ML_pid=True, fake=False)
    dic_perfect_E_pandora = calculate_event_mass_resolution(matched_pandora, True, perfect_pid=False, mass_zero=False, ML_pid=True, fake=False)
    (
        mean_energy_over_true,
        var_energy_over_true,
        _,
        _,
    ) = get_sigma_gaussian(dic_model["E_over_true"], np.linspace(0, 2, 400), epsilon=0.005)
    (
        mean_energy_over_true_pandora,
        var_energy_over_true_pandora,
        _,
        _,
    ) = get_sigma_gaussian(dic_pandora["E_over_true"], np.linspace(0, 2, 400), epsilon=0.005)
    dic = {}
    dic["mean_p"] = mean_p
    dic["variance_om_p"] = variance_om_p
    dic["variance_om"] = variance_om
    dic["mean"] = mean
    dic["energy_resolutions"] = x
    dic["energy_resolutions_p"] = x_p
    dic["mean_baseline"] = mean_baseline
    dic["variance_om_baseline"] = variance_om_baseline
    dic["distributions_pandora"] = distr_p
    dic["distributions_model"] = distr
    dic["mass_over_true_model"] = dic_model["mass_over_true_p"]
    dic["mass_over_true_model_perfect_E_pandora"] = dic_perfect_E_pandora["mass_over_true_p"]
    dic["mass_over_true_model_perfect_E"] = dic_perfect_E["mass_over_true_p"]
    dic["mass_over_true_pandora"] = dic_pandora["mass_over_true_p"]
    dic["mass_model"] = dic_model["mass_over_true_p"] * dic_model["mass_true"]
    dic["mass_pandora"] = dic_pandora["mass_over_true_p"] * dic_pandora["mass_true"]
    dic["mass_true"] =  dic_pandora["mass_true"]
    dic["mean_mass_model"] = dic_model["mean_mass"]
    dic["mean_mass_pandora"] = dic_pandora["mean_mass"]
    dic["var_mass_model"] =  dic_model["var_mass"]
    dic["var_mass_pandora"] = dic_pandora["var_mass"]
    dic["energy_over_true"] = dic_model["E_over_true"]
    dic["energy_over_true_pandora"] = dic_pandora["E_over_true"]
    dic["mean_energy_over_true"] = mean_energy_over_true
    dic["mean_energy_over_true_pandora"] = mean_energy_over_true_pandora
    dic["var_energy_over_true"] = var_energy_over_true
    dic["var_energy_over_true_pandora"] = var_energy_over_true_pandora
    dic["energy_over_true_reco"] = dic_model["E_over_true_reco"]
    dic["energy_over_true_reco_pandora"] = dic_pandora["E_over_true_reco"]
    return dic

colors = {"ML": "red", "ML GTC": "green"}

def plot_mass_resolution(event_res_dic, PATH_store):
    old_font_size = matplotlib.rcParams['font.size']
    matplotlib.rcParams.update({'font.size': 22})
    pandora_dic = event_res_dic["ML"]
    fig, ax = plt.subplots(2, 2, figsize=(32, 8*2))
    # set fontsize to 20
    ax[0,0].set_xlabel(r"$M_{pred}/M_{true}$")
    bins = np.linspace(0, 2, 100)
    bins_mass = np.linspace(0, 250, 200)
    ax[0,0].hist(
        pandora_dic["mass_over_true_pandora"],
        bins=bins,
        histtype="step",
        label="Pandora $\mu$={}".format(
            round((pandora_dic["mean_mass_pandora"]), 4)
        )+"\n"+"$\sigma/\mu$={}".format(round((pandora_dic["var_mass_pandora"]), 4),
        ),
        color="blue",
        density=True,
        linewidth=1.5
    )
    mean_e_over_true_pandora, sigma_e_over_true_pandora = round(pandora_dic["mean_energy_over_true_pandora"], 4), round(
        pandora_dic["var_energy_over_true_pandora"], 4)
    ax[0,1].hist(pandora_dic["energy_over_true_pandora"], bins=bins, histtype="step",
                # label=r"Pandora $\mu$={} $\sigma / \mu$={}".format(mean_e_over_true_pandora,
                #                                                   sigma_e_over_true_pandora),
                label="Pandora  $\mu$={}".format(
                            mean_e_over_true_pandora
                        )+"\n"+"$\sigma/\mu$={}".format(sigma_e_over_true_pandora
                        ),
                color="blue",
                density=True)
    for key in event_res_dic:
        ax[0,0].hist(
                event_res_dic[key]["mass_over_true_model"],
                bins=bins,
                histtype="step",
                label="ML $\mu$={}".format(
                    round((event_res_dic[key]["mean_mass_model"]), 4)
                )+"\n"+"$\sigma/\mu$={}".format(round((event_res_dic[key]["var_mass_model"]), 4),
                ),
                color=colors[key],
                density=True,
                linewidth=1.5
        )
        mean_e_over_true, sigma_e_over_true = round(event_res_dic[key]["mean_energy_over_true"], 4), round(
            event_res_dic[key]["var_energy_over_true"], 4)
        ax[0,1].hist(event_res_dic[key]["energy_over_true"], bins=bins, 
                histtype="step",
                label=str(key) +" $\mu$={}".format(
                    mean_e_over_true
                )+"\n"+"$\sigma/\mu$={}".format(sigma_e_over_true
                ),
                #    label=str(key) + r" $\mu$={} $\sigma / \mu$={}".format(mean_e_over_true, sigma_e_over_true),
                color=colors[key],
                density=True)
        ax[1,0].hist(event_res_dic[key]["energy_over_true_reco"], bins=bins, histtype="step",
            label="ML",
            color=colors[key],
            density=True)
        ax[1,1].hist(
                event_res_dic[key]["mass_model"],
                histtype="step",
                bins =bins_mass,
                label="ML",
                color=colors[key],
                density=True,
                linewidth=1.5
        )
    ax[1,0].hist(pandora_dic["energy_over_true_reco_pandora"], bins=bins, histtype="step",
                label="Pandora",
                color="blue",
                density=True)
    ax[1,1].hist(pandora_dic["mass_pandora"], histtype="step",
                label="Pandora",
                bins =bins_mass,
                color="blue",
                density=True)
    ax[1,1].hist(pandora_dic["mass_true"],histtype="step",
                label="True",
                bins =bins_mass,
                color="black",
                density=True)
    
    print("energy_over_true_reco_pandora", pandora_dic["energy_over_true_reco_pandora"])
    ax[0,0].grid(1)
    ax[0,0].legend(loc='upper left')
    ax[0,1].grid(1)
    ax[0,1].set_xlabel(r"$E_{vis,pred} / E_{vis,true}$")
    ax[1,0].grid(1)
    ax[1,0].set_xlabel(r"$E_{vis,pred} / E_{vis,reco}$")
    ax[0,1].legend(loc='upper left')
    ax[1,0].legend(loc='upper left')
    ax[1,1].grid(1)
    ax[1,1].legend(loc='upper left')
    fig.tight_layout()
    import os
    fig.savefig(os.path.join(PATH_store, "mass_resolution.pdf"), bbox_inches="tight")
    matplotlib.rcParams.update({'font.size': old_font_size})
