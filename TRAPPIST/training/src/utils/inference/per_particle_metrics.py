import numpy as np
import matplotlib
import os
from src.layers.obtain_statistics import stacked_hist_plot
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from src.utils.pid_conversion import our_to_pandora_mapping, pandora_to_our_mapping, pid_conversion_dict
import pandas as pd
import matplotlib.pyplot as plt
from src.utils.inference.inference_metrics import obtain_MPV_and_68
import concurrent.futures
from src.utils.inference.inference_metrics import calculate_eff, calculate_fakes
import torch
from pathlib import Path
import seaborn as sns
from scipy.optimize import curve_fit
from src.utils.inference.inference_metrics import get_sigma_gaussian
from torch_scatter import scatter_sum, scatter_mean
from src.utils.inference.event_metrics import (
    get_response_for_event_energy,
    plot_mass_resolution, get_mass_contribution_per_PID,
    get_mass_contribution_per_category
)
from matplotlib.colors import LogNorm
from matplotlib.ticker import FormatStrFormatter

def get_mask_id(id, pids_pandora):
    mask_id = np.full((len(pids_pandora)), False, dtype=bool)
    for i in id:
        mask_i = pids_pandora == i
        mask_id = mask_id + mask_i
    mask_id = mask_id.astype(bool)
    return mask_id

def get_response_for_id_i(id, matched_pandora, matched_, tracks=False, perfect_pid=False, mass_zero=False, ML_pid=False):
    pids_pandora = np.abs(matched_pandora["pid"].values)
    mask_id = get_mask_id(id, pids_pandora)
    df_id_pandora = matched_pandora[mask_id]
    pids = np.abs(matched_["pid"].values)
    mask_id = get_mask_id(id, pids)
    df_id = matched_[mask_id]
    (
        mean_p,
        variance_om_p,
        mean_true_rec_p,
        variance_om_true_rec_p,
        energy_resolutions_p,
        energy_resolutions_reco_p,
        mean_baseline,
        variance_om_baseline,
        e_over_e_distr_pandora,
        mean_errors_p,
        variance_errors_p,
        mean_pxyz_pandora, variance_om_pxyz_pandora, masses_pandora, pxyz_true_p, pxyz_pred_p, sigma_phi_pandora, sigma_theta_pandora, distr_phi_pandora, distr_theta_pandora, distr_E_reco_pandora
    ) = calculate_response(df_id_pandora, True, False, tracks=tracks, perfect_pid=perfect_pid, mass_zero=mass_zero, ML_pid=ML_pid, pid=id[0])
    # Pandora: TODO: do some sort of PID for Pandora
    (
        mean,
        variance_om,
        mean_true_rec,
        variance_om_true_rec,
        energy_resolutions,
        energy_resolutions_reco,
        mean_baseline,
        variance_om_baseline,
        e_over_e_distr_model,
        mean_errors,
        variance_errors,
        mean_pxyz, variance_om_pxyz, masses, pxyz_true, pxyz_pred, sigma_phi, sigma_theta, distr_phi, distr_theta, distr_E_reco
    ) = calculate_response(df_id, False, False, tracks=tracks, perfect_pid=perfect_pid, mass_zero=mass_zero, ML_pid=ML_pid, pid=id[0])
    print("COR:__________________________________")
    print(variance_om_p)
    print(variance_om)
    print("RECO:__________________________________")
    print(variance_om_true_rec_p)
    print(variance_om_true_rec)
    dic = {}
    dic["mean_p"] = mean_p
    dic["variance_om_p"] = variance_om_p
    dic["variance_om"] = variance_om
    dic["mean"] = mean
    dic["mean_errors"] = np.array(mean_errors)
    dic["variance_errors"] = np.array(variance_errors)
    dic["variance_errors_p"] = np.array(variance_errors_p)
    dic["mean_errors_p"] = np.array(mean_errors_p)
    dic["energy_resolutions"] = np.array(energy_resolutions)
    dic["energy_resolutions_p"] = np.array(energy_resolutions_p)
    dic["mean_p_reco"] = np.array(mean_true_rec_p)
    dic["variance_om_p_reco"] = np.array(variance_om_true_rec_p)
    dic["energy_resolutions_p_reco"] = np.array(energy_resolutions_reco_p)
    dic["mean_reco"] = mean_true_rec
    dic["variance_om_reco"] = np.array(variance_om_true_rec)
    dic["energy_resolutions_reco"] = np.array(energy_resolutions_reco)
    dic["mean_baseline"] = mean_baseline
    dic["variance_om_baseline"] = np.array(variance_om_baseline)
    dic["distributions_pandora"] = e_over_e_distr_pandora
    dic["distributions_model"] = e_over_e_distr_model
    dic["distributions_pandora_reco"] = distr_E_reco_pandora
    dic["distributions_model_reco"] = distr_E_reco
    dic["mean_pxyz"] = mean_pxyz
    dic["variance_om_pxyz"] = variance_om_pxyz
    dic["mean_pxyz_pandora"] = mean_pxyz_pandora
    dic["variance_om_pxyz_pandora"] = variance_om_pxyz_pandora
    dic["mass_histogram"] = masses
    dic["mass_histogram_pandora"] = masses_pandora
    dic["pxyz_true_p"] = pxyz_true_p
    dic["pxyz_pred_p"] = pxyz_pred_p
    dic["pxyz_true"] = pxyz_true
    dic["pxyz_pred"] = pxyz_pred
    dic["sigma_phi_pandora"] = sigma_phi_pandora
    dic["sigma_theta_pandora"] = sigma_theta_pandora
    dic["sigma_phi"] = sigma_phi
    dic["sigma_theta"] = sigma_theta
    dic["distr_phi"] = distr_phi
    dic["distr_theta"] = distr_theta
    dic["distr_phi_pandora"] = distr_phi_pandora
    dic["distr_theta_pandora"] = distr_theta_pandora
    return dic

def plot_X(
    title,
    photons_dic,
    electrons_dic,
    y_axis,
    PATH_store,
    label1,
    label2,
    reco,
    plot_label1=False,
    plot_label2=False,
):
    colors_list = ["#fde0dd", "#c994c7", "#dd1c77"]  # color list poster neurips
    colors_list = ["#FF0000", "#FF0000", "#0000FF"]
    fig = plt.figure()
    j = 0
    plt.xlabel("Energy [GeV]", fontsize=30)
    # ax[row_i, j].set_xscale("log")
    plt.title(title, fontsize=30)
    plt.grid()
    if plot_label1:
        plt.scatter(
            photons_dic["energy_resolutions" + reco],
            photons_dic[y_axis + reco],
            facecolors=colors_list[1],
            edgecolors=colors_list[1],
            label="ML " + label1,
            marker="x",
            s=50,
        )
        plt.scatter(
            photons_dic["energy_resolutions_p" + reco],
            photons_dic[y_axis + "_p" + reco],
            facecolors=colors_list[2],
            edgecolors=colors_list[2],
            label="Pandora " + label1,
            marker="x",
            s=50,
        )
    if plot_label2:
        plt.scatter(
            electrons_dic["energy_resolutions" + reco],
            electrons_dic[y_axis + reco],
            facecolors=colors_list[1],
            edgecolors=colors_list[1],
            marker="o",
            label="ML " + label2,
            s=50,
        )
        plt.scatter(
            electrons_dic["energy_resolutions_p" + reco],
            electrons_dic[y_axis + "_p" + reco],
            facecolors=colors_list[2],
            edgecolors=colors_list[2],
            marker="o",
            label="Pandora " + label2,
            s=50,
        )
    if title == "Electromagnetic Resolution" or title == "Hadronic Resolution":
        if reco == "":
            if plot_label2:
                plt.scatter(
                    electrons_dic["energy_resolutions"],
                    electrons_dic["variance_om_baseline"],
                    facecolors="black",
                    edgecolors="black",
                    marker=".",
                    label="Baseline " + label2,
                    s=50,
                )
            if plot_label1:
                plt.scatter(
                    photons_dic["energy_resolutions"],
                    photons_dic["variance_om_baseline"],
                    facecolors="black",
                    edgecolors="black",
                    marker=".",
                    label="Baseline " + label1,
                    s=50,
                )
            dic0_fit = get_fit(
                photons_dic["energy_resolutions"], photons_dic["variance_om_baseline"]
            )
            dic01_fit = get_fit(
                electrons_dic["energy_resolutions"],
                electrons_dic["variance_om_baseline"],
            )
        dic1_fit = get_fit(
            photons_dic["energy_resolutions" + reco], photons_dic[y_axis + reco]
        )
        dic1_fit_pandora = get_fit(
           photons_dic["energy_resolutions_p" + reco],
           photons_dic[y_axis + "_p" + reco],
        )
        dic2_fit = get_fit(
            electrons_dic["energy_resolutions" + reco], electrons_dic[y_axis + reco]
        )
        dic2_fit_pandora = get_fit(
            electrons_dic["energy_resolutions_p" + reco],
            electrons_dic[y_axis + "_p" + reco],
        )
        if reco == "":
            fits_l1 = [
                dic0_fit,
                dic1_fit,
                dic1_fit_pandora,
            ]
            fits_l2 = [
                dic01_fit,
                dic2_fit,
                dic2_fit_pandora,
            ]
            color_list_fits_l1 = [
                "black",
                colors_list[1],
                colors_list[2],
            ]
            color_list_fits_l2 = [
                "black",
                colors_list[1],
                colors_list[2],
            ]
            line_type_fits_l1 = ["-", "-", "-."]
            line_type_fits_l2 = ["-", "-", "-."]
        else:
            fits = [dic1_fit, dic1_fit_pandora, dic2_fit, dic2_fit_pandora]
            fits_l1 = dic1_fit
            fits_l2 = dic2_fit
            line_type_fits_l1 = ["-", "-", "-."]
            line_type_fits_l2 = ["-", "-", "-."]
            color_list_fits = [
                colors_list[1],
                colors_list[2],
                colors_list[1],
                colors_list[2],
            ]
            line_type_fits = ["-", "-", "-.", "-."]
        #if plot_label1:
        #    plot_fit(fits_l1, line_type_fits_l1, color_list_fits_l1)
        #if plot_label2:
        #    plot_fit(fits_l2, line_type_fits_l2, color_list_fits_l2)
        if reco == "_reco":
            plt.yscale("log")
        else:
            if title == "Electromagnetic Resolution":
                ymax = 0.3
            else:
                ymax = 0.5
            plt.ylim([0, ymax])
        plt.xlim([0, 55])
        ylabel = r"$\frac{\sigma_{E_{reco}}}{\langle E_{reco} \rangle}$"
        plt.ylabel(ylabel, fontsize=30)
    else:
        ylabel = r"$\langle E_{reco} \rangle / E_{true}$"
        plt.ylabel(ylabel, fontsize=30)
    # loc="upper right",
    plt.tick_params(axis="both", which="major", labelsize=40)
    if title == "Electromagnetic Response" or title == "Hadronic Response":
        plt.ylim([0.6, 1.4])
    plt.legend(fontsize=30, bbox_to_anchor=(1.05, 1), loc="upper left")
    if plot_label1:
        label = label1
    if plot_label2:
        label = label2
    path = os.path.join(PATH_store, title + reco + label + ".pdf")
    fig.savefig(path, bbox_inches="tight")


def plot_fit(fits, line_type_fits, color_list_fits, ax=None):
    fitlabel1 = r"$\frac{\sigma_E}{\langle E \rangle} = \sqrt{\frac{a^2}{E} + \frac{b^2}{E^2} + c^2}$"
    # fitlabel1 = r"$\frac{\sigma_E}{\langle E \rangle} = \sqrt{\frac{a^2}{E} + c^2}$"
    fitlabel2 = ""
    for id_fix, fit in enumerate(fits):
        if id_fix == 0:
            fitlabel = fitlabel1
        else:
            fitlabel = fitlabel2
        fit_a = f"{np.abs(fit[1][0]):.2f}"
        fit_b = f"{np.abs(fit[1][1]):.2f}"
        fit_c = f"{np.abs(fit[1][2]):.2f}"
        if ax is None:
            a = plt
        else:
            a = ax
        a.plot(
            fit[0],
            resolution(fit[0], *fit[1]),
            line_type_fits[id_fix],
            c=color_list_fits[id_fix],
            #'''label=fitlabel
            #+ "\nFit: a = "
            #+ fit_a
            #+ "; b = "
            ##+ fit_b
            #+ "; c = "
            #+ fit_c,'''
        )


def get_fit(energies, errors):
    energies = energies
    errors = errors
    popt, pcov = curve_fit(resolution, energies, errors)
    xdata = np.arange(0, 51, 0.1)
    return [xdata, popt, np.sqrt(np.diag(pcov))]


def resolution(E, a, b, c):
    return (a**2 / E + c**2 + b**2 / E**2) ** 0.5
    # return (a**2 / E + c**2) ** 0.5


def plot_per_energy_resolution2(
    sd_pandora, sd_hgb, matched_pandora, matched_, PATH_store, tracks=False
):
    mask = matched_["calibration_factor"] > 0
    matched_ = matched_[mask]
    if tracks:
        tracks_label = "tracks"
    else:
        tracks_label = ""
    plot_response = True
    if plot_response:
        event_numbers = [0, 1, 2, 3]
        for event_number in event_numbers:
            filename = os.path.join(PATH_store, f"event_{event_number}_pandora.html")
            # plot_event(matched_, pandora=False, output_dir=filename)
            # plot_event(
            #     matched_pandora[matched_pandora.number_batch == event_number],
            #     pandora=True,
            #     output_dir=filename,
            # )
        list_plots = [""]  #  "", "_reco"
        photons_dic = get_response_for_id_i(
            [22], matched_pandora, matched_, tracks=tracks
        )
        hadrons_dic2 = get_response_for_id_i(
            [211], matched_pandora, matched_, tracks=tracks
        )
        # neutrons = get_response_for_id_i(
        #     [2112], matched_pandora, matched_, tracks=tracks
        # )
        # protons = get_response_for_id_i(
        #     [2212], matched_pandora, matched_, tracks=tracks
        # )
        #event_res_dic = get_response_for_event_energy(matched_pandora, matched_)
        plot_per_particle = True
        if plot_per_particle:
            for el in list_plots:
                plot_one_label(
                    "Electromagnetic Resolution",
                    photons_dic,
                    "variance_om",
                    PATH_store,
                    "Photons",
                    el,
                    tracks=tracks_label,
                )
                plot_one_label(
                    "Electromagnetic Response",
                    photons_dic,
                    "mean",
                    PATH_store,
                    "Photons",
                    el,
                    tracks=tracks_label,
                )
                plot_one_label(
                    "Hadronic Resolution",
                    hadrons_dic2,
                    "variance_om",
                    PATH_store,
                    "Pions",
                    el,
                    tracks=tracks_label,
                )
                plot_one_label(
                    "Hadronic Response",
                    hadrons_dic2,
                    "mean",
                    PATH_store,
                    "Pions",
                    el,
                    tracks=tracks_label,
                )

def plot_hist_distr(values, label, ax, color, bins=np.linspace(0, 3, 50)):
    ax.hist(values, bins=bins, histtype="step", label=label, color=color, density=True)
    ax.legend()
    ax.grid(1)

def plot_pxyz_resolution(x, resolutions_pxyz_pandora, resolutions_pxyz_model, axs, key):
    for i in [0, 1, 2, 3]:
        axs[i].scatter(
            x,
            resolutions_pxyz_model[:, i],
            facecolors="red",
            edgecolors="red",
            label=key,
            marker="x",
            s=50,
        )
        if resolutions_pxyz_pandora.shape[1] < i:
            axs[i].scatter(
                x,
                resolutions_pxyz_pandora[:, i],
                facecolors="blue",
                edgecolors="blue",
            
                label="Pandora",
                marker="x",
                s=50,
            )
        axs[i].grid(1)
        axs[i].legend()

def plot_mass_hist(masses_lst, masses_pandora_lst, axs, bars=[], energy_ranges=[[0, 5], [5, 15], [15, 35], [35, 50]]):
    # bars: list of energies at which to plot a vertical line
    return
    masses = masses_lst[0]
    masses_pandora = masses_pandora_lst[0]
    is_trk_in_clust_pandora = [x.values for x in masses_pandora_lst[1]]
    for i in range(4):
        # percentage of nans
        perc_nan_model = int(torch.sum(torch.isnan(masses[i])) / len(masses[i]) * 100)
        perc_nan_pandora = int(torch.sum(torch.isnan(masses_pandora[i])) / len(masses_pandora[i]) * 100)
        #bins = np.linspace(-1, 1, 50)
        bins = 100
        axs[i].hist(masses[i], bins=bins, histtype="step", label="ML (nan {} %)".format(perc_nan_model), color="red", density=True)
        filt = is_trk_in_clust_pandora[i]
        #axs[i].hist(masses_pandora[i][filt==1], bins=bins, histtype="step", label="Pandora (nan {}%), track in cluster".format(perc_nan_pandora), color="blue", density=True)
        #axs[i].hist(masses_pandora[i][filt==0], bins=bins, histtype="step", label="Pandora (nan {}%), track not in cluster".format(perc_nan_pandora), color="green", density=True)
        axs[i].hist(masses_pandora[i], bins=bins, histtype="step", label="Pandora (nan {}%)".format(perc_nan_pandora), color="blue", density=True)
        #max_mass = max(masses_pandora[i].max(), masses[i].max())
        axs[i].set_title(f"[{energy_ranges[i][0]}, {energy_ranges[i][1]}] GeV")
        axs[i].legend()
        axs[i].grid(1)
        axs[i].set_yscale("log")
        mean_mass = masses[i][torch.isnan(masses[i])].mean()
        mean_mass_pandora = masses_pandora[i][torch.isnan(masses_pandora[i])].mean()
        #for bar in bars:
        #    if bar * 0.95 < mean_mass:
        #        axs[i].axvline(bar, color="black", linestyle="--")

def mixed_percentages(cm, fake_row, fake_norm="column"):
    """
    Return a percentage matrix where all rows are row-normalized,
    except you can set the fake_row to use 'row' or 'column' normalization.
    Default is 'column' for the fake row.
    """
    cm = cm.astype(float)

    # Row-normalization for everything
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        cm_row = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0) * 100.0

    if fake_norm == "row":
        return cm_row

    # Column-normalization (we'll only use it to replace the fake_row)
    col_sums = cm.sum(axis=0, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        cm_col = np.divide(cm, col_sums, out=np.zeros_like(cm, dtype=float), where=col_sums != 0) * 100.0

    disp = cm_row.copy()
    disp[fake_row, :] = cm_col[fake_row, :]   
    return disp

def plot_confusion_matrix(sd_hgb1, save_dir, add_pie_charts=False, ax=None, ax1=None, ax2=None, suffix="", prefix="ML"):
    #sd_hgb1["pid_4_class_true"] = sd_hgb1["pid"].map(pid_conversion_dict)
    # sd_hgb1["pred_pid_matched"][sd_hgb1["pred_pid_matched"] < -1] = np.nan
    #sd_hgb1.loc[sd_hgb1["pred_pid_matched"] == -1, "pred_pid_matched"] = np.nan
    class_true = np.array(sd_hgb1["pid_4_class_true"].values)
    class_pred = np.array(sd_hgb1["pred_pid_matched"].values)
    n_classes = class_true[~np.isnan(class_true)].max() + 1
    n_classes_pred = class_pred[~np.isnan(class_pred)].max() + 1
    n_classes = max(n_classes, n_classes_pred)
    class_nan = n_classes # For 'fake' and 'missed'
    is_muons = n_classes == 5
    # print("Unique classes", set(list(class_true[~np.isnan(class_true)])), set(list(class_pred[~np.isnan(class_pred)])))
    assert ((np.isnan(class_true)) * (np.isnan(class_pred))).sum() == 0
    pid_true = sd_hgb1["pid"].values
    is_trk = sd_hgb1.is_track_in_cluster.values
    no_nan_filter = np.ones_like(class_true, dtype=bool)
    #no_nan_filter[np.isnan(class_true)] = False
    #no_nan_filter[np.isnan(class_pred)] = False
    class_true[np.isnan(class_true)] = class_nan
    class_pred[np.isnan(class_pred)] = class_nan
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(class_true, class_pred, labels=list(range(n_classes + 1)))
    if cm.shape[0] < n_classes + 1:
        cm = np.pad(cm, ((0, n_classes + 1 - cm.shape[0]), (0, n_classes + 1 - cm.shape[1]))) # for the GT clustering
    effi_diag = np.trace(cm) / np.sum(cm)
    #assert cm.shape[0] == cm.shape[1]
    #assert cm.shape[0] == n_classes + 1
    savefigs = ax is None
    # Plot cm
    # Add pie charts
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    if add_pie_charts:
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if cm[i, j] > 0:
                    x = j
                    y = i
                    size = 1.0 # Adjust size as needed
                    true_pids = pid_true[(class_true == i) & (class_pred == j)]
                    unique, counts = np.unique(true_pids, return_counts=True)
                    # Pie chart in here
                    ax_inset = inset_axes(ax, width="100%", height="100%", loc="center",
                                          bbox_to_anchor=(x, y, size, size),
                                          bbox_transform=ax.transData,
                                          borderpad=0)
                    # each label should always have the same color, i.e. 130-> purple etc.
                    ax_inset.pie(counts, labels=unique, textprops={'fontsize': 5})
    if not is_muons:
        class_names_true = ["e", "CH", "NH", "gamma", "fake"]
        class_names_pred = ["e", "CH", "NH", "gamma", "missed"]
    else:
        class_names_true = ["e", "CH", "NH", "gamma", "mu", "fake"]
        class_names_pred = ["e", "CH", "NH", "gamma", "mu", "missed"]
    if not add_pie_charts:
        fake_row = len(class_names_pred) - 1  # last row
        cm_percent = mixed_percentages(cm, fake_row, fake_norm="column") 
        palette = sns.dark_palette("#69d", reverse=True, as_cmap=True)
        sns.heatmap(cm_percent, annot=cm, fmt="d", xticklabels=class_names_pred, yticklabels=class_names_true, ax=ax, cmap=palette)
        # Loop over each cell to add custom annotations
        ax.hlines(fake_row, xmin=0, xmax=cm.shape[0], linewidth=3, color="black") #bold line to seperate fakes
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                # Raw count
                count = cm[i, j]
                # Percentage with one decimal place
                percent = cm_percent[i, j]
                # Display count on top, percentage below in smaller font
                ax.text(j + 0.5, i + 0.85, f"{percent:.1f}%",
                        ha="center", va="center", color="0.8",
                        fontsize=9)
    else:
        # Now plot just the ticks on the plot without the CM
        ax.set_xticks(np.arange(cm.shape[1]) + 0.5, minor=False)
        ax.set_yticks(np.arange(cm.shape[0]) + 0.5, minor=False)
        ax.set_xticklabels(class_names_pred, minor=False)
        ax.set_yticklabels(class_names_true, minor=False)
        ax.grid(False)
    # axes
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(prefix + " " + suffix)
    suffix = ""
    if add_pie_charts:
        suffix = "_pie_charts"
    if savefigs:
        fig.savefig(os.path.join(save_dir, "confusion_matrix_PID" + suffix + ".pdf"), bbox_inches="tight")
        plt.clf()
    f =  (is_trk == 1)
    f1 = (is_trk == 0)

    cm = confusion_matrix(class_true[f], class_pred[f] , labels=list(range(n_classes + 1)))
    cm1 = confusion_matrix(class_true[f1], class_pred[f1], labels=list(range(n_classes + 1)))
    ax.set_title(f"{prefix} {suffix}, $\\epsilon_{{\\mathrm{{diag}}}}$={effi_diag:.2f}")
    # plot cm
    savefigs = ax1 is None
    if ax1 is None:
        fig, ax1 = plt.subplots()
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_percent = cm / row_sums * 100  # Get percentages per row
    palette = sns.dark_palette("#69d", reverse=True, as_cmap=True)
    sns.heatmap(cm_percent, annot=cm, fmt="d", xticklabels=class_names_pred, yticklabels=class_names_true, ax=ax1, cmap=palette)
    #sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, ax=ax1, norm=LogNorm())
    # Loop over each cell to add custom annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            # Raw count
            count = cm[i, j]
            # Percentage with one decimal place
            percent = cm_percent[i, j]
            # Display count on top, percentage below in smaller font
            ax1.text(j + 0.5, i + 0.85, f"{percent:.1f}%",
                    ha="center", va="center", color="0.8",
                    fontsize=10)
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("True")
    ax1.set_title("Confusion Matrix (track in cluster)" + suffix)
    if savefigs:
        fig.savefig(os.path.join(save_dir, "confusion_matrix_PID_track_in_cluster.pdf"), bbox_inches="tight")
    savefigs = ax2 is None
    if ax2 is None:
        fig, ax2 = plt.subplots()
    #sns.heatmap(cm1, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, ax=ax2, norm=LogNorm())
    row_sums = cm1.sum(axis=1, keepdims=True)
    cm_percent = cm1 / row_sums * 100  # Get percentages per row
    palette = sns.dark_palette("#69d", reverse=True, as_cmap=True)
    sns.heatmap(cm_percent, annot=cm1, fmt="d", xticklabels=class_names_pred, yticklabels=class_names_true, ax=ax2, cmap=palette)
    # Loop over each cell to add custom annotations
    for i in range(cm1.shape[0]):
        for j in range(cm1.shape[1]):
            # Raw count
            count = cm1[i, j]
            # Percentage with one decimal place
            percent = cm_percent[i, j]
            # Display count on top, percentage below in smaller font
            ax2.text(j + 0.5, i + 0.85, f"{percent:.1f}%",
                    ha="center", va="center", color="0.8",
                    fontsize=10)
    # axes
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("True")
    ax2.set_title("Confusion Matrix (no track in cluster)" + suffix)
    if savefigs:
        fig.savefig(os.path.join(save_dir, "confusion_matrix_PID_NO_track_in_cluster.pdf"), bbox_inches="tight")

def nanindex(x, list):
    if pd.isna(x):
        return np.nan
    return list.index(x)

def plot_confusion_matrix_pandora(sd_pandora, save_dir, add_pie_charts=False, ax=None, ax1=None, ax2=None, suffix=""):
    class_true = np.array(sd_pandora.pid.values)
    class_pred = np.array(sd_pandora.pandora_pid.values)
    #allowed_pids =  sorted(sd_pandora.pandora_pid.dropna().unique())
    #all_pids = [0] + [int(x) for x in allowed_pids]
    #class_true[~np.isin(class_true, allowed_pids)] = 0 # 0 = "other" class
    pids_true = sd_pandora["pid"].values
    is_trk = sd_pandora.is_track_in_cluster.values
    #class_true = np.array([nanindex(x, all_pids) for x in class_true])
    #class_pred = np.array([nanindex(x, all_pids) for x in class_pred])
    class_true_no_nan = np.array([pid_conversion_dict[x] for x in class_true[~np.isnan(class_true)]])
    max_class = class_true_no_nan.max()
    assert max_class in [3, 4]
    is_muons = max_class == 4
    assert ((np.isnan(class_true)) * (np.isnan(class_pred))).sum() == 0 # Maybe the pid_conversion_dict is not fully up-to-date?
    class_true = np.array([pid_conversion_dict.get(x, max_class+1) for x in class_true])
    class_pred = np.array([pandora_to_our_mapping.get(x, max_class+1) for x in class_pred])
    #no_nan_filter = ~np.isnan(class_pred) & ~np.isnan(class_true)
    no_nan_filter = np.ones_like(class_true, dtype=bool)
    assert np.isnan(class_true).sum() == 0
    assert np.isnan(class_pred).sum() == 0
    from sklearn.metrics import confusion_matrix
    savefigs = ax is None
    if savefigs:
        fig, ax = plt.subplots(figsize=(6, 6))
    cm = confusion_matrix(class_true[no_nan_filter], class_pred[no_nan_filter])
    effi_diag = np.trace(cm) / np.sum(cm)
    if add_pie_charts:
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if cm[i, j] > 0:
                    x = j
                    y = i
                    size = 1.0 # Adjust size as needed
                    true_pids = pids_true[(class_true == i) & (class_pred == j)]
                    unique, counts = np.unique(true_pids[~np.isnan(true_pids)], return_counts=True)
                    # Pie chart in here
                    ax_inset = inset_axes(ax, width="100%", height="100%", loc="center",
                                          bbox_to_anchor=(x, y, size, size),
                                          bbox_transform=ax.transData,
                                          borderpad=0)
                    percentage_counts = counts / np.sum(counts) * 100
                    # move categories less than 0.5% into 'other'
                    unique = np.array([x if y >= 1 else "other" for x, y in zip(unique, percentage_counts)])
                    counts = np.array([y if y >= 1 else np.sum(counts[percentage_counts < 1]) for y in counts])
                    # fix - now 'other' repeats too many times
                    unique1, counts1 = [], []
                    for u, c in zip(unique, counts):
                        if u not in unique1:
                            unique1.append(u)
                            counts1.append(c)
                        else:
                            pass
                    ax_inset.pie(counts1, labels=unique1, textprops={'fontsize': 10})
    if is_muons:
        class_names_pred = ["e", "CH", "NH", "gamma", "mu", "missed"]
        class_names_true = ["e", "CH", "NH", "gamma", "mu", "fake"]
    else:
        class_names_pred = ["e", "CH", "NH", "gamma", "missed"]
        class_names_true = ["e", "CH", "NH", "gamma", "fake"]

    if not add_pie_charts:
        # Calculate percentages
        fake_row = len(class_names_pred) - 1  # last row
        cm_percent = mixed_percentages(cm, fake_row, fake_norm="column") 
        palette = sns.dark_palette("#69d", reverse=True, as_cmap=True)
        sns.heatmap(cm_percent, annot=cm, fmt="d", xticklabels=class_names_pred, yticklabels=class_names_true, ax=ax,
                    cmap=palette)
        ax.hlines(fake_row, xmin=0, xmax=cm.shape[0], linewidth=3, color="black") #bold line to seperate fakes
        # Loop over each cell to add custom annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                # Raw count
                count = cm[i, j]
                # Percentage with one decimal place
                percent = cm_percent[i, j]
                # Display count on top, percentage below in smaller font
                ax.text(j + 0.5, i + 0.85, f"{percent:.1f}%",
                        ha="center", va="center", color="0.8",
                        fontsize=10)

    else:
        # now plot just the ticks on the plot without the CM
        ax.set_xticks(np.arange(cm.shape[1]) + 0.5, minor=False)
        ax.set_yticks(np.arange(cm.shape[0]) + 0.5, minor=False)
        ax.set_xticklabels(class_names_pred, minor=False)
        ax.set_yticklabels(class_names_true, minor=False)
        ax.grid(False)
    # axes
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Pandora " + suffix)
    suffix = ""
    if add_pie_charts:
        suffix = "_pie_charts"
    if savefigs:
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "Pandora_confusion_matrix_PID" + suffix + ".pdf"), bbox_inches="tight")
    if add_pie_charts:
        return
    f = no_nan_filter & (is_trk == 1)
    f1 = no_nan_filter & (is_trk == 0)
    cm = confusion_matrix(class_true[f], class_pred[f])
    cm1 = confusion_matrix(class_true[f1], class_pred[f1])
    ax.set_title(f"Pandora {suffix}, $\\epsilon_{{\\mathrm{{diag}}}}$={effi_diag:.2f}")
    # plot cm
    savefigs = ax1 is None
    if ax1 is None:
        fig, ax1 = plt.subplots()
    #sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, ax=ax1, norm=LogNorm())
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_percent = cm / row_sums * 100  # Get percentages per row
    palette = sns.dark_palette("#69d", reverse=True, as_cmap=True)
    sns.heatmap(cm_percent, annot=cm, fmt="d", xticklabels=class_names_pred, yticklabels=class_names_true, ax=ax1,
                cmap=palette)
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("True")
    ax1.set_title("Pandora CM (track in cluster)" + suffix)
    # Loop over each cell to add custom annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            # Raw count
            count = cm[i, j]
            # Percentage with one decimal place
            percent = cm_percent[i, j]
            # Display count on top, percentage below in smaller font
            ax1.text(j + 0.5, i + 0.85, f"{percent:.1f}%",
                    ha="center", va="center", color="0.8",
                    fontsize=10)
    if savefigs:
        fig.savefig(os.path.join(save_dir, "Pandora_confusion_matrix_PID_track_in_cluster.pdf"), bbox_inches="tight")
    savefigs = ax2 is None
    if savefigs:
        fig, ax2 = plt.subplots()
    #sns.heatmap(cm1, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, ax=ax2, norm=LogNorm())
    palette = sns.dark_palette("#69d", reverse=True, as_cmap=True)
    row_sums = cm1.sum(axis=1, keepdims=True)
    cm_percent = cm1 / row_sums * 100  # Get percentages per row
    sns.heatmap(cm_percent, annot=cm1, fmt="d", xticklabels=class_names_pred, yticklabels=class_names_true, ax=ax2,
                cmap=palette)
    # Loop over each cell to add custom annotations
    for i in range(cm1.shape[0]):
        for j in range(cm1.shape[1]):
            # Raw count
            count = cm1[i, j]
            # Percentage with one decimal place
            percent = cm_percent[i, j]
            # Display count on top, percentage below in smaller font
            ax2.text(j + 0.5, i + 0.85, f"{percent:.1f}%",
                    ha="center", va="center", color="0.8",
                    fontsize=10)
    # axes
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("True")
    ax2.set_title("Pandora CM (no track in cluster)" + suffix)
    if savefigs:
        fig.savefig(os.path.join(save_dir, "Pandora_confusion_matrix_PID_NO_track_in_cluster.pdf"), bbox_inches="tight")
        plt.clf()

def compute_score_certainty(scores):
    # scores: (n_samples, n_classes)
    # certainty is defined as the difference between the largest in each sample and the second-largest
    # scores: (n_samples, n_classes)
    # certainty is defined as the difference between the largest in each sample and the second largest
    top2_scores, _ = torch.topk(scores, 2, dim=1)
    certainty = top2_scores[:, 0] - top2_scores[:, 1]
    return certainty

def analyze_fakes_PID(matched_pandora, matched_all, PATH_store):
    fakes_pandora = matched_pandora[pd.isna(matched_pandora.true_showers_E)]
    nonfakes_pandora = matched_pandora[~pd.isna(matched_pandora.true_showers_E)]
    fakes_model = matched_all[pd.isna(matched_all.true_showers_E)]
    nonfakes_model = matched_all[~pd.isna(matched_all.true_showers_E)]
    fakes_model_extra_f = np.stack(fakes_model.matched_extra_features.values)
    pid_logits_fakes = torch.tensor(fakes_model_extra_f[:, 2:])
    nonfakes_model_extra_f = np.stack(nonfakes_model.matched_extra_features.values)
    energies = [0, 5, 15, 35, 50]
    pid_logits_nonfakes = torch.tensor(nonfakes_model_extra_f[:, 2:])
    pid_logits_fakes = torch.softmax(pid_logits_fakes, dim=1)
    pid_logits_nonfakes = torch.softmax(pid_logits_nonfakes, dim=1)
    certainty_fakes = compute_score_certainty(pid_logits_fakes)
    certainty_nonfakes = compute_score_certainty(pid_logits_nonfakes)
    fig, ax = plt.subplots()
    ax.hist(certainty_fakes, histtype='step', bins=np.linspace(0, 1, 100), label="fakes", color="red", density=1)
    ax.hist(certainty_nonfakes, histtype='step', bins=np.linspace(0, 1, 100), label="nonfakes", color="blue", density=1)
    ax.legend()
    ax.set_yscale("log")
    fig.savefig(os.path.join(PATH_store, "fakes_certainty.pdf"))

def analyze_fakes(matched_pandora, matched_all, PATH_store):
    fakes_pandora = matched_pandora[pd.isna(matched_pandora.true_showers_E)]
    nonfakes_pandora = matched_pandora[~pd.isna(matched_pandora.true_showers_E)]
    fakes_model = matched_all[pd.isna(matched_all.true_showers_E)]
    nonfakes_model = matched_all[~pd.isna(matched_all.true_showers_E)]
    # energy bins:
    fakes_model_extra_f = np.stack(fakes_model.matched_extra_features.values)
    highest_beta = fakes_model_extra_f[:, 1]
    nonfakes_model_extra_f = np.stack(nonfakes_model.matched_extra_features.values)
    energies = [0, 5, 15, 35, 50]
    filt_model_fakes = (np.isnan(highest_beta)) | (highest_beta >= 0.05) | (highest_beta <= 0.03)
    fakes_filt = fakes_model[~filt_model_fakes]
    highest_beta = nonfakes_model_extra_f[:, 1]
    filt_model_nonfakes = (np.isnan(highest_beta)) | (highest_beta >= 0.05) | (highest_beta <= 0.03)
    nonfakes_filt = nonfakes_model[~filt_model_nonfakes]
    fig, ax = plt.subplots(2, 1, figsize=(5, 10))
    bins = np.linspace(0,5,200)
    ax[0].hist(fakes_filt.calibrated_E, histtype='step', bins=bins, label="fakes", color="red")
    ax[0].hist(nonfakes_filt.calibrated_E, histtype='step', bins=bins, label="nonfakes", color="blue")
    ax[0].legend()
    ax[0].set_title("calibrated E")
    bins = np.linspace(-2, 2, 100)
    frac_fakes = ((fakes_filt.pred_showers_E - fakes_filt.calibrated_E) / fakes_filt.pred_showers_E).values
    frac_nonfakes = ((nonfakes_filt.pred_showers_E - nonfakes_filt.calibrated_E) / nonfakes_filt.pred_showers_E).values
    ax[1].hist(frac_fakes[frac_fakes != 0], histtype='step', bins=bins, label="fakes", color="red")
    ax[1].hist(frac_nonfakes[frac_nonfakes!=0] ,histtype='step', bins=bins, label="nonfakes", color="blue")
    ax[1].legend()
    ax[1].set_title("(reco E - cali E) / reco E")
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    fig.show()
    fig, ax = plt.subplots()
    bins = np.arange(0, 100, 1)
    ax.hist(fakes_model_extra_f[:, 0], histtype='step', bins=bins, label="ML fakes", color="red", density=True)
    ax.hist(nonfakes_model_extra_f[:, 0], histtype='step', bins=bins, label="ML matched", color="blue", density=True)
    ax.legend()
    ax.set_title("Reco energy distribution for fakes")
    fig.show()

    fig, ax = plt.subplots(figsize=(8,8))
    bins = np.linspace(0, 1, 200)
    ax.hist(fakes_model_extra_f[:, 1], histtype='step', bins=bins, label="ML fakes", color="red")
    ax.hist(nonfakes_model_extra_f[:, 1], histtype='step', bins=bins, label="ML matched", color="blue")
    ax.legend()
    ax.set_yscale("log")
    ax.set_xlabel(r"Highest $\beta$")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(os.path.join(PATH_store, "fakes_highest_beta_value.pdf"))

    # for each bin, plot a pie chart with the PIDs and their percentages.
    fig, ax = plt.subplots(2, 4, figsize=(10, 5))
    for i in range(len(energies) - 1):
        fakes_pandora_i = fakes_pandora[(fakes_pandora.pred_showers_E > energies[i]) & (fakes_pandora.pred_showers_E < energies[i + 1])]
        fakes_model_i = fakes_model[(fakes_model.pred_showers_E > energies[i]) & (fakes_model.pred_showers_E < energies[i + 1])]
        # plot the PID distribution for fakes
        pid_names = ["e", "CH", "NH", "gamma", "mu"]
        pid_model = fakes_model_i.pred_pid_matched.values
        pid_pandora = fakes_pandora_i.pandora_pid.values
        pid_pandora = [pid_names[pandora_to_our_mapping[int(x)]] for x in pid_pandora]
        pid_model = [pid_names[int(x)] for x in pid_model]
        pid_model = pd.value_counts(pid_model)
        pid_pandora = pd.value_counts(pid_pandora)
        colors = ["blue", "red", "green", "purple", "orange"]
        #pid_model = [pid_names[int(x)] for x in pid_model]
        #pid_pandora.index = [pid_names[pandora_to_our_mapping[int(x)]] for x in pid_pandora.index]
        ax[0, i].pie(pid_pandora, labels=pid_pandora.index)
        ax[0, i].set_title(f"Pandora [{energies[i]}, {energies[i + 1]}] GeV")
        ax[1, i].pie(pid_model, labels=pid_model.index)
        ax[1, i].set_title(f"Model [{energies[i]}, {energies[i + 1]}] GeV")
        # plot the PID distribution for nonfakes
        #nonfakes_pandora_i = nonfakes_pandora[(nonfakes_pandora.pred_showers_E > energies[i]) & (nonfakes_pandora.pred_showers_E < energies[i + 1])]
        #nonfakes_model_i = nonfakes_model[(nonfakes_model.pred_showers_E > energies[i]) & (nonfakes_model.pred_showers_E < energies[i + 1])]
        #pid_pandora = nonfakes_pandora_i.pid.value_counts()
    fig.savefig(os.path.join(PATH_store, "fakes_PID_structure.pdf"))


    '''fig, ax = plt.subplots(4, 1, figsize=(10, 15))
    bins = np.linspace(0, 5, 100)
    density=False
    # plot the reco energy histogram for fakes and nonfakes
    ax[0].hist(fakes_pandora.pred_showers_E, bins=bins, histtype="step", label="Pandora fakes reco", color="blue", density=density)
    ax[0].hist(nonfakes_pandora.pred_showers_E, bins=bins, histtype="step", label="Pandora nonfakes reco", color="red", density=density)
    ax[1].hist(fakes_model.pred_showers_E, bins=bins, histtype="step", label="Model fakes reco", color="blue", density=density)
    ax[1].hist(nonfakes_model.pred_showers_E, bins=bins, histtype="step", label="Model nonfakes reco", color="red", density=density)
    # plot the reco energy resolution for fakes and nonfakes
    ax[2].hist(fakes_pandora.pandora_calibrated_pfo, bins=bins, histtype="step", label="Pandora fakes", color="blue", density=density)
    ax[2].hist(nonfakes_pandora.pandora_calibrated_pfo, bins=bins, histtype="step", label="Pandora nonfakes", color="red", density=density)
    ax[3].hist(fakes_model.calibrated_E, bins=bins, histtype="step", label="Model fakes", color="blue", density=density)
    ax[3].hist(nonfakes_model.calibrated_E, bins=bins, histtype="step", label="Model nonfakes", color="red", density=density)
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    ax[3].legend()
    ax[0].set_yscale("log")
    ax[1].set_yscale("log")
    ax[2].set_yscale("log")
    ax[3].set_yscale("log")
    fig.savefig(os.path.join(PATH_store, "fakes_analysis.pdf"))'''


def plot_cm_per_energy(sd_hgb, sd_pandora, path_store_summary_plots, path_store, sd_hgb_gt=None):
    # Plot confusion matrix for each energy bin
    n_plots = 3
    if sd_hgb_gt is None:
        n_plots = 2
    fig, ax = plt.subplots(n_plots, 3, figsize=(12, 12 * n_plots / 3))
    energies = [0, 1, 10, 100]
    for i in range(len(energies) - 1):
        cond = ((sd_hgb.true_showers_E > energies[i]) & (sd_hgb.true_showers_E < energies[i + 1])) | (np.isnan(sd_hgb.pid) & ((sd_hgb.pred_showers_E > energies[i]) & (sd_hgb.pred_showers_E < energies[i + 1])))
        sd_hgb_i = sd_hgb[cond]
        cond_pandora = ((sd_pandora.true_showers_E > energies[i]) & (sd_pandora.true_showers_E < energies[i + 1])) |  ((np.isnan(sd_pandora.pid)) & ((sd_pandora.pred_showers_E > energies[i]) & (sd_pandora.pred_showers_E < energies[i + 1])))
        sd_pandora_i = sd_pandora[cond_pandora]
        suffix = "[{},{}] GeV".format(energies[i], energies[i + 1])
        #plot_confusion_matrix(sd_hgb_i, path_store, add_pie_charts=False, ax=ax[0, i], ax1=ax[2, i], ax2=ax[4, i], suffix=suffix)
        #plot_confusion_matrix_pandora(sd_pandora_i, path_store, add_pie_charts=False, ax=ax[1, i], ax1=ax[3, i], ax2=ax[5, i], suffix=suffix)
        #plot_confusion_matrix(sd_hgb_i, path_store, add_pie_charts=True, ax=ax[6, i], suffix=suffix)
        #plot_confusion_matrix_pandora(sd_pandora_i, path_store, add_pie_charts=True, ax=ax[7, i], suffix=suffix)
        plot_confusion_matrix(sd_hgb_i, path_store, add_pie_charts=False, ax=ax[1, i], ax1=None, ax2=None, suffix=suffix)
        if n_plots == 3:
            cond_gt = ((sd_hgb_gt.true_showers_E > energies[i]) & (sd_hgb_gt.true_showers_E < energies[i + 1])) | (np.isnan(sd_hgb_gt.pid) & ((sd_hgb_gt.pred_showers_E > energies[i]) & (sd_hgb_gt.pred_showers_E < energies[i + 1])))
            plot_confusion_matrix(sd_hgb_gt[cond_gt], path_store, add_pie_charts=False, ax=ax[2, i], ax1=None, ax2=None, suffix=suffix, prefix="ML GTC")
        plot_confusion_matrix_pandora(sd_pandora_i, path_store, add_pie_charts=False, ax=ax[0, i], ax1=None, ax2=None, suffix=suffix)
    fig.tight_layout()
    fig.savefig(os.path.join(path_store_summary_plots, "confusion_matrix_per_energy.pdf"))

def plot_cm_per_energy_on_overview(sd_hgb, sd_pandora, path_store, ax):
    # Plot confusion matrix for each energy bin, but on an external plot
    # Ax should be as follows: ax[i, 0]: CM by energy; ax[i, 1]: CM-Pandora by energy
    energies = [0, 1, 10, 100]
    for i in range(len(energies) - 1):
        cond = ((sd_hgb.true_showers_E > energies[i]) & (sd_hgb.true_showers_E < energies[i + 1])) | (np.isnan(sd_hgb.pid) & ((sd_hgb.pred_showers_E > energies[i]) & (sd_hgb.pred_showers_E < energies[i + 1])))
        sd_hgb_i = sd_hgb[cond]
        cond_pandora = ((sd_pandora.true_showers_E > energies[i]) & (sd_pandora.true_showers_E < energies[i + 1])) |  ((np.isnan(sd_pandora.pid)) & ((sd_pandora.pred_showers_E > energies[i]) & (sd_pandora.pred_showers_E < energies[i + 1])))
        sd_pandora_i = sd_pandora[cond_pandora]
        suffix = "[{},{}] GeV".format(energies[i], energies[i + 1])
        plot_confusion_matrix(sd_hgb_i, path_store, add_pie_charts=False, ax=ax[i, 0], suffix=suffix)
        plot_confusion_matrix_pandora(sd_pandora_i, path_store, add_pie_charts=False, ax=ax[i, 1], suffix=suffix)

def quick_plot_mass(matched_, matched_pandora, path_store):
    evt = get_response_for_event_energy(
        matched_pandora, matched_, perfect_pid=0, mass_zero=0, ML_pid=1
    )
    plot_mass_resolution({"ML": evt}, path_store)
    print("Saved mass resolution")


def plot_mass_contribution_per_PID(matched_, matched_pandora, path_store):
    old_fontsize = matplotlib.rcParams["font.size"]
    matplotlib.rcParams["font.size"] = 11
    bins_mass = np.linspace(0, 2, 200)
    energy_bins = [[0, 1], [1, 10], [10, 100]]
    matplotlib.rcParams.update({'font.size': 11})
    for bin in energy_bins:
        filter_pandora = (matched_pandora.true_showers_E > bin[0]) & (matched_pandora.true_showers_E < bin[1])
        filter_model = (matched_.true_showers_E > bin[0]) & (matched_.true_showers_E < bin[1])
        (massPID, massPIDpandora, EPID, EPIDpandora, frac_E_model,
         frac_E_pandora, frac_E_model_true, frac_E_pandora_true,
         pid_true_over_true) = get_mass_contribution_per_PID(matched_pandora[filter_pandora],
                                                             matched_[filter_model],
                                                             ML_pid=1,
                                                             perfect_pid=0,
                                                             mass_zero=0)
        figs_mass_hist, axs_mass_hist = plt.subplots(6, 3, figsize=(8, 15)) # also plot fraction of the whole event energy
        for i, pid in enumerate([22, 11, 130, 211, 2112, 2212]):
            if pid not in massPID or pid not in massPIDpandora:
                continue
            axs_mass_hist[i, 0].hist(massPID[pid], bins=bins_mass, histtype="step", label="ML", color="red",
                                     density=True)
            axs_mass_hist[i, 0].hist(massPIDpandora[pid], bins=bins_mass, histtype="step", label="Pandora", color="blue",
                                     density=True)
            axs_mass_hist[i, 1].grid(1)
            axs_mass_hist[i, 1].set_xlabel(f"E pred., {pid} / E true, {pid}")
            axs_mass_hist[i, 1].hist(EPID[pid], bins=bins_mass, histtype="step", label="ML", color="red", density=True)
            axs_mass_hist[i, 1].hist(EPIDpandora[pid], bins=bins_mass, histtype="step", label="Pandora", color="blue",
                                     density=True)
            binsE = np.linspace(0, 1.0, 100)
            axs_mass_hist[i, 0].legend()
            axs_mass_hist[i, 1].legend()
            axs_mass_hist[i, 2].grid(1)
            axs_mass_hist[i, 2].set_title(f"Pred. {pid} contr. to pred. evt. E")
            axs_mass_hist[i, 2].set_xlabel(f"Pred. E {pid} / Pred. E")
            axs_mass_hist[i, 0].set_xlabel(f"Pred. m {pid} / Pred. m")
            axs_mass_hist[i, 2].hist(frac_E_model[pid], bins=binsE, histtype="step", label="ML", color="red", density=True)
            axs_mass_hist[i, 2].hist(frac_E_pandora[pid], bins=binsE, histtype="step", label="Pandora", color="blue", density=True)
            axs_mass_hist[i, 2].hist(pid_true_over_true[pid], bins=binsE, histtype="step", label="Truth", color="green", density=True)
            axs_mass_hist[i, 2].legend()
            axs_mass_hist[i, 0].set_yscale("log")
            axs_mass_hist[i, 1].set_yscale("log")
            axs_mass_hist[i, 2].set_yscale("log")
        figs_mass_hist.tight_layout()
        figs_mass_hist.savefig(os.path.join(path_store, f"E_breakdown_by_PID_{bin[0]}_{bin[1]}_GeV.pdf"))
        #matplotlib.rcParams["font.size"] = old_fontsize

def plot_mass_contribution_per_category(matched_, matched_pandora, matched_gt, path_store, energy_bins=None, perfect_pid=False):
    old_fontsize = matplotlib.rcParams["font.size"]
    matplotlib.rcParams["font.size"] = 10
    bins_mass = np.linspace(0, 2, 50)
    massPID, massPIDpandora, EPID, EPIDpandora, frac_E_model, frac_E_pandora, pid_true_over_true, mass_over_true_gt = get_mass_contribution_per_category(matched_pandora,
                                                             matched_,
                                                             matched_gt,
                                                             ML_pid=1,
                                                             perfect_pid=perfect_pid,
                                                             mass_zero=0, energy_bins=energy_bins)
   
    figs_mass_hist, axs_mass_hist = plt.subplots(3, 3, figsize=(8*3, 8*3))  # e, CH, NH, gamma
    for i, pid in enumerate([0, 1, 2]):
   
        axs_mass_hist[i, 0].hist(mass_over_true_gt[pid], bins=bins_mass, histtype="step", label="ML_GT", color="black",
                                density=True)
        
        axs_mass_hist[i, 0].hist(massPID[pid], bins=bins_mass, histtype="step", label="ML", color="red",
                                 density=True)
        axs_mass_hist[i, 0].hist(massPIDpandora[pid], bins=bins_mass, histtype="step", label="Pandora",
                                 color="blue",
                                 density=True)
        axs_mass_hist[i, 1].grid(1)
        pid_text = {0: "e+CH", 1: "NH+gamma", 2: "e+CH no track", 3: "gamma", 4: "mu"}.get(pid)
        axs_mass_hist[i, 1].set_xlabel(f"E pred., {pid_text} / E true, {pid_text}")
        axs_mass_hist[i, 1].hist(EPID[pid], bins=bins_mass, histtype="step", label="ML", color="red",
                                 density=True)
        axs_mass_hist[i, 1].hist(EPIDpandora[pid], bins=bins_mass, histtype="step", label="Pandora",
                                 color="blue",
                                 density=True)
        axs_mass_hist[i, 0].set_title(pid_text)
        axs_mass_hist[i, 1].set_title(pid_text)
        binsE = np.linspace(0, 1.0, 50)
        if energy_bins is not None and (energy_bins[0] == 0.0 and energy_bins[1] == 1.0):
            # [0, 1] GeV
            binsE=np.linspace(0, 0.15, 50)
        elif energy_bins is not None and (energy_bins[0] == 1.0 and energy_bins[1] == 10.0):
            binsE=np.linspace(0, 0.8, 50)
        axs_mass_hist[i, 0].legend()
        axs_mass_hist[i, 1].legend()
        axs_mass_hist[i, 2].grid(1)
        axs_mass_hist[i, 2].set_title(f"Pred. {pid_text} contr. to true evt. E")
        axs_mass_hist[i, 2].set_xlabel(f"Pred. E {pid_text} / Pred. E")
        axs_mass_hist[i, 2].hist(frac_E_model[pid], bins=binsE, histtype="step", label="ML", color="red",
                                 density=True)
        axs_mass_hist[i, 2].hist(frac_E_pandora[pid], bins=binsE, histtype="step", label="Pandora",
                                 color="blue", density=True)
        axs_mass_hist[i, 2].hist(pid_true_over_true[pid], bins=binsE, histtype="step", label="Truth",
                                 color="green", density=True)
        axs_mass_hist[i, 2].legend()
        axs_mass_hist[i, 0].set_yscale("log")
        axs_mass_hist[i, 1].set_yscale("log")
        axs_mass_hist[i, 2].set_yscale("log")
    figs_mass_hist.tight_layout()
    if energy_bins is not None:
        figs_mass_hist.savefig(os.path.join(path_store, "E_breakdown_by_PID_category_{}_{}_GeV.pdf".format(energy_bins[0], energy_bins[1])))
    else:
        figs_mass_hist.savefig(os.path.join(path_store, f"E_breakdown_by_PID_category.pdf"))
    matplotlib.rcParams["font.size"] = old_fontsize


def plot_per_energy_resolution2_multiple(
    matched_pandora, matched_all, PATH_store, tracks=False, perfect_pid=False, mass_zero=False, ML_pid=False, PATH_store_detailed_plots=None
):
    # matched_all: label -> matched df
    figs, axs = {}, {}  # resolution
    figs_r, axs_r = {}, {}
    figs_distr, axs_distr = {}, {}  # response
    figs_distr_HE, axs_distr_HE = {}, {}  # response high energy
    figs_theta_res, axs_theta_res = {}, {} # theta resolution
    figs_resolution_pxyz, axs_resolution_pxyz = {}, {} # px, py, pz resolution
    figs_response_pxyz, axs_response_pxyz = {}, {} # px, py, pz response
    # distribution at some energy slice for each particle (the little histogram plots)
    # colors = {"DNN": "green", "GNN+DNN": "purple", "DNN w/o FT": "blue"}
    colors = {
        "ML": "red",
        "ML GTC": "green"
    }
    plot_pandora, plot_baseline = True, True
    figs_mass_hist, axs_mass_hist = plt.subplots(6, 3, figsize=(8, 15)) # also plot fraction of the whole event energy
    (massPID, massPIDpandora, EPID, EPIDpandora, frac_E_model,
     frac_E_pandora, frac_E_model_true, frac_E_pandora_true,
     pid_true_over_true) = get_mass_contribution_per_PID(matched_pandora,
                                                         matched_all["ML"],
                                                         ML_pid=1,
                                                         perfect_pid=0,
                                                         mass_zero=0)
    
    bins_mass = np.linspace(0, 2, 200)
    for i, pid in enumerate([22, 11, 130, 211, 2112, 2212]):
        figs_theta_res[pid], axs_theta_res[pid] = plt.subplots(1, 1, figsize=(7, 7))
        figs[pid], axs[pid] = plt.subplots(2, 1, figsize=(15, 10), sharex=False)
        figs_r[pid], axs_r[pid] = plt.subplots(2, 1, figsize=(15, 10), sharex=False)
        figs_distr[pid], axs_distr[pid] = plt.subplots(1, 1, figsize=(7, 7))
        figs_distr_HE[pid], axs_distr_HE[pid] = plt.subplots(1, 1, figsize=(7, 7))
        figs_resolution_pxyz[pid], axs_resolution_pxyz[pid] = plt.subplots(4, 1, figsize=(8, 15), sharex=True)
        figs_response_pxyz[pid], axs_response_pxyz[pid] = plt.subplots(4, 1, figsize=(8, 15), sharex=True)
        #figs_mass_hist[pid], axs_mass_hist[pid] = plt.subplots(1, 1, figsize=(8, 20), sharex=False)
        axs_resolution_pxyz[pid][0].set_title(f"{pid} px resolution")
        axs_resolution_pxyz[pid][1].set_title(f"{pid} py resolution")
        axs_resolution_pxyz[pid][2].set_title(f"{pid} pz resolution")
        axs_resolution_pxyz[pid][3].set_title("p norm resolution [GeV]")
        axs_resolution_pxyz[pid][2].set_xlabel("Energy [GeV]")
        axs_response_pxyz[pid][0].set_title(f"{pid} px response")
        axs_response_pxyz[pid][1].set_title(f"{pid} py response")
        axs_response_pxyz[pid][2].set_title(f"{pid} pz response")
        axs_response_pxyz[pid][3].set_title("p norm response [GeV]")
        axs_response_pxyz[pid][2].set_xlabel("Energy [GeV]")
        axs_mass_hist[i, 0].grid(1)
        #axs_mass_hist[i, 0].set_title(f"{pid} contribution to mass")
        axs_mass_hist[i, 0].set_xlabel(f"M pred., {pid} / M true, {pid}")
        axs_mass_hist[i, 0].hist(massPID[pid], bins=bins_mass, histtype="step", label="ML", color="red", density=True)
        axs_mass_hist[i, 0].hist(massPIDpandora[pid], bins=bins_mass, histtype="step", label="Pandora", color="blue", density=True)
        # same with energy into [i, 1]
        axs_mass_hist[i, 1].grid(1)
        #axs_mass_hist[i, 1].set_title(f"{pid} contribution to E")
        axs_mass_hist[i, 0].set_xlabel(f"m pred., {pid} / m true, {pid}")
        axs_mass_hist[i, 1].set_xlabel(f"E pred., {pid} / E true, {pid}")
        axs_mass_hist[i, 1].hist(EPID[pid], bins=bins_mass, histtype="step", label="ML", color="red", density=True)
        axs_mass_hist[i, 1].hist(EPIDpandora[pid], bins=bins_mass, histtype="step", label="Pandora", color="blue", density=True)
        binsE = np.linspace(0, 1.0, 100)
        axs_mass_hist[i, 0].legend()
        axs_mass_hist[i, 1].legend()
        axs_mass_hist[i, 2].grid(1)
        axs_mass_hist[i, 2].set_title(f"Pred. {pid} contr. to pred. evt. E")
        axs_mass_hist[i, 2].set_xlabel(f"Pred. E {pid} / Pred. E")
        axs_mass_hist[i, 2].hist(frac_E_model[pid], bins=binsE, histtype="step", label="ML", color="red", density=True)
        axs_mass_hist[i, 2].hist(frac_E_pandora[pid], bins=binsE, histtype="step", label="Pandora", color="blue", density=True)
        axs_mass_hist[i, 2].hist(pid_true_over_true[pid], bins=binsE, histtype="step", label="Truth", color="green", density=True)
        axs_mass_hist[i, 2].legend()
        axs_mass_hist[i, 0].set_yscale("log")
        axs_mass_hist[i, 1].set_yscale("log")
        axs_mass_hist[i, 2].set_yscale("log")
    event_res_dic = {} # Event energy resolution
    figs_mass_hist.tight_layout()
    figs_mass_hist.savefig(
        os.path.join(PATH_store_detailed_plots, f"mass_breakdown_by_PID.pdf"),
        bbox_inches="tight",
    )
    #############################################################################
    fig_event_res, ax_event_res = plt.subplots(1, 1, figsize=(7, 7))
    PIDs = [22, 11, 130, 211, 2112, 2212]
    fig_distr, ax_distr = plt.subplots(len(PIDs), 3, figsize=(len(PIDs) * 5, 18))
    matplotlib.rcParams["font.size"] = 11
    fig_distr_reco, ax_distr_reco = plt.subplots(len(PIDs), 3, figsize=(len(PIDs)*3, 9*3))
    photons_dic_all, electrons_dic_all, hadrons_dic_all, hadrons_dic2_all, neutrons_all, protons_all = {}, {}, {}, {}, {}, {}
   
    for key in matched_all:
        matched_ = matched_all[key]
        #mask = matched_["calibration_factor"] > 0
        #matched_ = matched_[mask]
        if tracks:
            tracks_label = "tracks"
        else:
            tracks_label = ""
        plot_response = True
        if plot_response:
            list_plots = [""]  # "","_reco"
            
            event_res_dic[key] = get_response_for_event_energy(
                matched_pandora, matched_, perfect_pid=perfect_pid, mass_zero=mass_zero, ML_pid=ML_pid
            )
            print("PID:22___________________________________________")
            photons_dic = get_response_for_id_i(
                [22], matched_pandora, matched_, tracks=tracks, perfect_pid=perfect_pid, mass_zero=mass_zero,
                ML_pid=ML_pid
            )
            np.save(PATH_store+"/22.npy", photons_dic)
            print("PID:11___________________________________________")
            electrons_dic = get_response_for_id_i(
                [11], matched_pandora, matched_, tracks=tracks, perfect_pid=perfect_pid , mass_zero=mass_zero, ML_pid=ML_pid
            )
            np.save(PATH_store+"/11.npy", electrons_dic)
            print("PID:130___________________________________________")
            hadrons_dic = get_response_for_id_i(
                [130], matched_pandora, matched_, tracks=tracks, perfect_pid=perfect_pid, mass_zero=mass_zero, ML_pid=ML_pid
            )
            np.save(PATH_store+"/130.npy", hadrons_dic)
            print("PID:211___________________________________________")
            hadrons_dic2 = get_response_for_id_i(
                [211], matched_pandora, matched_, tracks=tracks, perfect_pid=perfect_pid, mass_zero=mass_zero, ML_pid=ML_pid
            )
            np.save(PATH_store+"/211.npy", hadrons_dic2)
            print("PID:neutron___________________________________________")
            neutrons = get_response_for_id_i(
                [2112], matched_pandora, matched_, tracks=tracks, perfect_pid=perfect_pid, mass_zero=mass_zero, ML_pid=ML_pid
            )
            np.save(PATH_store+"/2112.npy", neutrons)
            print("PID:proton___________________________________________")
            protons = get_response_for_id_i(
                [2212], matched_pandora, matched_, tracks=tracks, perfect_pid=perfect_pid, mass_zero=mass_zero, ML_pid=ML_pid
            )
            np.save(PATH_store+"2212.npy", protons)
            print(PATH_store+"/2212.npy")
            hadrons_dic_all[key] = hadrons_dic
            hadrons_dic2_all[key] = hadrons_dic2
            photons_dic_all[key] = photons_dic
            electrons_dic_all[key] = electrons_dic
            neutrons_all[key] = neutrons
            protons_all[key] = protons


            # For neutrons
            if True:
                if len(neutrons["distributions_pandora"]) :
                    plot_hist_distr(neutrons["distributions_pandora"][0], "Pandora", axs_distr[2112], "blue")
                if len(hadrons_dic["distributions_pandora"]):
                    # Same for 130
                    plot_hist_distr(hadrons_dic["distributions_pandora"][0], "Pandora", axs_distr[130], "blue")
                mean_e_over_true_pandora, sigma_e_over_true_pandora = round(event_res_dic["ML"]["mean_energy_over_true_pandora"], 2), round(event_res_dic["ML"]["var_energy_over_true_pandora"], 2)
                mean_e_over_true, sigma_e_over_true = round(event_res_dic["ML"]["mean_energy_over_true"], 2), round(event_res_dic["ML"]["var_energy_over_true"], 2)
                ax_event_res.hist(event_res_dic["ML"]["energy_over_true_pandora"], bins=np.linspace(0.5, 1.5, 100), histtype="step",
                                  label=r"Pandora $\mu$={} $\sigma / \mu$={}".format(mean_e_over_true_pandora, sigma_e_over_true_pandora), color="blue", density=True)
                ax_event_res.hist(event_res_dic[key]["energy_over_true"], bins=np.linspace(0.5, 1.5, 100), histtype="step",
                                  label=str(key) + r" $\mu$={} $\sigma / \mu$={}".format(mean_e_over_true, sigma_e_over_true), color=colors[key], density=True)
                ax_event_res.grid(1)
                ax_event_res.set_xlabel(r"$E_{vis,pred} / E_{vis,true}$")
                ax_event_res.legend()
                fig_event_res.savefig(
                    os.path.join(PATH_store_detailed_plots, "total_visible_energy_resolution.pdf"), bbox_inches="tight"
                )
                # for pions 211
                if len(hadrons_dic2["distributions_pandora"]):
                    plot_hist_distr(hadrons_dic2["distributions_pandora"][0], "Pandora", axs_distr[211], "blue")
                # same for 11
                if len(electrons_dic["distributions_pandora"]):
                    plot_hist_distr(electrons_dic["distributions_pandora"][0], "Pandora", axs_distr[11], "blue")
                # same for 2212
                if len(protons["distributions_pandora"]) > 0:
                    plot_hist_distr(protons["distributions_pandora"][0], "Pandora", axs_distr[2212], "blue", bins=np.linspace(0.5, 1.1, 200))
                    plot_hist_distr(protons["distributions_pandora"][-1], "Pandora", axs_distr_HE[2212], "blue",
                                    bins=np.linspace(0.9, 1.1, 100))
                if len(photons_dic["distributions_pandora"]) > 0:
                    bins=np.linspace(0, 5, 100)
                    plot_hist_distr(photons_dic["distributions_pandora"][0], "Pandora", axs_distr[22], "blue")
                    distances_p = np.linalg.norm(photons_dic["pxyz_true_p"][0] - photons_dic["pxyz_pred_p"][0], axis=1)
                    distances = np.linalg.norm(photons_dic["pxyz_true"][0] - photons_dic["pxyz_pred"][0], axis=1)
                    fig, ax = plt.subplots()
                    ax.hist(distances_p, bins=bins, histtype="step", label="Pandora", color="blue", density=True)
                    ax.hist(distances, bins=bins, histtype="step", label="ML", color="red", density=True)
                    ax.legend()
                    ax.grid(1)
                    ax.set_yscale("log")
                    ax.set_title("Photons p distance from truth [0,5] GeV")
                    fig.tight_layout()
                    fig.savefig(PATH_store + "/Photons_p_distance.pdf", bbox_inches="tight")
            if len(neutrons["distributions_model"]) > 0:
                plot_hist_distr(neutrons["distributions_model"][0], key, axs_distr[2112], colors[key])
            axs_distr[2112].set_title("Neutrons [0, 5] GeV")
            axs_distr[2112].set_xlabel("$E_{pred.} / E_{true}$")
            # y label density
            axs_distr[2112].set_ylabel("Density")
            axs_distr[2112].legend()
            if len(protons["distributions_model"]) > 0:
                plot_hist_distr(protons["distributions_model"][0], key, axs_distr[2212], colors[key], bins=np.linspace(0.5, 1.1, 200))
                axs_distr[2212].set_title("Protons [0, 5] GeV")
                axs_distr[2212].set_xlabel("$E_{pred.} / E_{true}$")
                axs_distr[2212].set_ylabel("Density")
                axs_distr[2212].legend()
            if len(hadrons_dic["distributions_model"]) > 0:
                plot_hist_distr(hadrons_dic["distributions_model"][0], key, axs_distr[130], colors[key])
            axs_distr[130].set_ylabel("Density")
            axs_distr[130].set_title("$K_L$ [0, 5] GeV")
            axs_distr[130].set_xlabel("$E_{pred.} / E_{true}$")
            axs_distr[130].legend()
            if len(hadrons_dic2["distributions_model"]) > 0:
                plot_hist_distr(hadrons_dic2["distributions_model"][0], key, axs_distr[211], colors[key])
                axs_distr[211].set_ylabel("Density")
                axs_distr[211].set_title("Pions [0, 5] GeV")
                axs_distr[211].set_xlabel("$E_{pred.} / E_{true}$")
                axs_distr[211].legend()
                axs_distr[211].set_yscale("log")
                plot_hist_distr(hadrons_dic2["distributions_model"][0], key, axs_distr[11], colors[key])
            axs_distr[11].set_ylabel("Density")
            axs_distr[11].set_title("Electrons [0, 5] GeV")
            axs_distr[11].set_xlabel("$E_{pred.} / E_{true}$")
            axs_distr[11].legend()
            axs_distr[11].set_yscale("log")
            if len(photons_dic["distributions_model"]) > 0:
                plot_hist_distr(photons_dic["distributions_model"][0], key, axs_distr[22], colors[key])
            axs_distr[22].set_title("Photons [0, 5] GeV")
            axs_distr[22].set_xlabel("$E_{pred.} / E_{true}$")
            axs_distr[22].set_ylabel("Density")
            axs_distr[22].legend()
            if len(neutrons["distributions_model"]) > 0:
                plot_hist_distr(neutrons["distributions_model"][-1], key, axs_distr_HE[2212], colors[key], bins=np.linspace(0.9, 1.1, 100))
            axs_distr_HE[2112].set_title("Protons [35, 50] GeV")
            axs_distr_HE[2112].set_xlabel("$E_{pred.} / E_{true}$")
            axs_distr_HE[2112].set_ylabel("Density")
            axs_distr_HE[2112].legend()
            if len(neutrons["distributions_model"]) > 0:
                plot_hist_distr(neutrons["distributions_model"][-1], key, axs_distr_HE[130], colors[key], bins=np.linspace(0.9, 1.1, 100))
            axs_distr_HE[130].set_title("K_L [35, 50] GeV")
            axs_distr_HE[130].set_xlabel("$E_{pred.} / E_{true}$")
            axs_distr_HE[130].set_ylabel("Density")
            axs_distr_HE[130].legend()

            '''
            (
                "Event Energy Resolution",
                event_res_dic[key],
                fig_event_res,
                ax_event_res,
                plot_pandora,
                prefix=key + " ",
                color=colors[key],
            )'''

            neutral_masses = [0.0, 497.611/1000, 0.939565]
            charged_masses = [0.139570, 0.511/1000]
            neutral_masses = [x**2 for x in neutral_masses]
            charged_masses = [x**2 for x in charged_masses]
            for el in list_plots:
                #if len(photons_dic["mass_histogram"]) > 2:
                #    plot_pxyz_resolution(photons_dic["energy_resolutions"], photons_dic["variance_om_pxyz_pandora"], photons_dic["variance_om_pxyz"], axs_resolution_pxyz[22], key)
                #    plot_pxyz_resolution(photons_dic["energy_resolutions"], photons_dic["mean_pxyz_pandora"],
                #                         photons_dic["mean_pxyz"], axs_response_pxyz[22], key)
                #    plot_mass_hist(photons_dic["mass_histogram"], photons_dic["mass_histogram_pandora"], axs_mass_hist[22], bars=neutral_masses)
                #if len(neutrons["mass_histogram"]) > 2:
                #    plot_pxyz_resolution(neutrons["energy_resolutions"], neutrons["variance_om_pxyz_pandora"], neutrons["variance_om_pxyz"], axs_resolution_pxyz[2112], key)
                #if len(hadrons_dic["mass_histogram"]) > 2:
                #    #plot_pxyz_resolution(hadrons_dic["energy_resolutions"], hadrons_dic["variance_om_pxyz_pandora"], hadrons_dic["variance_om_pxyz"], axs_resolution_pxyz[130], key)
                #if len(hadrons_dic2["mass_histogram"]) > 2:
                #    #plot_mass_hist(hadrons_dic["mass_histogram"], hadrons_dic["mass_histogram_pandora"], axs_mass_hist[130], bars=neutral_masses)
                #if len(hadrons_dic2["energy_resolutions"]) > 1:
                ##    #plot_pxyz_resolution(hadrons_dic2["energy_resolutions"], hadrons_dic2["variance_om_pxyz_pandora"], hadrons_dic2["variance_om_pxyz"], axs_resolution_pxyz[211], key)
                #    #plot_pxyz_resolution(hadrons_dic2["energy_resolutions"], hadrons_dic2["mean_pxyz_pandora"],
                #    #                     hadrons_dic2["mean_pxyz"], axs_response_pxyz[211], key)
                #    #plot_mass_hist(hadrons_dic2["mass_histogram"], hadrons_dic2["mass_histogram_pandora"], axs_mass_hist[211], bars=charged_masses)
                '''if len(electrons_dic["energy_resolutions"]) > 1 and len(electrons_dic["mass_histogram"]) > 2:
                    plot_pxyz_resolution(electrons_dic["energy_resolutions"], electrons_dic["variance_om_pxyz_pandora"], electrons_dic["variance_om_pxyz"], axs_resolution_pxyz[11], key)
                    plot_pxyz_resolution(electrons_dic["energy_resolutions"], electrons_dic["mean_pxyz_pandora"],
                                         electrons_dic["mean_pxyz"], axs_response_pxyz[11], key)
                    plot_mass_hist(electrons_dic["mass_histogram"], electrons_dic["mass_histogram_pandora"], axs_mass_hist[11], bars=charged_masses)
                if len(neutrons["energy_resolutions"]) > 2:
                    plot_pxyz_resolution(neutrons["energy_resolutions"], neutrons["mean_pxyz_pandora"], neutrons["mean_pxyz"], axs_response_pxyz[2112], key)
                #plot_pxyz_resolution(event_res_dic[key]["energy_resolutions"], protons["variance_om_pxyz_pandora"], protons["variance_om_pxyz_pandora"], axs_resolution_pxyz[2212], key)
                # same but for response instead of resolution. use "mean_pxyz" instead of "variance_om_pxyz"
                if len(neutrons["energy_resolutions"]) > 2:
                    plot_pxyz_resolution(neutrons["energy_resolutions"], neutrons["mean_pxyz_pandora"], neutrons["mean_pxyz"], axs_response_pxyz[2112], key)
                if len(neutrons["mass_histogram"]) > 2:
                    plot_mass_hist(neutrons["mass_histogram"], neutrons["mass_histogram_pandora"], axs_mass_hist[2112], bars=neutral_masses)
                if len(hadrons_dic["energy_resolutions"]) > 0:
                    plot_pxyz_resolution(hadrons_dic["energy_resolutions"], hadrons_dic["mean_pxyz_pandora"], hadrons_dic["mean_pxyz"], axs_response_pxyz[130], key)'''
                #plot_pxyz_resolution(event_res_dic[key]["energy_resolutions"], protons["mean_pxyz_pandora"], protons["mean_pxyz"], axs_response_pxyz[2212], key)
                fig_phi, ax_phi = plt.subplots(len(PIDs), 3, figsize=(len(PIDs)*2.5, 10))
                fig_theta, ax_theta = plt.subplots(len(PIDs), 3, figsize=(len(PIDs)*2.5, 10))
                # matplotlib.rcParams["font.size"] = 10
                fig_all_angles, ax_all_angles = plt.subplots(5, 2, figsize=(8, 14))  # For the total energy resolution
                for j, angle in enumerate(["theta", "phi"]):
                    if len(photons_dic["distr_phi"]) > 0:
                        stacked_hist_plot(photons_dic["distr_phi"], photons_dic["distr_phi_pandora"], PATH_store, r"Photons $\Phi$", "Photons_Phi", ax=ax_phi[PIDs.index(22)])
                        stacked_hist_plot(photons_dic["distr_theta"], photons_dic["distr_theta_pandora"], PATH_store, r"Photons $\theta$", "Photons_Theta", ax=ax_theta[PIDs.index(22)])
                        ax_angle = ax_all_angles[PIDs.index(22), j]
                        #plot_sigma_angle_vs_energy(photons_dic, PATH_store, "photons", angle, "$\gamma$", ax=ax_angle)
                        ax_angle.grid(1)
                    if len(neutrons["distr_phi"]) > 0:
                        stacked_hist_plot(neutrons["distr_phi"], neutrons["distr_phi_pandora"], PATH_store, "Neutrons $\Phi$", "Neutrons_Phi", ax=ax_phi[PIDs.index(2112)])
                        stacked_hist_plot(neutrons["distr_theta"], neutrons["distr_theta_pandora"], PATH_store, "Neutrons $\Theta$", "Neutrons_Theta", ax=ax_theta[PIDs.index(2112)])
                        ax_angle = ax_all_angles[PIDs.index(2112), j]
                        #plot_sigma_angle_vs_energy(neutrons, PATH_store, "neutrons", angle, "Neutrons", ax=ax_angle)
                        ax_angle.grid(1)
                    if len(hadrons_dic["distr_phi"]) > 0:
                        ax_angle = ax_all_angles[PIDs.index(130), j]
                        stacked_hist_plot(hadrons_dic["distr_phi"], hadrons_dic["distr_phi_pandora"], PATH_store, r"K_L $\Phi$", "KL_Phi", ax=ax_phi[PIDs.index(130)])
                        stacked_hist_plot(hadrons_dic["distr_theta"], hadrons_dic["distr_theta_pandora"], PATH_store, r"K_L $\theta$", "KL_Theta", ax=ax_theta[PIDs.index(130)])
                        #plot_sigma_angle_vs_energy(hadrons_dic, PATH_store, "KL", angle, "$K_L$", ax=ax_angle)
                        ax_angle.grid(1)
                    if len(hadrons_dic2["distr_phi"]) > 0:
                        stacked_hist_plot(hadrons_dic2["distr_phi"], hadrons_dic2["distr_phi_pandora"], PATH_store, r"$\Pi^{\pm}$ $\Phi$", "Pions_Phi", ax=ax_phi[PIDs.index(211)], narrow=True)
                        stacked_hist_plot(hadrons_dic2["distr_theta"], hadrons_dic2["distr_theta_pandora"], PATH_store, r"$\Pi^{\pm}$ $\theta$", "Pions_Theta", ax=ax_theta[PIDs.index(211)], narrow=True)
                        ax_angle = ax_all_angles[PIDs.index(211), j]
                        #plot_sigma_angle_vs_energy(hadrons_dic2, PATH_store, "Pions", angle, "Pions", ax=ax_angle)
                        ax_angle.grid(1)
                    if len(electrons_dic["distr_phi"]) > 0:
                        stacked_hist_plot(electrons_dic["distr_phi"], electrons_dic["distr_phi_pandora"], PATH_store,r"e $\Phi$", "Electrons_Phi", ax=ax_phi[PIDs.index(11)], narrow=True)
                        stacked_hist_plot(electrons_dic["distr_theta"], electrons_dic["distr_theta_pandora"], PATH_store, r"e $\theta$", "Electrons_Theta", ax=ax_theta[PIDs.index(11)], narrow=True)
                        ax_angle = ax_all_angles[PIDs.index(11), j]
                        #plot_sigma_angle_vs_energy(electrons_dic, PATH_store, "electrons", angle, "Electrons", ax=ax_angle)
                        ax_angle.grid(1)
                fig_theta.tight_layout()
                fig_phi.tight_layout()
                fig_theta.savefig(os.path.join(PATH_store_detailed_plots, "theta.pdf"), bbox_inches="tight")
                fig_phi.savefig(os.path.join(PATH_store_detailed_plots, "phi.pdf"), bbox_inches="tight")
                #fig_all_angles.tight_layout()
                #fig_all_angles.savefig(os.path.join(PATH_store_detailed_plots, "Angular_Resolution_per_PID.pdf"), bbox_inches="tight")
                if len(photons_dic["energy_resolutions"]) > 1:
                    plot_one_label(
                        "Electromagnetic Resolution",
                        photons_dic,
                        "variance_om",
                        PATH_store,
                        "Photons",
                        el,
                        tracks=tracks_label,
                        fig=figs[22],
                        ax=axs[22],
                        save=False,
                        plot_pandora=plot_pandora,
                        plot_baseline=plot_baseline,
                        color=colors[key],
                        pandora_label="Pandora",
                        ax_distr=ax_distr[PIDs.index(22)],
                        ax_distr_reco=ax_distr_reco[PIDs.index(22)],
                    )
                    plot_one_label(
                        "Electromagnetic Response",
                        photons_dic,
                        "mean",
                        PATH_store,
                        "Photons",
                        el,
                        tracks=tracks_label,
                        fig=figs_r[22],
                        ax=axs_r[22],
                        save=False,
                        plot_pandora=plot_pandora,
                        plot_baseline=plot_baseline,
                        color=colors[key],
                        pandora_label="Pandora",
                    )
                if len(electrons_dic["energy_resolutions"]) > 2:
                    plot_one_label(
                        "Electromagnetic Response",
                        electrons_dic,
                        "mean",
                        PATH_store,
                        "Electrons",
                        el,
                        tracks=tracks_label,
                        fig=figs_r[11],
                        ax=axs_r[11],
                        save=False,
                        plot_pandora=plot_pandora,
                        plot_baseline=plot_baseline,
                        color=colors[key],
                        pandora_label="Pandora"
                    )
                    plot_one_label(
                        "Electromagnetic Resolution",
                        electrons_dic,
                        "variance_om",
                        PATH_store,
                        "Electrons",
                        el,
                        tracks=tracks_label,
                        fig=figs[11],
                        ax=axs[11],
                        save=False,
                        plot_pandora=plot_pandora,
                        plot_baseline=plot_baseline,
                        color=colors[key],
                        pandora_label="Pandora",
                        ax_distr=ax_distr[PIDs.index(11)],
                        ax_distr_reco=ax_distr_reco[PIDs.index(11)],
                    )
                if len(hadrons_dic["energy_resolutions"]) > 1:
                    plot_one_label(
                        "Hadronic Resolution",
                        hadrons_dic,
                        "variance_om",
                        PATH_store,
                        "K long",
                        el,
                        tracks=tracks_label,
                        fig=figs[130],
                        ax=axs[130],
                        save=False,
                        plot_pandora=plot_pandora,
                        plot_baseline=plot_baseline,
                        color=colors[key],
                        pandora_label="Pandora",
                        ax_distr=ax_distr[PIDs.index(130)],
                        ax_distr_reco=ax_distr_reco[PIDs.index(130)],
                    )
                    plot_one_label(
                        "Hadronic Response",
                        hadrons_dic,
                        "mean",
                        PATH_store,
                        "K long",
                        el,
                        tracks=tracks_label,
                        fig=figs_r[130],
                        ax=axs_r[130],
                        save=False,
                        plot_pandora=plot_pandora,
                        plot_baseline=plot_baseline,
                        color=colors[key],
                        pandora_label="Pandora"
                    )
                if len(hadrons_dic2["mean_baseline"]) > 1: # if there are pions in dataset
                    plot_one_label(
                        "Hadronic Resolution",
                        hadrons_dic2,
                        "variance_om",
                        PATH_store,
                        "pi+",
                        el,
                        tracks=tracks_label,
                        fig=figs[211],
                        ax=axs[211],
                        save=False,
                        plot_pandora=plot_pandora,
                        plot_baseline=plot_baseline,
                        color=colors[key],
                        pandora_label="Pandora",
                        ax_distr=ax_distr[PIDs.index(211)],
                        ax_distr_reco=ax_distr_reco[PIDs.index(211)],
                    )
                    plot_one_label(
                        "Hadronic Response",
                        hadrons_dic2,
                        "mean",
                        PATH_store,
                        "pi+",
                        el,
                        tracks=tracks_label,
                        fig=figs_r[211],
                        ax=axs_r[211],
                        save=False,
                        plot_pandora=plot_pandora,
                        plot_baseline=plot_baseline,
                        color=colors[key],
                        pandora_label="Pandora"
                    )
                # plot the neutrons and protons
                if len(neutrons["mean_baseline"]) > 2: # If there are neutrons in dataset
                    plot_one_label(
                        "Hadronic Resolution",
                        neutrons,
                        "variance_om",
                        PATH_store,
                        "Neutrons",
                        el,
                        tracks=tracks_label,
                        fig=figs[2112],
                        ax=axs[2112],
                        save=False,
                        plot_pandora=plot_pandora,
                        plot_baseline=plot_baseline,
                        color=colors[key],
                        pandora_label="Pandora",
                        ax_distr=ax_distr[PIDs.index(2112)],
                        ax_distr_reco=ax_distr_reco[PIDs.index(2112)],
                    )
                    plot_one_label(
                        "Hadronic Response",
                        neutrons,
                        "mean",
                        PATH_store,
                        "Neutrons",
                        el,
                        tracks=tracks_label,
                        fig=figs_r[2112],
                        ax=axs_r[2112],
                        save=False,
                        plot_pandora=plot_pandora,
                        plot_baseline=plot_baseline,
                        color=colors[key],
                        pandora_label="Pandora"
                    )
                    plot_one_label(
                        "Hadronic Response",
                        neutrons,
                        "mean",
                        PATH_store,
                        "Protons",
                        el,
                        tracks=tracks_label,
                        fig=figs_r[2212],
                        ax=axs_r[2212],
                        save=False,
                        plot_pandora=plot_pandora,
                        plot_baseline=plot_baseline,
                        color=colors[key],
                        pandora_label="Pandora"
                    )
                if len(protons["mean_baseline"]) > 2: # if there are protons in dataset
                    plot_one_label(
                        "Hadronic Resolution",
                        protons,
                        "variance_om",
                        PATH_store,
                        "Protons",
                        el,
                        tracks=tracks_label,
                        fig=figs[2212],
                        ax=axs[2212],
                        save=False,
                        plot_pandora=plot_pandora,
                        plot_baseline=plot_baseline,
                        color=colors[key],
                        pandora_label="Pandora",
                        ax_distr=ax_distr[PIDs.index(2212)],
                        ax_distr_reco=ax_distr_reco[PIDs.index(2212)],
                    )
                plot_pandora = False
                plot_baseline = False
    plot_mass_resolution(event_res_dic, PATH_store_detailed_plots)
    plot_full_comparison(photons_dic_all, electrons_dic_all, hadrons_dic_all, hadrons_dic2_all, neutrons_all,
                         protons_all, os.path.join(PATH_store_detailed_plots, "E_resolution_per_PID_full_comparison.pdf"))
    for key in figs:
        for a in axs[key]:
            a.grid(1)
        for a in axs_r[key]:
            a.grid(1)
        axs_distr[key].grid(1)
        figs[key].tight_layout()
        figs_resolution_pxyz[key].tight_layout()
        Path(os.path.join(PATH_store, "p_resolutions")).mkdir(parents=True, exist_ok=True)
        figs_resolution_pxyz[key].savefig(
            os.path.join(PATH_store, "p_resolutions", f"resolution_pxyz_{key}.pdf"),
            bbox_inches="tight",
        )
        figs_response_pxyz[key].tight_layout()
        Path(os.path.join(PATH_store, "p_response")).mkdir(parents=True, exist_ok=True)
        figs_response_pxyz[key].savefig(
            os.path.join(PATH_store, "p_response", f"response_pxyz_{key}.pdf"),
            bbox_inches="tight",
        )
        figs[key].savefig(
            os.path.join(PATH_store, f"comparison_resolution_{key}.pdf"),
            bbox_inches="tight",
        )
        figs_r[key].tight_layout()
        figs_r[key].savefig(
            os.path.join(PATH_store, f"comparison_response_{key}.pdf"),
            bbox_inches="tight",
        )
        figs_distr[key].tight_layout()
        figs_distr[key].savefig(
            os.path.join(PATH_store, f"distr_5_10_GeV_{key}.pdf"),
            bbox_inches="tight",
        )
        figs_distr_HE[key].tight_layout()
        figs_distr_HE[key].savefig(
            os.path.join(PATH_store, f"distr_35_50_GeV_{key}.pdf"),
            bbox_inches="tight",
        )
        fig_distr.tight_layout()
        fig_distr.savefig(
            os.path.join(PATH_store_detailed_plots, "energy_resolution_histograms.pdf")
        )
        fig_distr_reco.tight_layout()
        fig_distr_reco.savefig(
            os.path.join(PATH_store_detailed_plots, "energy_resolution_histograms_reco.pdf")
        )
        dist_pandora, pids, phi_dist_pandora, eta_dist_pandora = calc_unit_circle_dist(matched_pandora, pandora=True)
        dist_ml, pids_ml, phi_dist_ml, eta_dist_ml = calc_unit_circle_dist(matched_, pandora=False)
        for pid in [22, -211, 211, 2112, 130, 11]:
            # plot histogram
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            figphi, axphi = plt.subplots(1, 1, figsize=(10, 10))
            figeta, axeta = plt.subplots(1, 1, figsize=(10, 10))
            bins = np.linspace(0, 1, 100)
            bins_log = np.linspace(-5, 0, 100)
            bins_phi = np.linspace(-0.1, 0.1, 200)
            mu, var, _ , _ = get_sigma_gaussian((dist_pandora[np.where(pids == pid)]), bins)
            ax.hist(
                np.log10(dist_pandora[np.where(pids == pid)]),
                bins=bins_log,
                histtype="step",
                label="Pandora $\mu$={} $\sigma/\mu$={}".format(
                round(mu, 2),
                      round(var, 2)),
                color="blue",
            )
            mu, var, _ , _  = get_sigma_gaussian((dist_ml[np.where(pids_ml == pid)]), bins_phi)
            ax.hist(
                np.log10(dist_ml[np.where(pids_ml == pid)]),
                bins=bins_log,
                histtype="step",
                label="Model $\mu$={} $\sigma/\mu$={}".format(
                round(mu, 2),
                round(var, 2)),
                color="red",
            )
            mu, var, _ , _  = get_sigma_gaussian((phi_dist_pandora[np.where(pids == pid)]), bins_phi)
            var *= mu
            axphi.hist(
                phi_dist_pandora[np.where(pids == pid)],
                bins=bins_phi,
                histtype="step",
                label="Pandora $\sigma={}$".format(
                round(var, 4)),
                color="blue",
            )
            mu, var, _ , _  = get_sigma_gaussian((phi_dist_ml[np.where(pids_ml == pid)]), bins_phi)
            var_phi_model = var * mu
            axphi.hist(
                phi_dist_ml[np.where(pids_ml == pid)],
                bins=bins_phi,
                histtype="step",
                label=r"Model $\sigma={}$".format(
                round(var_phi_model, 4)),
                color="red",
            )
            ax.set_xlabel("log norm $n-n_{pred}$")
            ax.set_yscale("log")
            ax.legend()
            ax.grid(True)
            fig.savefig(os.path.join(PATH_store, f"unit_circle_dist_{pid}.pdf"))
            axphi.set_xlabel(r"$\Delta \Phi$")
            axphi.set_yscale("log")
            axphi.legend()
            axphi.grid(True)
            figphi.savefig(os.path.join(PATH_store, f"phi_dist_{pid}.pdf"))
            mu, var, _ ,_ = get_sigma_gaussian((eta_dist_pandora[np.where(pids == pid)]), bins_phi)
            # also for eta dist
            var_eta_pandora = var*mu
            mu, var, _, _ = get_sigma_gaussian((eta_dist_ml[np.where(pids_ml == pid)[0]]), bins_phi)
            # also for eta dist
            var_eta_model = var * mu
            axeta.hist(
                eta_dist_pandora[np.where(pids == pid)],
                bins=bins_phi,
                histtype="step",
                label="Pandora $\sigma$={}".format(
                 round(var_eta_pandora, 4)),
                color="blue",
            )
            axeta.hist(
                eta_dist_ml[np.where(pids_ml == pid)],
                bins=bins_phi,
                histtype="step",
                label="Model $\sigma$={}".format(
                 round(var_eta_model, 4)),
                color="red",
            )
            axeta.set_xlabel(r"$\Delta \theta$")
            axeta.set_yscale("log")
            axeta.legend()
            axeta.grid(True)
            figeta.savefig(os.path.join(PATH_store, f"eta_dist_{pid}.pdf"))

    #fig_mass_res.savefig(
    #    os.path.join(PATH_store, "event_mass_resolution.pdf"), bbox_inches="tight"
    #)

def reco_hist(ml, pandora, PATH_store, pids=[22, 130, 2112, 211, 2212, 310]):
    e_bins = [[0,5], [5, 15], [15, 50]]
    path_reco = os.path.join(PATH_store, "reco_histograms")
    if not os.path.exists(path_reco):
        os.makedirs(path_reco)
    for pid in pids:
        # make n rows, where n is the number of energy bins
        fig, ax = plt.subplots(len(e_bins), 1, figsize=(15, 10))
        for i, bin in enumerate(e_bins):
            filt_ml = (ml.pid==pid) & (ml.true_showers_E < bin[1]) & (ml.true_showers_E >= bin[0])
            filt_pandora = (pandora.pid==pid) & (pandora.true_showers_E < bin[1]) & (pandora.true_showers_E >= bin[0])
            reco_ml = ml.pred_showers_E[filt_ml] / ml.reco_showers_E[filt_ml]
            reco_pandora = pandora.pandora_calibrated_pfo[filt_pandora] / pandora.true_showers_E[filt_pandora]
            bins = np.linspace(0, 3,300)
            if i == 0 and pid == 22:
                fig1, ax1 = plt.subplots()
                ax1.hist(reco_ml, bins=bins, histtype='step', label='ML', color='red', density=True)
                ax1.hist(reco_pandora, bins=bins, histtype='step', label='Pandora', color='blue', density=True)
                ax1.set_xlabel(r"$E_{reco, pred.}/E_{reco, true}$")
                ax1.set_ylabel("Density")
                ax1.set_title("Photons [0, 5] GeV")
                ax1.set_yscale("log")
                ax1.legend()
                fig1.savefig(os.path.join(PATH_store, "reco_hist_photons_5GeV.pdf"))
            #filt_ml = (ml.pid == 130) & (ml.true_showers_E < 5)
            #filt_pandora = (pandora.pid == 130) & (pandora.true_showers_E < 5)
            #reco_ml = ml.pred_showers_E[filt_ml] / ml.reco_showers_E[filt_ml]
            #reco_pandora = pandora.pandora_calibrated_pfo[filt_pandora] / pandora.true_showers_E[filt_pandora]
            #bins = np.linspace(0, 2, 200)
            #fig, ax = plt.subplots()
            ax[i].hist(reco_ml, bins=bins, histtype='step', label='ML', color='red', density=True)
            ax[i].hist(reco_pandora, bins=bins, histtype='step', label='Pandora', color='blue', density=True)
            ax[i].set_xlabel(r"$E_{reco, pred.}/E_{reco, true}$")
            ax[i].set_ylabel("Density")
            ax[i].set_title("PID: {}, E range: {} GeV".format(pid, bin))
            ax[i].set_yscale("log")
            ax[i].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(path_reco, "reco_hist_{}.pdf".format(pid)))


def plot_per_energy_resolution(
     matched_pandora, matched_, PATH_store, tracks=False
):
    plot_response = True
    if plot_response:
        list_plots = ["_reco"]  # "","_reco"
        for el in list_plots:
            colors_list = ["#fde0dd", "#c994c7", "#dd1c77"]  # Color list poster Neurips
            marker_size = 15
            log_scale = True
            photons_dic = get_response_for_id_i([22], matched_pandora, matched_)
            electrons_dic = get_response_for_id_i([11], matched_pandora, matched_)
            plot_X(
                "Electromagnetic Response",
                photons_dic,
                electrons_dic,
                "mean",
                PATH_store,
                "Photons",
                "Electrons",
                el,
                plot_label1=True,
            )
            plot_X(
                "Electromagnetic Resolution",
                photons_dic,
                electrons_dic,
                "variance_om",
                PATH_store,
                "Photons",
                "Electrons",
                el,
                plot_label1=True,
            )
            plot_X(
                "Electromagnetic Resolution",
                photons_dic,
                electrons_dic,
                "variance_om",
                PATH_store,
                "Photons",
                "Electrons",
                el,
                plot_label2=True,
            )
            pions_dic = get_response_for_id_i([211], matched_pandora, matched_)
            kaons_dic = get_response_for_id_i([130], matched_pandora, matched_)
            plot_X(
                "Hadronic Response",
                pions_dic,
                kaons_dic,
                "mean",
                PATH_store,
                "Pions",
                "Kaons",
                el,
                plot_label1=True,
            )
            plot_X(
                "Hadronic Resolution",
                pions_dic,
                kaons_dic,
                "variance_om",
                PATH_store,
                "Pions",
                "Kaons",
                el,
                plot_label1=True,
            )
            plot_X(
                "Hadronic Resolution",
                pions_dic,
                kaons_dic,
                "variance_om",
                PATH_store,
                "Pions",
                "Kaons",
                el,
                plot_label2=True,
            )

def plot_fake_and_missed_energy_regions(sd_pandora, sd_hgb, path_store):
    evt = get_response_for_event_energy(
        sd_pandora, sd_hgb, perfect_pid=False, mass_zero=False, ML_pid=True
    )
    model = evt["mass_over_true_model"]
    pand = evt["mass_over_true_pandora"]

    def fake_energy_frac(df):
        is_pandora = "pandora_calibrated_pfo" in df.columns
        fakes = pd.isna(df.pid)
        if is_pandora:
            col = "pandora_calibrated_pfo"
        else:
            col = "calibrated_E"
        return df[fakes][col].sum() / df.true_showers_E.sum()

    def missed_energy_frac(df):
        is_pandora = "pandora_calibrated_pfo" in df.columns
        missed = pd.isna(df.pred_showers_E) | (df.pred_showers_E == 0)
        if is_pandora:
            col = "pandora_calibrated_pfo"
        else:
            col = "calibrated_E"
        return df[missed].true_showers_E.sum() / df.true_showers_E.sum()

    filt_regions = [[0, 0.75], [1.1, 1.5]]
    matplotlib.rcParams["font.size"] = 10
    fig, ax = plt.subplots(len(filt_regions), 2, figsize=(8, 4 * (len(filt_regions))))

    for i, item in enumerate(filt_regions):
        filt = np.where((model < item[1]) * (model > item[0]))[0]
        fake_energy_frac_distr = [fake_energy_frac(sd_hgb[sd_hgb.number_batch == q]) for q in filt]
        missed_energy_frac_distr = [missed_energy_frac(sd_hgb[sd_hgb.number_batch == q]) for q in filt]
        fake_energy_frac_distr_pandora = [fake_energy_frac(sd_pandora[sd_pandora.number_batch == q]) for q in filt]
        missed_energy_frac_distr_pandora = [missed_energy_frac(sd_pandora[sd_pandora.number_batch == q]) for q in filt]
        ax[i, 0].hist(fake_energy_frac_distr, bins=np.linspace(0, 1, 100), histtype="step", label="ML", color="red")
        ax[i, 0].hist(fake_energy_frac_distr_pandora, bins=np.linspace(0, 1, 100), histtype="step", label="Pandora",
                      color="blue")
        ax[i, 0].set_yscale("log")
        ax[i, 0].legend()
        ax[i, 0].grid()
        ax[i, 1].hist(missed_energy_frac_distr, bins=np.linspace(0, 1, 100), histtype="step", label="ML", color="red")
        ax[i, 1].hist(missed_energy_frac_distr_pandora, bins=np.linspace(0, 1, 100), histtype="step", label="Pandora",
                      color="blue")
        ax[i, 1].set_yscale("log")
        ax[i, 1].legend()
        ax[i, 1].grid()
        ax[i, 0].set_title(f"Model in [{item[0]:.2f},{item[1]:.2f}] - fake energy frac.")
        ax[i, 1].set_title(f"Model in [{item[0]:.2f},{item[1]:.2f}] - missed energy frac.")

    fig.tight_layout()
    fig.savefig(os.path.join(path_store, "fake_and_missed_energy_regions.pdf"))


def plot_efficiency_all(sd_pandora, df_list, PATH_store, labels, ax=None):
    matplotlib.rcParams["font.size"] = 22
    print("photons pandora")
    photons_dic = create_eff_dic_pandora(sd_pandora, 22)
    electrons_dic = create_eff_dic_pandora(sd_pandora, 11)
    pions_dic = create_eff_dic_pandora(sd_pandora, 211)
    kaons_dic = create_eff_dic_pandora(sd_pandora, 130)
    #fakes_dic_p = calculate_fakes(sd_pandora, None, False, pandora=True, id=22)
    #fakes_dic_p = {"fakes_p": fakes_dic_p[0], "energy_fakes_p": fakes_dic_p[1],
    #               "fake_percent_energy_p": fakes_dic_p[2], "fake_energy_reco_p": fakes_dic_p[3]}
    #photons_dic.update(fakes_dic_p)
    #fakes_dic_p = calculate_fakes(sd_pandora, None, False, pandora=True, id=11)
    #fakes_dic_p = {"fakes_p": fakes_dic_p[0], "energy_fakes_p": fakes_dic_p[1],
    #               "fake_percent_energy_p": fakes_dic_p[2]}
    #electrons_dic.update(fakes_dic_p)
    #fakes_dic_p = calculate_fakes(sd_pandora, None, False, pandora=True, id=211)
    #fakes_dic_p = {"fakes_p": fakes_dic_p[0], "energy_fakes_p": fakes_dic_p[1],
    #               "fake_percent_energy_p": fakes_dic_p[2]}
    #pions_dic.update(fakes_dic_p)
    #fakes_dic_p = calculate_fakes(sd_pandora, None, False, pandora=True, id=130)
    #fakes_dic_p = {"fakes_p": fakes_dic_p[0], "energy_fakes_p": fakes_dic_p[1],"fake_percent_energy_p": fakes_dic_p[2]}
    #kaons_dic.update(fakes_dic_p)
    for var_i, sd_hgb in enumerate(df_list):
        print("photons")
        photons_dic = create_eff_dic(photons_dic, sd_hgb, 22, var_i=var_i)
        #fakes_dic = calculate_fakes(sd_hgb, None, False, pandora=False, id=22)
        #photons_dic.update({"fakes_" + str(var_i): fakes_dic[0], "energy_fakes_" + str(var_i): fakes_dic[1], "fake_percent_energy_" + str(var_i): fakes_dic[2]})
        #photons_dic.update(create_fakes_dic(photons_dic, sd_hgb, 22, var_i))
        electrons_dic = create_eff_dic(electrons_dic, sd_hgb, 11, var_i=var_i)
        #fakes_dic = calculate_fakes(sd_hgb, None, False, pandora=False, id=11)
        #electrons_dic.update({"fakes_" + str(var_i): fakes_dic[0], "energy_fakes_" + str(var_i): fakes_dic[1], "fake_percent_energy_" + str(var_i): fakes_dic[2]})
        #electrons_dic.update(create_fakes_dic(electrons_dic, sd_hgb, 11, var_i))
        pions_dic = create_eff_dic(pions_dic, sd_hgb, 211, var_i=var_i)
        #fakes_dic = calculate_fakes(sd_hgb, None, False, pandora=False, id=211)
        #pions_dic.update({"fakes_" + str(var_i): fakes_dic[0], "energy_fakes_" + str(var_i): fakes_dic[1], "fake_percent_energy_" + str(var_i): fakes_dic[2]})
        kaons_dic = create_eff_dic(kaons_dic, sd_hgb, 130, var_i=var_i) # NH
        #fakes_dic = calculate_fakes(sd_hgb, None, False, pandora=False, id=130)
        #kaons_dic.update({"fakes_" + str(var_i): fakes_dic[0], "energy_fakes_" + str(var_i): fakes_dic[1], "fake_percent_energy_" + str(var_i): fakes_dic[2]})
        #kaons_dic.update(create_fakes_dic(kaons_dic, sd_hgb, 130, var_i))
        #plot_eff_and_fakes( # Doesn't look super nice
        #    "Electromagnetic",
        #    photons_dic,
        #    "Photons",
        #    PATH_store,
        #    labels,
        #)
    plot_eff(
        "Electromagnetic",
        photons_dic,
        "Photons",
        PATH_store,
        labels,
        ax=ax[0, 0]
    )
    plot_fakes(
        "Electromagnetic",
        photons_dic,
        "Photons",
        PATH_store,
        labels,
        ax=ax[0, 1]
    )
    plot_fakes_E(
        "Electromagnetic",
        photons_dic,
        "Photons",
        PATH_store,
        labels,
        ax=ax[0, 2]
    )
    plot_fakes_E(
        "Electromagnetic",
        photons_dic,
        "Photons",
        PATH_store,
        labels,
        ax=ax[0, 3],
        reco="reco_"
    )
    '''plot_fakes(
        "Electromagnetic",
        fakes_dic_p,
        "Photons",
        PATH_store,
        labels,
    )
    plot_fakes_E(
        "Electromagnetic",
        fakes_dic_p,
        "Photons",
        PATH_store,
        labels,
    )'''
    if len(electrons_dic["eff_p"]) > 0:
        plot_eff(
            "Electromagnetic",
            electrons_dic,
            "Electrons",
            PATH_store,
            labels,
            ax=ax[3, 0]
        )
        plot_fakes(
            "Electromagnetic",
            electrons_dic,
            "Electrons",
            PATH_store,
            labels,
            ax=ax[3, 1]
        )
        plot_fakes_E(
            "Electromagnetic",
            electrons_dic,
            "Electrons",
            PATH_store,
            labels,
            ax=ax[3, 2]
        )
        plot_fakes_E(
            "Electromagnetic",
            electrons_dic,
            "Electrons",
            PATH_store,
            labels,
            ax=ax[3, 3],
            reco="reco_"
        )
    if len(pions_dic["eff_p"]) > 0:
        plot_eff(
            "Hadronic",
            pions_dic,
            "Charged hadrons",
            PATH_store,
            labels,
            ax=ax[1, 0]
        )
        plot_fakes(
            "Hadronic",
            pions_dic,
            "Charged hadrons",
            PATH_store,
            labels,
            ax=ax[1, 1]
        )
        plot_fakes_E(
            "Hadronic",
            pions_dic,
            "Charged hadrons",
            PATH_store,
            labels,
            ax=ax[1, 2]
        )
        plot_fakes_E(
            "Hadronic",
            pions_dic,
            "Charged hadrons",
            PATH_store,
            labels,
            ax=ax[1, 3],
            reco="reco_"
        )
    if len(kaons_dic["eff_p"]) > 0:
        plot_eff(
            "Hadronic",
            kaons_dic,
            "Neutral hadrons",
            PATH_store,
            labels,
            ax=ax[2, 0]
        )
        plot_fakes(
            "Hadronic",
            kaons_dic,
            "Neutral hadrons",
            PATH_store,
            labels,
            ax=ax[2, 1]
        )
        plot_fakes_E(
            "Hadronic",
            kaons_dic,
            "Neutral hadrons",
            PATH_store,
            labels,
            ax=ax[2, 2]
        )
        plot_fakes_E(
            "Hadronic",
            kaons_dic,
            "Neutral hadrons",
            PATH_store,
            labels,
            ax=ax[2, 3],
            reco="reco_"

        )

def create_eff_dic_pandora(matched_pandora, id):
    pids_pandora = np.abs(matched_pandora["pid"].values)
    our_id = pandora_to_our_mapping[id]
    id_group = our_to_pandora_mapping[our_id]
    mask_id_true = matched_pandora.pid.isin(id_group)
    #df_id_pandora = matched_pandora[mask_id]
    eff_p, energy_eff_p, errors_p = calculate_eff(matched_pandora[mask_id_true], False, pandora=True)
    fakes_p, energy_fakes_p, fake_percent_energy, fake_percent_reco, fake_errors = calculate_fakes(matched_pandora, None, False, pandora=True, id=id)
    photons_dic = {}
    photons_dic["eff_p"] = eff_p
    photons_dic["errors_p"]= errors_p
    photons_dic["energy_eff_p"] = energy_eff_p
    photons_dic["fakes_p"] = fakes_p
    photons_dic["fakes_errors_p"]  = fake_errors
    photons_dic["energy_fakes_p"] = energy_fakes_p
    photons_dic["fake_percent_energy_p"] = fake_percent_energy
    photons_dic["fake_percent_energy_reco_p"] = fake_percent_reco
    return photons_dic

def create_eff_dic(photons_dic, matched_, id, var_i):
    pids = np.abs(matched_["pid"].values)
    our_id = pandora_to_our_mapping[id]
    id_group = our_to_pandora_mapping[our_id]
    mask_id = matched_.pred_pid_matched.isin(id_group)
    mask_id_gt = matched_.pid.isin(id_group)
    eff, energy_eff, errors = calculate_eff(matched_[mask_id_gt], False)
    fakes, energy_fakes, fake_percent_energy, fake_percent_reco, fake_errors = calculate_fakes(matched_, None, False, pandora=False, id=id)
    photons_dic["eff_" + str(var_i)] = eff
    photons_dic["errors_" + str(var_i)] = errors
    photons_dic["energy_eff_" + str(var_i)] = energy_eff
    photons_dic["fakes_" + str(var_i)] = fakes
    photons_dic["fakes_errors" + str(var_i)] = fake_errors
    photons_dic["energy_fakes_" + str(var_i)] = energy_fakes
    photons_dic["fake_percent_energy_" + str(var_i)] = fake_percent_energy
    photons_dic["fake_percent_energy_reco_" + str(var_i)] = fake_percent_reco
    return photons_dic

def create_fakes_dic(photons_dic, matched_, id, var_i):
    pids = np.abs(matched_["pid"].values)
    mask_id = pids == id
    our_to_pandora_mapping
    df_id = matched_[mask_id]
    eff, energy_eff, errors = calculate_eff(df_id, False)
    fakes, energy_fakes, fake_percent_energy, fake_percent_reco = calculate_fakes(df_id, None, False, pandora=False, id=id)
    photons_dic["eff_" + str(var_i)] = eff
    photons_dic["errors_fakes_" + str(var_i)] = errors
    photons_dic["energy_eff_" + str(var_i)] = energy_eff
    photons_dic["fakes_" + str(var_i)] = fakes
    photons_dic["energy_fakes_" + str(var_i)] = energy_fakes
    return photons_dic

def limit_error_bars(y, yerr, upper_limit=1):
    yerr_upper = np.minimum(y + yerr, upper_limit) - y
    yerr_lower = yerr  # Lower error bars remain unchanged
    return yerr_lower, yerr_upper

def plot_eff(title, photons_dic, label1, PATH_store, labels, ax=None):
    colors_list = ["red", "green", "blue"]
    savefig = ax is None
    if ax is None:
        fig, ax = plt.subplots()
    j = 0
    ax.set_xlabel("Energy [GeV]")
    ax.set_ylabel("Efficiency")
    ax.set_title(label1)
    ax.grid()
    for i in range(0, len(labels)):
        ax.plot(photons_dic["energy_eff_" + str(i)],
            photons_dic["eff_" + str(i)], "--", color=colors_list[i])
        ax.scatter(
            photons_dic["energy_eff_" + str(i)],
            photons_dic["eff_" + str(i)],
            label=labels[i], # temporarily, for the ML-Pandora comparison plots, change if plotting more labels!
            color=colors_list[i],
            s=50,
            marker="x"
        )
    energy = photons_dic["energy_eff_" + str(i)]
    eff = photons_dic["eff_" + str(i)]
    error_y = photons_dic["errors_"+ str(i)]
    yerr_lower, yerr_upper = limit_error_bars(eff, np.array(error_y)/2, upper_limit=1)
    ax.errorbar(energy, eff ,yerr= [yerr_lower, yerr_upper], ecolor=colors_list[i], linestyle='none', capsize=4)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.plot(photons_dic["energy_eff_p"],
        photons_dic["eff_p"], "--", color=colors_list[2])
    ax.scatter(
        photons_dic["energy_eff_p"],
        photons_dic["eff_p"],
        #facecolors=colors_list[2],
        #edgecolors=colors_list[2],
        color=colors_list[2],
        label="Pandora",
        # Add -- line
        s=50,
        marker="x"
    )
    energy = photons_dic["energy_eff_p"]
    eff = photons_dic["eff_p"]
    error_y = photons_dic["errors_p"]
    yerr_lower, yerr_upper = limit_error_bars(eff, np.array(error_y)/2, upper_limit=1)
    ax.errorbar(energy, eff ,yerr= [yerr_lower, yerr_upper], ecolor=colors_list[2], linestyle='none', capsize=4)

    ax.legend(loc="lower right")
    # ax.set_xscale("log")
    if savefig:
        fig.savefig(
            os.path.join(PATH_store, "Efficiency_" + label1 + ".pdf"),
            bbox_inches="tight",
        )
    else:
        plot_eff(title, photons_dic, label1, PATH_store, labels, ax=None)

def plot_eff_and_fakes(title, photons_dic, label1, PATH_store, labels):
    colors_list = ["#FF0000",  "#00FF00", "#0000FF"]
    fig, ax = plt.subplots()
    j = 0
    ax.set_xlabel("Energy [GeV]")
    ax.set_ylabel("Efficiency")
    # ax[row_i, j].set_xscale("log")
    ax.set_title(title)
    ax.grid(1)
    for i in range(0, len(labels)):
        ax.plot(photons_dic["energy_eff_" + str(i)],
            photons_dic["eff_" + str(i)], "--", color=colors_list[i])
        ax.scatter(
            photons_dic["energy_eff_" + str(i)],
            photons_dic["eff_" + str(i)],
            label="ML " + label1, # Temporarily, for the ML-Pandora comparison plots, change if plotting more labels!
            color=colors[labels[i]],
            s=50,
        )
    ax.plot(photons_dic["energy_eff_p"],
        photons_dic["eff_p"], "--", color=colors_list[2])
    ax.scatter(
        photons_dic["energy_eff_p"],
        photons_dic["eff_p"],
        facecolors=colors_list[2],
        edgecolors=colors_list[2],
        label="Pandora " + label1,
        s=50,
    )
    ax.legend(loc="upper right")
    if title == "Electromagnetic":
        ax.set_ylim([0.5, 1.1])
    else:
        ax.set_ylim([0.5, 1.1])
    ax.set_xscale("log")
    ax.set_xlabel("Efficiency")
    ax_fakes = inset_axes(ax,
                            width="50%",  # width = 30% of parent_bbox
                            height="40%",  # height : 1 inch
                            loc="lower right")
    ax_fakes.set_ylabel("Fake rate")
    for i in range(0, len(labels)):
        ax_fakes.plot(photons_dic["energy_fakes_" + str(i)],
            photons_dic["fakes_" + str(i)], "--", color=colors_list[0])
        ax_fakes.scatter(
            photons_dic["energy_fakes_" + str(i)],
            photons_dic["fakes_" + str(i)],
            label="ML", # Temporarily, for the ML-Pandora comparison plots, change if plotting more labels!
            color=colors_list[0],
            s=50,
        )
    ax_fakes.grid()
    ax_fakes.set_xlabel("Energy [GeV]")
    ax_fakes.plot(photons_dic["energy_fakes_p"],
        photons_dic["fakes_p"], "--", color=colors_list[2])
    ax_fakes.scatter(
        photons_dic["energy_fakes_p"],
        photons_dic["fakes_p"],
        facecolors=colors_list[2],
        edgecolors=colors_list[2],
        label="Pandora",
        # add -- line
        s=50,
    )
    fig.savefig(
        os.path.join(PATH_store, "DoublePlot_" + title + label1 + ".pdf"),
        bbox_inches="tight",
    )


def plot_fakes_E(title, photons_dic, label1, PATH_store, labels, ax=None, reco=""): # Set reco to 'reco_'
    colors_list = ["red",  "green", "blue"]
    savefig = ax is None
    if ax is None:
        fig, ax = plt.subplots()
    j = 0
    ax.set_xlabel("Energy [GeV]")
    if "reco" in reco:
        ax.set_ylabel("Fake reco energy rate")
    else:
        ax.set_ylabel("Fake energy rate")
    # ax[row_i, j].set_xscale("log")
    ax.set_title(label1)
    ax.grid()
    for i in range(0, len(labels)):
        ax.plot(photons_dic["energy_fakes_" + str(i)],
            photons_dic["fake_percent_energy_" + reco + str(i)], "--", color=colors_list[i])
        ax.scatter(
            photons_dic["energy_fakes_" + str(i)],
            photons_dic["fake_percent_energy_" + reco + str(i)],
            label=labels[i], # Temporarily, for the ML-Pandora comparison plots, change if plotting more labels!
            color=colors_list[i],
            s=50,
            marker="x"
        )
    ax.plot(photons_dic["energy_fakes_p"],
        photons_dic["fake_percent_energy_" + reco + "p"], "--", color=colors_list[2])
    ax.scatter(
        photons_dic["energy_fakes_p"],
        photons_dic["fake_percent_energy_" + reco + "p"],
        facecolors=colors_list[2],
        edgecolors=colors_list[2],
        label="Pandora",
        s=50,
        marker="x"
    )
    ax.legend(loc="upper right")
    #if title == "Electromagnetic":
    #    plt.ylim([0.0, 0.5])
    #else:
    #    plt.ylim([0.0, 0.5])
    ax.set_xscale("log")
    if savefig:
        fig.savefig(
            os.path.join(PATH_store, "Fake_Energy_Rate_"  + reco + label1 + ".pdf"),
            bbox_inches="tight",
        )
    #else:
    #    plot_fakes_E(title, photons_dic, label1, PATH_store, labels, ax=None)

def plot_fakes(title, photons_dic, label1, PATH_store, labels, ax=None):
    colors_list = ["red",  "green", "blue"]
    savefig = ax is None
    if ax is None:
        fig, ax = plt.subplots()
    j = 0
    ax.set_xlabel("Energy [GeV]")
    ax.set_ylabel("Fake rate")
    ax.set_title(label1)
    ax.grid()
    for i in range(0, len(labels)):
        ax.plot(photons_dic["energy_fakes_" + str(i)],
            photons_dic["fakes_" + str(i)], "--", color=colors_list[i])
        ax.scatter(
            photons_dic["energy_fakes_" + str(i)],
            photons_dic["fakes_" + str(i)],
            label=labels[i], # Temporarily, for the ML-Pandora comparison plots, change if plotting more labels!
            color=colors_list[i],
            s=50,
            marker="x"
        )
    ax.plot(photons_dic["energy_fakes_p"],
        photons_dic["fakes_p"], "--", color=colors_list[2])
    ax.scatter(
        photons_dic["energy_fakes_p"],
        photons_dic["fakes_p"],
        facecolors=colors_list[2],
        edgecolors=colors_list[2],
        label="Pandora",
        marker="x",
        # add -- line
        s=50,

    )
    energy = photons_dic["energy_fakes_" + str(i)]
    eff = photons_dic["fakes_" + str(i)]
    error_y = photons_dic["fakes_errors"+ str(i)]
    # yerr_lower, yerr_upper = limit_error_bars(eff, np.array(error_y)/2, upper_limit=1)
    print("NOT pandora", error_y, energy, eff)
    error_y = np.array(error_y)/2
    ax.errorbar(energy, eff ,yerr= [error_y, error_y], ecolor=colors_list[i], linestyle='none', capsize=4)

    energy = photons_dic["energy_fakes_p"]
    eff = photons_dic["fakes_p"]
    error_y = photons_dic["fakes_errors_p"]
    # yerr_lower, yerr_upper = limit_error_bars(eff, np.array(error_y)/2, upper_limit=1)
    print("pandora", error_y, energy, eff)
    error_y = np.array(error_y)/2
    ax.errorbar(energy, eff ,yerr= [error_y, error_y], ecolor=colors_list[2], linestyle='none', capsize=4)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.legend()
    #if title == "Electromagnetic":
    #    plt.ylim([0.0, 0.07])
    #else:
    #    plt.ylim([0.0, 0.07])
    # ax.set_xscale("log")
    if savefig:
        fig.savefig(
            os.path.join(PATH_store, "Fake_Rate_" + label1 + ".pdf"),
            bbox_inches="tight",
        )
    else:
        plot_fakes(title, photons_dic, label1, PATH_store, labels, ax=None)

def calculate_phi(x, y, z=None):
    return torch.arctan2(y, x)

def calculate_eta(x, y, z):
    theta = torch.arctan2(torch.sqrt(x**2 + y**2), z)
    return -torch.log(torch.tan(theta / 2))

from copy import copy
import plotly
import plotly.graph_objs as go
import plotly.express as px

def plot_event(df, pandora=True, output_dir="", graph=None, y=None, labels=None, is_track_in_cluster=None, pid_filter=[], color_hittype=True):
    return
    fig = go.Figure()
    color_list = px.colors.qualitative.Light24 + px.colors.qualitative.Dark24 + px.colors.qualitative.Plotly
    if graph is not None:
        sum_hits = scatter_sum(
            graph.ndata["e_hits"].flatten(), graph.ndata["particle_number"].long()
        )
        hit_pos = graph.ndata["pos_hits_xyz"].numpy()
        mask = torch.ones(len(hit_pos))
        if len(pid_filter):
            pid_list = y.pid.flatten().int()
            pids = graph.ndata["particle_number"].long()
            pids = torch.tensor([pid_list[i-1] for i in pids])
            mask = torch.isin(pids, torch.tensor(pid_filter).int())
            assert len(mask) == len(hit_pos)
        scale = np.mean(np.linalg.norm(hit_pos, axis=1))
        truepos = np.array(df.true_pos.values.tolist()) * scale
        #truepos = truepos[~np.isnan(truepos[:, 0])]
        #vertices = np.zeros_like(truepos[~np.isnan(truepos[:, 0])])
        vertices = np.stack(df.vertex.values)
        #if y is not None:
        #    assert vertices.shape == y.vertex.shape
        #    vertices = y.vertex
        # fig.add_trace(go.Scatter3d(x=hit_pos[:, 0], y=hit_pos[:, 1], z=hit_pos[:, 2], mode='markers', color=graph.ndata["particle_number"], name='hits'))
        # fix this: color by particle number (it is an array of size [0,0,0,0,1,1,1,2,2,2...]
        energies = y.E_corrected.flatten()
        hit_types=graph.ndata["hit_type"]
        ht = [
            f"part. {int(i)}, sum_hits={sum_hits[i]:.2f}, E={energies[i-1]:.2f}, hit_type={hit_types[n].item()}"
            for n, i in enumerate(graph.ndata["particle_number"].long().tolist())
        ]
        if labels is not None:
            has_track = scatter_sum(graph.ndata["h"][:, 8], labels.long().cpu())
            #if has_track.sum() == 0.0:
            #    return   # Filter !!! Only plot those with tracks
            # pids = df.pid.values
            ht_clusters = [f"cluster {i}, has_track={has_track[i]}" for i in labels]
            ht = zip(ht, ht_clusters)
            ht = [f"{a}, {b}" for a, b in ht]

        def get_from_list(lst, i):
            if i < len(lst):
                return lst[i]
            else:
                return "gray"
        if labels is not None:
            c = [get_from_list(color_list, i) for i in labels]
        if color_hittype:
            c = [color_list[int(i.item())] for i in graph.ndata["hit_type"]]
        else:
            c = [color_list[int(i.item())] for i in graph.ndata["particle_number"]]

        if pandora:
            c = [color_list[int(i.item())] for i in graph.ndata["pandora_cluster"]]

        hit_pos = hit_pos[mask]
        ht = [ht[i] for i in range(len(ht)) if mask[i]]

        fig.add_trace(
            go.Scatter3d(
                x=hit_pos[:, 0],
                y=hit_pos[:, 1],
                z=hit_pos[:, 2],
                mode="markers",
                marker=dict(size=4, color=c),
                name="hits",
                hovertext=ht,
                hoverinfo="text",
            )
        )
        if is_track_in_cluster is not None:
            plt.title(f"Track in cluster {is_track_in_cluster}")
            fig.update_layout(title_text=f"Track in cluster {is_track_in_cluster}")
            fig.update_layout(title_text=f"Track in cluster {is_track_in_cluster}")
        # set scale to avg hit_pos
        if "pos_pxpypz" in graph.ndata.keys():
            vectors = graph.ndata["pos_pxpypz"].numpy()
            if "pos_pxpypz_at_vertex" in graph.ndata.keys():
                pos_at_vertex = graph.ndata["pos_pxpypz_at_vertex"].numpy()
                ps_vertex = graph.ndata["pos_pxpypz_at_vertex"]
                ps_vertex = torch.norm(ps_vertex, dim=1).numpy()
            else:
                pos_at_vertex = None
            trks = graph.ndata["pos_hits_xyz"].numpy()
            # filt = (vectors[:, 0] != 0) & (vectors[:, 1] != 0) & (vectors[:, 2] != 0)
            filt = graph.ndata["h"][:, 8] > 0
            hit_type = graph.ndata["hit_type"][filt]
            vf = vectors[filt]
            vectors = vectors[filt]
            if pos_at_vertex is not None:
                pos_at_vertex = pos_at_vertex[filt]
                ps_vertex = ps_vertex[filt]
            trks = trks[filt]  # track positions
            # normalize 3-comp vectors / np.linalg.norm(vectors, axis=1) * scale # remove zero vectors
            vectors = vectors / np.linalg.norm(vectors, axis=1).reshape(-1, 1) * scale
            if pos_at_vertex is not None:
                pos_at_vertex = (
                    pos_at_vertex
                    / np.linalg.norm(pos_at_vertex, axis=1).reshape(-1, 1)
                    * scale
                )
            track_p = graph.ndata["h"][:, 8].numpy()[filt]
            pnum = graph.ndata["particle_number"].long()[filt]
            # plot these vectors
            for i in range(len(vectors)):
                if hit_type[i] == 0:
                    line = dict(color="black", width=1)
                elif hit_type[i] == 1:
                    line = dict(color="black", width=1, dash="dash")
                else:
                    line = dict(color="purple", width=1, dash="dot")
                    pass # muons
                    #raise Exception
                def plot_single_arrow(
                    fig, vec, hovertext="", init_pt=[0, 0, 0], line=line
                ):
                    # init_pt: initial point of the vector
                    fig.add_trace(
                        go.Scatter3d(
                            x=[init_pt[0], vec[0] + init_pt[0]],
                            y=[init_pt[1], init_pt[1] + vec[1]],
                            z=[init_pt[2], init_pt[2] + vec[2]],
                            mode="lines",
                            line=line,
                        )
                    )
                    fig.add_trace(
                        go.Scatter3d(
                            x=[vec[0] + init_pt[0]],
                            y=[vec[1] + init_pt[1]],
                            z=[vec[2] + init_pt[2]],
                            mode="markers",
                            marker=dict(size=4, color="black"),
                            hovertext=hovertext,
                        )
                    )
                if False:
                    plot_single_arrow(
                        fig,
                        vectors[i] / 5,
                        hovertext=f"track {track_p[i]} , pxpypz={vf[i]}",
                        init_pt=trks[i],
                    )  # a bit smaller
                    if pos_at_vertex is not None:
                        plot_single_arrow(
                            fig,
                            pos_at_vertex[i],
                            hovertext=f"track at DCA {ps_vertex[i]}, px,py,pz={pos_at_vertex[i]}",
                            init_pt=[0, 0, 0],
                        )
        # Color this by graph.ndata["particle_number"]
    # Get the norm of vertices
    displacement = np.linalg.norm(vertices, axis=1)
    #if displacement.max() < 400:
    #    return
    #else:
    #    print("Displaced")
    pids = [str(x) for x in df.pid.values]
    col = np.arange(1, len(truepos) + 1)
    true_E = df.true_showers_E.values
    true_P = np.array(df.true_pos.values.tolist())
    true_P /= np.linalg.norm(true_P, axis=1).reshape(-1, 1)
    true_P *= scale
    ht = [
        f"GT E={true_E[i]:.2f}, PID={pids[i]}, p={true_P[i]}"
        for i in range(len(true_E))
    ]
    if pandora:
        pandora_cluster = graph.ndata["pandora_cluster"].long() + 1
        pandorapos = np.array(df.pandora_calibrated_pos.values.tolist())
        mask = ~np.isnan(df.pandora_calibrated_E.values)
        pandoramomentum = np.linalg.norm(pandorapos, axis=1)
        pandorapos = pandorapos / np.linalg.norm(pandorapos, axis=1).reshape(-1, 1)
        #pandorapos = pandorapos[1:]
        # normalize
        #pandora_ref_pt = scatter_mean(
        #    graph.ndata["pandora_reference_point"], pandora_cluster.long(), dim=0
        #)
        #pandora_ref_pt = pandora_ref_pt[1:]
        pandora_ref_pt = torch.tensor(np.stack(df.pandora_ref_pt.values.tolist()))
        #pandora_ref_pt #/= np.linalg.norm(pandora_ref_pt, axis=1).reshape(-1, 1)
        assert pandora_ref_pt.shape == pandorapos.shape
        ref_pt = pandora_ref_pt
        pandora_ref_pt_diff =  pandora_ref_pt[mask] - vertices[mask]
        pandora_ref_pt_diff_norm = pandora_ref_pt_diff / np.linalg.norm(pandora_ref_pt_diff, axis=1).reshape(-1, 1)
        GT_translation = pandora_ref_pt_diff_norm.numpy()
        assert pandora_ref_pt.shape == pandorapos.shape
        pandorapos *= scale
        # fig.add_trace(go.Scatter3d(x=vertices[:, 0] + pandorapos[:, 0], y=vertices[:, 1] + pandorapos[:, 1], z=vertices[:, 2] + pandorapos[:, 2], mode='markers', marker=dict(size=4, color='green'), name='Pandora', hovertext=df.pandora_calibrated_E.values, hoverinfo="text"))
        fig.add_trace(
            go.Scatter3d(
                x=torch.tensor(pandora_ref_pt[:, 0]) + pandorapos[:, 0],
                y=torch.tensor(pandora_ref_pt[:, 1]) + pandorapos[:, 1],
                z=torch.tensor(pandora_ref_pt[:, 2]) + pandorapos[:, 2],
                mode="markers",
                marker=dict(size=4, color="green"),
                name="Pandora",
                hovertext=pandoramomentum.flatten()[1:],
                hoverinfo="text",
            ))
        # Also add the lines herepandora_ref_pt.shape == pandorapos.shape
        for i in range(len(pandorapos)):
            #v = [0, 0, 0]  # Temporarily. TODO: find the Pandora 'vertex'
            v = pandora_ref_pt[i]
            #v = [0,0,0]
            fig.add_trace(
                go.Scatter3d(
                    x=[v[0], v[0] + pandorapos[i, 0]],
                    y=[v[1], v[1] + pandorapos[i, 1]],
                    z=[v[2], v[2] + pandorapos[i, 2]],
                    mode="lines",
                    line=dict(color="green", width=1),
                )
            )
    else:
        predpos = np.array(df.pred_pos_matched.values.tolist())
        mask = ~np.isnan(np.stack(df.pred_pos_matched.values)[:, 0])
        predpos = predpos[mask]
        predpos /= np.linalg.norm(predpos, axis=1).reshape(-1, 1)
        predpos *= scale
        pred_ref_pt = np.array(df.pred_ref_pt_matched.values.tolist())
        ref_pt = pred_ref_pt
        pred_ref_pt_diff = pred_ref_pt[mask] - vertices[mask]
        pred_ref_pt_diff_norm = pred_ref_pt_diff / np.linalg.norm(pred_ref_pt_diff, axis=1).reshape(-1, 1)
        #GT_translation = pred_ref_pt_diff_norm[mask]
        #pred_ref_pt /= np.linalg.norm(pred_ref_pt, axis=1).reshape(-1, 1)
        #v = [0, 0, 0]  # Do an average of the hits for plotting this
        #v = vertices[i]
        # ... Add lines ...
        if False:
            fig.add_trace(
                go.Scatter3d(
                    x=ref_pt[mask][:, 0] + predpos[:, 0],
                    y=ref_pt[mask][:, 1] + predpos[:, 1],
                    z=ref_pt[mask][:, 2] + predpos[:, 2],
                    mode="markers",
                    marker=dict(size=4, color="red"),
                    name="ML",
                    hovertext=df.calibrated_E.values,
                )
            )
            for i in range(len(predpos)):
                v = ref_pt[i]
                fig.add_trace(
                    go.Scatter3d(
                        x=[v[0], v[0] + predpos[i, 0]],
                        y=[v[1], v[1] + predpos[i, 1]],
                        z=[v[2], v[2] + predpos[i, 2]],
                        mode="lines",
                        line=dict(color="red", width=1),
                    )
        )
    # Plot GT (but translate it according to the Pandora or ML reference pt)
    # Add lines from (0, 0, 0) to these points...
    truepos_norm = truepos / np.linalg.norm(truepos, axis=1).reshape(-1, 1)  # From vertex!
    truepos_norm_from_ref_pt = truepos_norm# - GT_translation
    truepos_norm_from_ref_pt /= np.linalg.norm(truepos_norm_from_ref_pt, axis=1).reshape(-1, 1)
    truepos_norm_from_ref_pt *= scale
    if False:
        fig.add_trace(
            go.Scatter3d(
                x=ref_pt[:, 0] + truepos_norm_from_ref_pt[:, 0],
                y=ref_pt[:, 1] + truepos_norm_from_ref_pt[:, 1],
                z=ref_pt[:, 2] + truepos_norm_from_ref_pt[:, 2],
                mode="markers",
                marker=dict(size=4, color=[color_list[c] for c in col]),
                name="ground truth",
                hovertext=ht,
                hoverinfo="text",
            )
        )
    if False:
        # ignore this block for now and just look at the hits
        # Also plot a dotted black line for each vertex from vertex to the true position
        for i in range(len(truepos[mask])):
            v = ref_pt[i]
            fig.add_trace(
                go.Scatter3d(
                    x=[v[0], v[0] + truepos_norm_from_ref_pt[i, 0]],
                    y=[v[1], v[1] + truepos_norm_from_ref_pt[i, 1]],
                    z=[v[2], v[2] + truepos_norm_from_ref_pt[i, 2]],
                    mode="lines",
                    line=dict(color="blue", width=1),
                )
            )
            v = vertices[mask][i]
            fig.add_trace(
                go.Scatter3d(
                    x=[v[0], v[0] + truepos_norm_from_ref_pt[i, 0]],
                    y=[v[1], v[1] + truepos_norm_from_ref_pt[i, 1]],
                    z=[v[2], v[2] + truepos_norm_from_ref_pt[i, 2]],
                    mode="lines",
                    line=dict(color="black", width=2),
                )
            )
            v = [0, 0, 0]
            fig.add_trace(
                go.Scatter3d(
                    x=[v[0], v[0] + truepos_norm_from_ref_pt[i, 0]],
                    y=[v[1], v[1] + truepos_norm_from_ref_pt[i, 1]],
                    z=[v[2], v[2] + truepos_norm_from_ref_pt[i, 2]],
                    mode="lines",
                    line=dict(color="gray", width=2),
                )
            )
        assert output_dir != ""
        if "25.0" in output_dir:
            print("26")
    plotly.offline.plot(fig, filename=output_dir + "event.html")
    # also plot a png for big events which cannot be rendered fast
    fig.write_image(output_dir + "event.png")


def calculate_theta(x, y, z):
    return torch.acos(z / torch.sqrt(x ** 2 + y ** 2 + z ** 2))

def phi_dist(phi_pred, phi_true):
    # if the difference is larger than pi, take the smaller angle
    diff = phi_pred - phi_true
    # diff has to be a Gaussian centered around zero
    diff = np.where(diff > np.pi, 2 * np.pi - diff, diff)
    diff = np.where(diff < -np.pi, 2 * np.pi + diff, diff)
    return diff

def calc_unit_circle_dist(df, pandora=False):
    # A quick histogram of distances between unit vectors of directions - to compare with the light training model
    # Also returns the delta phi and delta theta
    if pandora:
        assert "pandora_calibrated_pos" in df.columns
    pids = []
    distances = []
    true_e = df.true_showers_E.values
    batch_idx = df.number_batch
    if pandora:
        pred_vect = np.array(df.pandora_calibrated_pos.values.tolist())
        true_vect = (
            np.array(df.true_pos.values.tolist())
            * torch.tensor(true_e).unsqueeze(1).repeat(1, 3).numpy()
        )
        pred_vect = torch.tensor(pred_vect)
        true_vect = torch.tensor(true_vect)
        # normalize
        pred_vect = pred_vect / torch.norm(pred_vect, dim=1).reshape(-1, 1)
        true_vect = true_vect / torch.norm(true_vect, dim=1).reshape(-1, 1)
    else:
        pred_vect = np.array(df.pred_pos_matched.values.tolist())
        true_vect = (
            np.array(df.true_pos.values.tolist())
            * torch.tensor(true_e).unsqueeze(1).repeat(1, 3).numpy()
        )
        pred_vect = torch.tensor(pred_vect)
        true_vect = torch.tensor(true_vect)
        # normalize
        pred_vect = pred_vect / torch.norm(pred_vect, dim=1).reshape(-1, 1)
        true_vect = true_vect / torch.norm(true_vect, dim=1).reshape(-1, 1)
    phi_pred, phi_true = calculate_phi(pred_vect[:, 0], pred_vect[:, 1]), calculate_phi(
        true_vect[:, 0], true_vect[:, 1]
    )
    #eta_pred, eta_true = calculate_eta(
    #    pred_vect[:, 0], pred_vect[:, 1], pred_vect[:, 2]
    #), calculate_eta(true_vect[:, 0], true_vect[:, 1], true_vect[:, 2])
    theta_pred = calculate_theta(pred_vect[:, 0], pred_vect[:, 1], pred_vect[:, 2])
    theta_true = calculate_theta(true_vect[:, 0], true_vect[:, 1], true_vect[:, 2])
    dist = torch.sqrt(torch.sum((pred_vect - true_vect) ** 2, dim=1))
    phidist = phi_dist(phi_pred, phi_true)
    etadist = theta_pred - theta_true # formely eta
    return dist, df.pid.values, phidist, etadist

particle_masses = {22: 0, 11: 0.00511, 211: 0.13957, 130: 0.493677, 2212: 0.938272, 2112: 0.939565, 321: 0.493677, 13: 0.1056}
particle_masses_4_class = {0: 0.00511, 1: 0.13957, 2: 0.939565, 3: 0.0} # Electron, CH, NH, photon

def safeint(x):
    # if x is nan, return nan
    try:
        return int(x)
    except:
        return x

def calculate_response(matched, pandora, log_scale=False, tracks=False, perfect_pid=False, mass_zero=False, ML_pid=False, pid=None):
    if log_scale:
        bins = np.exp(np.arange(np.log(0.1), np.log(80), 0.3))
    else:
        #bins = np.linspace(0, 51, 5)
        bins = [0, 5, 15,  51]
    mean = []
    variance_om = []
    mean_baseline = []
    variance_om_baseline = []
    mean_true_rec = []
    variance_om_true_rec = []
    mean_errors = []
    variance_om_errors = []
    energy_resolutions = []
    energy_resolutions_reco = []
    distributions = []  # Distributions of E/E_{true} for plotting later
    distributions_reco = []
    mean_pxyz = []
    variance_pxyz = []
    masses = []
    is_track_in_cluster = []
    pxyz_true, pxyz_pred = [], []
    sigma_phi, sigma_theta = [], [] # for the angular resolution vs. energy
    distr_phi, distr_theta = [], []
    #binning = 1e-2 * 0.2
    bins_per_binned_E = np.linspace(0, 2, 1000)

    for i in range(len(bins) - 1):
        bin_i = bins[i]
        bin_i1 = bins[i + 1]
        mask_above = matched["true_showers_E"] <= bin_i1
        mask_below = matched["true_showers_E"] > bin_i
        mask_check = ~pd.isna(matched["pred_showers_E"])
        mask = mask_below * mask_above * mask_check
        true_e = matched.true_showers_E[mask]
        true_rec = matched.reco_showers_E[mask]
        print("sum mask", bin_i, bin_i1, np.sum(mask))
        if pandora:
            pred_e = matched.pandora_calibrated_pfo[mask]
            pred_pxyz = np.array(matched.pandora_calibrated_pos[mask].tolist())
        else:
            pred_e = matched.calibrated_E[mask]
            pred_pxyz = np.array(matched.pred_pos_matched[mask].tolist())
        pred_e_nocor = matched.pred_showers_E[mask]
        trk_in_clust = matched.is_track_in_cluster[mask]
        if perfect_pid or mass_zero or ML_pid:
            if len(pred_pxyz):
                pred_pxyz /= np.linalg.norm(pred_pxyz, axis=1).reshape(-1, 1)
            if perfect_pid:
                m = np.array([particle_masses[abs(int(i))] for i in matched.pid[mask]])
            elif ML_pid:
                if pandora:
                    print("Using Pandora PID")
                    m = np.array([particle_masses[abs(int(i))] for i in matched.pandora_pid[mask]])
                else:
                    m = np.array([particle_masses_4_class.get(safeint(i), 0.0) for i in matched.pred_pid_matched[mask]])
            if mass_zero:
                m = np.array([0 for _ in range(len(matched.pid[mask]))])
            p_squared = (pred_e**2 - m**2).values
            pred_pxyz = np.sqrt(p_squared).reshape(-1, 1) * pred_pxyz
        true_pxyz = np.array(matched.true_pos[mask].tolist())

        if np.sum(mask) > 0:  # if the bin is not empty
            e_over_true = pred_e / true_e
            e_over_reco = true_rec / true_e
            e_over_reco_ML = pred_e_nocor / true_rec
            pxyz_over_true = pred_pxyz / true_pxyz
            dist, _, phi_dist, eta_dist = calc_unit_circle_dist(matched[mask], pandora=pandora)
            #if eta_dist.abs().max() < 0.01:
            #    bins_angle = np.linspace(-0.01, +0.01, 1000)
            p_size_over_true = np.linalg.norm(pred_pxyz, axis=1) / np.linalg.norm(true_pxyz, axis=1)
            #mu, var, _, _ = get_sigma_gaussian(phi_dist, bins_angle)
            bins_angle = np.linspace(-0.05, +0.05, 1000)

            if pid in [211, -211, 2212, -2212, 11, -11] and i in [1, 2]:
                bins_angle = np.linspace(-0.01, 0.01, 1000)
            mu, var_phi , _ , _= get_sigma_gaussian(phi_dist, bins_angle, return_gaussian=False, return_divided=False)
            #mu, var, _, _ = get_sigma_gaussian(eta_dist, bins_angle)
            mu, var_theta , _, _= get_sigma_gaussian(eta_dist, bins_angle, return_gaussian=False, return_divided=False)
            sigma_phi.append(var_phi)
            sigma_theta.append(var_theta)
            distr_theta.append(eta_dist)
            distr_phi.append(phi_dist)
            distributions.append(e_over_true)
            distributions_reco.append(e_over_reco_ML)
            bins_per_binned_E = np.linspace(0, 2, 2000)
            (
                mean_predtotrue,
                var_predtotrue,
                err_mean_predtotrue,
                err_var_predtotrue,
            ) = get_sigma_gaussian(e_over_true, bins_per_binned_E)
            pred_ps = np.linalg.norm(pred_pxyz, axis=1)
            masses.append((torch.tensor(pred_e.values) ** 2 - torch.tensor(pred_ps) ** 2))
            (
                mean_reco_true,
                var_reco_true,
                err_mean_reco_true,
                err_var_reco_true,
            ) = get_sigma_gaussian(e_over_reco, bins_per_binned_E)
            (
                mean_reco_ML,
                var_reco_ML,
                err_mean_reco_ML,
                err_mean_var_reco_ML,
            ) = get_sigma_gaussian(e_over_reco_ML, bins_per_binned_E)
            if not pandora:
                print("Not Pandora")
            mean_pxyz_, var_pxyz_ = [], []
            pxyz_true.append(true_pxyz)
            pxyz_pred.append(pred_pxyz)
            for i in [0, 1, 2]: # x, y, z
                (
                    mean_px,
                    var_px,
                    _,
                    _,
                ) = get_sigma_gaussian(pxyz_over_true[:, i], bins_per_binned_E)
                mean_pxyz_.append(mean_px)
                var_pxyz_.append(var_px)
            (
                mean_px,
                var_px,
                _,
                _,
            ) = get_sigma_gaussian(p_size_over_true, bins_per_binned_E)
            mean_pxyz_.append(mean_px)
            var_pxyz_.append(var_px)
            #mean_pxyz_ = np.array(mean_pxyz_)
            #var_pxyz_ = np.array(var_pxyz_)
            # raise err if mean_reco_ML is nan
            #if np.isnan(mean_reco_ML):
            #    raise ValueError("mean_reco_ML is nan")
            mean_true_rec.append(mean_reco_ML)
            variance_om_true_rec.append(np.abs(var_reco_ML))
            mean_baseline.append(mean_reco_true)
            variance_om_baseline.append(np.abs(var_reco_true))
            mean.append(mean_predtotrue)
            variance_om.append(np.abs(var_predtotrue))
            energy_resolutions.append((bin_i1 + bin_i) / 2)
            energy_resolutions_reco.append((bin_i1 + bin_i) / 2)
            mean_errors.append(err_mean_predtotrue)
            variance_om_errors.append(err_var_predtotrue)
            mean_pxyz.append(mean_pxyz_)
            variance_pxyz.append(var_pxyz_)
            is_track_in_cluster.append(trk_in_clust)

    return (
        mean,
        variance_om,
        mean_true_rec,
        variance_om_true_rec,
        energy_resolutions,
        energy_resolutions_reco,
        mean_baseline,
        variance_om_baseline,
        distributions,
        mean_errors,
        variance_om_errors,
        np.array(mean_pxyz),
        np.array(variance_pxyz),
        [masses, is_track_in_cluster],
        pxyz_true,
        pxyz_pred,
        sigma_phi,
        sigma_theta,
        distr_phi,
        distr_theta,
        distributions_reco
    )


def process_element(i, bins, matched, pandora, bins_per_binned_E):
    # Your processing logic here
    bin_i = bins[i]
    bin_i1 = bins[i + 1]
    print(i)
    mask_above = matched["reco_showers_E"] <= bin_i1
    mask_below = matched["reco_showers_E"] > bin_i
    mask_check = matched["pred_showers_E"] > 0
    mask = mask_below * mask_above * mask_check

    pred_e = matched.calibrated_E[mask]
    true_rec = matched.reco_showers_E[mask]
    if np.sum(mask) > 0:  # if the bin is not empty
        e_over_rec = pred_e / true_rec

        mean_predtored, variance_om_true_rec_ = obtain_MPV_and_68(
            e_over_rec, bins_per_binned_E
        )

    return mean_predtored, variance_om_true_rec_, (bin_i1 + bin_i) / 2


def parallel_process(
    vector, fixed_arg1, fixed_arg2, fixed_arg3, fixed_arg4, num_workers=16
):
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Use executor.map to parallelize the processing of vector elements
        results1 = list(
            executor.map(
                process_element,
                vector,
                [fixed_arg1] * len(vector),
                [fixed_arg2] * len(vector),
                [fixed_arg3] * len(vector),
                [fixed_arg4] * len(vector),
            )
        )
    return results1

def plot_sigma_angle_vs_energy(dic, PATH_store, label, angle, title="", ax=None):
    assert angle in ['theta', 'phi']
    new_plot = ax is None
    if new_plot:
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    for key in dic:
        E = np.array(dic[key]["energy_resolutions"])
        if angle == 'theta':
            sigma = np.array(dic[key]["sigma_theta"])
            sigma_pandora = np.array(dic[key]["sigma_theta_pandora"])
            #if len(sigma_pandora) < len(sigma):
            #    sigma_pandora = np.pad(sigma_pandora, (0, len(sigma) - len(sigma_pandora)))
            #elif len(sigma_pandora) > len(sigma):
            #    sigma = np.pad(sigma, (0, len(sigma_pandora) - len(sigma)))
        else:
            sigma = np.array(dic[key]["sigma_phi"])
            sigma_pandora = np.array(dic[key]["sigma_phi_pandora"])
        ax.plot(E, sigma, "--", marker=".", label=key, color=colors[key])
    try:
        ax.plot(E, sigma_pandora, "--", marker=".", label="Pandora", color="blue")
    except:
        print("Error plotting pandora")
    ax.set_xlabel("Energy [GeV]")
    if angle == "theta":
        ax.set_ylabel(r"$\theta$ resolution")
    else:
        ax.set_ylabel(r"$\phi$ resolution")
    ax.set_title(title)
    ax.legend()
    ax.set_xticks([2.5, 10.0, 33.0])
    ax.set_xticklabels(["[0,5]","[5,15]","[15,50]"], fontsize=10)  # Set the corresponding bin range labels
    ax.tick_params(axis='x', which='both', direction='inout')
    if new_plot:
        fig.savefig(
            os.path.join(PATH_store, "angles_" + title + label + "-" + angle + ".pdf"),
            bbox_inches="tight",
        )

def plot_histograms_E(distr_model, distr_pandora, photons_dic, ax_distr, i, title, distr_model_reco=None):
    if type(distr_model) == torch.Tensor:
        distr_model = distr_model.numpy()
        distr_pandora = distr_pandora.numpy()
        if distr_model_reco is not None:
            distr_model_reco = distr_model_reco.numpy()
    else:
        distr_model = distr_model.values
        distr_pandora = distr_pandora.values
        if distr_model_reco is not None:
            distr_model_reco = distr_model_reco.values
    max_distr_model = np.max(distr_model)
    max_distr_pandora = np.max(distr_pandora)
    # remove everything higher than 2.0 and note the fraction of such events
    mask = distr_model < 3.0
    distr_model = distr_model[mask]
    frac_model_dropped = int(
        (1 - len(distr_model) / len(photons_dic["distributions_model"][i])) * 1000
    )
    mask = distr_pandora < 3.0
    distr_pandora = distr_pandora[mask]
    frac_pandora_dropped = int(
        (1 - len(distr_pandora) / len(photons_dic["distributions_pandora"][i]))
        * 1000
    )
    if "reco" not in title.lower():
        mu = photons_dic["mean"][i]
        sigma = (photons_dic["variance_om"][i]) * mu
        mu_pandora = photons_dic["mean_p"][i]
        sigma_pandora = (photons_dic["variance_om_p"][i]) * mu
    else:
        mu = photons_dic["mean_reco"][i]
        sigma = (photons_dic["variance_om_reco"][i]) * mu
        mu_pandora = photons_dic["mean_p_reco"][i]
        sigma_pandora = (photons_dic["variance_om_p_reco"][i]) * mu
        #if distr_model_reco is not None:
        #    #mu_baseline = photons_dic["mean_baseline"][i]
        #    #sigma_baseline = (photons_dic["variance_om_baseline"][i]) * mu_baseline
    if type(mu) == torch.tensor:
        mu = mu.item()
        sigma = sigma.item()
    ax_distr[i].hist(
        distr_model,
        bins=np.arange(0, 3, 1e-2),
        color="red",
        label=r"ML $\mu={} \sigma / \mu={}$".format(round(mu, 2), round(sigma, 2)),
        histtype="step",
        density=True
    )
    ax_distr[i].hist(
        distr_pandora,
        bins=np.arange(0, 3, 1e-2),
        color="blue",
        label=r"Pandora $\mu={} \sigma / \mu={}$".format(
            round(mu_pandora, 2), round(sigma_pandora, 2)
        ),
        histtype="step",
        density=True
    )
    if distr_model_reco is not None:
        ax_distr[i].hist(
            distr_pandora,
            bins=np.arange(0, 3, 1e-2),
            color="black",
            label=r"Baseline $\mu={} \sigma / \mu={}$",
            histtype="step",
            density=True
        )
    # ALSO PLOT MU AND SIGMA #
    ax_distr[i].axvline(mu, color="red", linestyle="-", ymin=0.95, ymax=1.0)
    ax_distr[i].axvline(
        mu + sigma, color="red", linestyle="--", ymin=0.95, ymax=1.0
    )
    ax_distr[i].axvline(
        mu - sigma, color="red", linestyle="--", ymin=0.95, ymax=1.0
    )
    ax_distr[i].axvline(mu_pandora, color="blue", linestyle="-", ymin=0.95, ymax=1.0)
    ax_distr[i].axvline(
        mu_pandora + sigma_pandora, color="blue", linestyle="--", ymin=0.95, ymax=1.0
    )
    ax_distr[i].axvline(
        mu_pandora - sigma_pandora, color="blue", linestyle="--", ymin=0.95, ymax=1.0
    )
    if not "reco" in title.lower():
        ax_distr[i].set_xlabel("$E_{pred.} / E_{true}$")
    else:
        ax_distr[i].set_xlabel("$E_{reco,pred.} / E_{reco,true}$")
    ax_distr[i].set_xlim([0, 3])
    ax_distr[i].set_title( title
        #f"{label1} {photons_dic['energy_resolutions'][i]:.2f} GeV"
    )
    ax_distr[i].legend()
    ax_distr[i].grid()
    ax_distr[i].set_yscale("log")

def plot_one_label(
    title,
    photons_dic,
    y_axis,
    PATH_store,
    label1,
    reco,
    tracks="",
    fig=None,
    ax=None,
    save=True,
    plot_pandora=True,
    plot_baseline=True,
    color=None,
    pandora_label="Pandora",
    ax_distr=None,
    ax_distr_reco=None
):
    if reco == "":
        label_add = " raw"
        label_add_pandora = " corrected"
    else:
        label_add = " raw"
        label_add_pandora = " raw"
    colors_list = ["#FF0000", "#00FF00", "#0000FF"]
    if color is not None:
        colors_list[1] = color
    savefigs = ax_distr is None
    if savefigs:
        fig_distr, ax_distr = plt.subplots(
            len(photons_dic["energy_resolutions"]), 1, figsize=(8, 12), sharex=True
        )
        fig_distr_reco, ax_distr_reco = plt.subplots(
            len(photons_dic["energy_resolutions"]), 1, figsize=(8, 12), sharex=True
        )

        #if title == "Event Energy Resolution":
        #    fig_distr, ax_distr = plt.subplots(
        #        len(photons_dic["energy_resolutions" + reco]),
        #        1,
        #        figsize=(14, 10),
        #        sharex=True,
        #    )
        if not type(ax_distr) == list and not type(ax_distr) == np.ndarray:
            ax_distr = [ax_distr]
            ax_distr_reco = [ax_distr_reco]
    for i in range(len(photons_dic["energy_resolutions" + reco])):
        distr_model = photons_dic["distributions_model"][i]
        distr_pandora = photons_dic["distributions_pandora"][i]
        distr_model_reco = photons_dic["distributions_model_reco"][i]
        plot_histograms_E(distr_model, distr_pandora, photons_dic, ax_distr, i, label1 + f" {photons_dic['energy_resolutions'][i]:.2f} GeV " + " (EC)")
        distr_model = photons_dic["distributions_model_reco"][i]
        distr_pandora = photons_dic["distributions_pandora_reco"][i]
        plot_histograms_E(distr_model, distr_pandora, photons_dic, ax_distr_reco, i, label1 + f" {photons_dic['energy_resolutions'][i]:.2f} GeV " + " (reco)")
    if savefigs:
        fig_distr.tight_layout()
        fig_distr_reco.tight_layout()
    if fig is None or ax is None:
        fig, ax = plt.subplots(2, 1, figsize=(14, 10), sharex=False)
    j = 0
    ax[1].set_xlabel("Energy [GeV]", fontsize=30)
    ax[0].set_xlabel("Energy [GeV]", fontsize=30)
    # ax[row_i, j].set_xscale("log")
    ax[0].set_title(title, fontsize=30)
    ax[0].grid()
    ax[1].grid()
    ax[1].set_yscale("log")
    """f y_axis == "mean":
        # error is the mean error
        errors = photons_dic["mean_errors"]
        pandora_errors = photons_dic["mean_errors_p"]
    else:
        errors = photons_dic["variance_errors"]
        pandora_errors = photons_dic["variance_errors_p"]"""
    for a in ax:
        a.errorbar(
            photons_dic["energy_resolutions" + reco],
            photons_dic[y_axis + reco],
            # yerr=errors,
            color=colors_list[1],
            # edgecolors=colors_list[1],
            label=label1,
            marker="x",
            markersize=8,
            linestyle="None",
            # error color
            ecolor=colors_list[1],
            capsize=5,
        )
        if plot_pandora:
            a.errorbar(
                photons_dic["energy_resolutions_p" + reco],
                photons_dic[y_axis + "_p" + reco],
                # yerr=pandora_errors,
                color=colors_list[2],
                # edgecolors=colors_list[2],
                label=pandora_label,
                marker="x",
                markersize=8,
                capsize=5,
                ecolor=colors_list[2],
                linestyle="None",
            )

    if title == "Electromagnetic Resolution" or title == "Hadronic Resolution":
        if reco == "":
            for a in ax:
                if plot_baseline:
                    a.scatter(
                        photons_dic["energy_resolutions"],
                        photons_dic["variance_om_baseline"],
                        facecolors="black",
                        edgecolors="black",
                        marker=".",
                        label="Baseline",
                        s=50,
                    )
            dic0_fit = get_fit(
                photons_dic["energy_resolutions"], photons_dic["variance_om_baseline"]
            )

        dic1_fit = get_fit(
            photons_dic["energy_resolutions" + reco], photons_dic[y_axis + reco]
        )
        dic1_fit_pandora = get_fit(
            photons_dic["energy_resolutions_p" + reco],
            photons_dic[y_axis + "_p" + reco],
        )
        if reco == "":
            fits_l1 = [
                # dic0_fit,
                dic1_fit,
                # dic1_fit_pandora,
            ]
            color_list_fits_l1 = [
                # "black",
                colors_list[1],
                # colors_list[2],
            ]
            line_type_fits_l1 = ["-"]  # , "-", "-."]
            if plot_baseline:
                fits_l1.append(dic0_fit)
                color_list_fits_l1.append("black")
                line_type_fits_l1.append("-")
            if plot_pandora:
                fits_l1.append(dic1_fit_pandora)
                color_list_fits_l1.append(colors_list[2])
                line_type_fits_l1.append("-.")
            for a in ax:
                plot_fit(fits_l1, line_type_fits_l1, color_list_fits_l1, ax=a)
        if reco == "_reco":
            plt.yscale("log")
        else:
            if title == "Electromagnetic Resolution":
                ymax = 0.3
            else:
                ymax = 0.6
            ax[0].set_ylim([0, ymax])
            ax[1].set_ylim([0, ymax])
        ax[0].set_xlim([0, 55])
        ax[1].set_xlim([0, 55])
        ylabel = r"$\frac{\sigma_{E}}{\langle E} \rangle}$"
        ax[0].set_ylabel(ylabel, fontsize=30)
        ax[1].set_ylabel(ylabel, fontsize=30)
    else:
        ylabel = r"$\langle E_{reco} \rangle / E_{true}$"
        ax[0].set_ylabel(ylabel, fontsize=30)
        ax[1].set_ylabel(ylabel, fontsize=30)
    # loc="upper right",
    # plt.tick_params(axis="both", which="major", labelsize=40)
    ax[0].tick_params(axis="both", which="major", labelsize=30)
    ax[1].tick_params(axis="both", which="major", labelsize=30)
    if title == "Electromagnetic Response" or title == "Hadronic Response":
        ax[0].set_ylim([0.6, 1.4])
        ax[1].set_ylim([0.6, 1.4])
    ax[0].legend(fontsize=20) #, bbox_to_anchor=(1.05, 1), loc="upper left")
    if savefigs:
        fig_distr.tight_layout()
        fig_distr.savefig(
            os.path.join(PATH_store, title + reco + label1 + tracks + "_E_corr_distributions.pdf"),
            bbox_inches="tight",
        )
        fig_distr_reco.tight_layout()
        fig_distr_reco.savefig(
            os.path.join(PATH_store, title + reco + label1 + tracks + "_E_reco_distributions.pdf"),
            bbox_inches="tight",
        )
    if save:
        fig.tight_layout()
        fig.savefig(
            PATH_store + title + reco + label1 + tracks + "_v1.pdf", bbox_inches="tight"
        )



def plot_histograms(
    title,
    photons_dic,
    fig_distr,
    ax_distr,
    plot_pandora,
    prefix="ML ",
    color="blue",
    normalize=True,
):
    assert title == "Event Energy Resolution" # Fix
    # if title == "Event Energy Resolution":
    #    fig_distr, ax_distr = plt.subplots(len(photons_dic["energy_resolutions"]), 1, figsize=(14, 10), sharex=True)
    # if not type(ax_distr) == list and not type(ax_distr) == np.ndarray:
    #    ax_distr = [ax_distr]
    distr_model = photons_dic["distributions_model"][0]
    distr_pandora = photons_dic["distributions_pandora"][0]
    if type(distr_model) == torch.Tensor:
        distr_model = distr_model.numpy()
        distr_pandora = distr_pandora.numpy()
    else:
        distr_model = distr_model.values
        distr_pandora = distr_pandora.values
    # max_distr_model = np.max(distr_model)
    # max_distr_pandora = np.max(distr_pandora)
    # remove everything higher than 2.0 and note the fraction of such events
    mask = distr_model < 2.0
    distr_model = distr_model[mask]
    mask = distr_pandora < 2.0
    distr_pandora = distr_pandora[mask]
    mu = photons_dic["mean"][0]
    sigma = (photons_dic["variance_om"][0]) * mu
    mu_pandora = photons_dic["mean_p"][0]
    sigma_pandora = (photons_dic["variance_om_p"][0]) * mu
    ax_distr.hist(
        distr_model,
        bins=np.arange(0, 3, 1e-2),
        color=color,
        label=prefix + r"$\mu={} \sigma/\mu={}$".format(round(mu, 2), round(sigma/mu, 2)),
        alpha=0.5,
        histtype="step",
        density=normalize,
    )
    if plot_pandora:
        ax_distr.hist(
            distr_pandora,
            bins=np.arange(0, 3, 1e-2),
            color="red",
            label=r"Pandora $\mu={} \sigma/\mu={}$".format(
                round(mu_pandora, 2), round(sigma_pandora/mu_pandora, 2)
            ),
            alpha=0.5,
            histtype="step",
            density=normalize,
        )
    # ALSO PLOT MU AND SIGMA #
    ax_distr.axvline(mu, color=color, linestyle="-", ymin=0.95, ymax=1.0)
    ax_distr.axvline(mu + sigma, color=color, linestyle="--", ymin=0.95, ymax=1.0)
    ax_distr.axvline(mu - sigma, color=color, linestyle="--", ymin=0.95, ymax=1.0)
    ax_distr.axvline(mu_pandora, color="red", linestyle="-", ymin=0.95, ymax=1.0)
    ax_distr.axvline(
        mu_pandora + sigma_pandora, color="red", linestyle="--", ymin=0.95, ymax=1.0
    )
    ax_distr.axvline(
        mu_pandora - sigma_pandora, color="red", linestyle="--", ymin=0.95, ymax=1.0
    )
    # variance_om
    ax_distr.set_xlabel("$E_{reco} / E_{true}$")
    ax_distr.set_xlim([0, 2])
    ax_distr.set_title(f"{title}")
    ax_distr.legend()
    ax_distr.set_yscale("log")
    fig_distr.tight_layout()

colors = {"ML": "red", "ML GTC": "green"}
def plot_full_comparison(photons_dic, electrons_dic, hadrons_dic, hadrons_dic2, neutrons, protons, path):
    matplotlib.rcParams.update({'font.size': 22})
    SMALL_SIZE = 15
    MEDIUM_SIZE = 15
    BIGGER_SIZE = 22
    plt.rc('font', size=SMALL_SIZE)
    plt.rc('axes', titlesize=SMALL_SIZE)
    plt.rc('axes', labelsize=MEDIUM_SIZE)
    plt.rc('xtick', labelsize=SMALL_SIZE)
    plt.rc('ytick', labelsize=SMALL_SIZE)
    plt.rc('legend', fontsize=SMALL_SIZE)
    plt.rc('figure', titlesize=BIGGER_SIZE)
    dics = [electrons_dic, hadrons_dic, neutrons, photons_dic, protons, hadrons_dic2]
    pids = ["11", "130", "2112", "22", "2212", "211"]
    pid_names = {"11": "$e^\pm$", "130": "$K_L$", "2112": "Neutrons", "22": "$\gamma$", "211": "$\pi^\pm$", "2212": "Protons"}
    fig_distr, ax_distr = plt.subplots(6, 4, figsize=(14, 14/4*6))
    default_key= "ML"
    for i, dic in enumerate(dics):
        bins = [0, 5, 15, 50]
        bin_labels = [f"[{bins[i]},{bins[i + 1]}]" for i in range(len(bins) - 1)]
        ax_distr[i, 0].plot(dic[default_key]["energy_resolutions_p"], dic[default_key]["variance_om_p_reco"] / dic[default_key]["energy_resolutions_p"], ".--", c="blue", label="Pandora")
        for key in dic:
            ax_distr[i, 0].plot(dic[key]["energy_resolutions"], dic[key]["variance_om_reco"] / dic[key]["energy_resolutions"], ".--", c=colors[key], label=key)
        # ax_distr[i, 0].plot(dic["energy_resolutions"], dic["variance_om_baseline"] / dic["energy_resolutions"], ".--", c="k", label="Baseline")
        ax_distr[i, 0].set_xticks([2.5, 10.0, 33.0])
        ax_distr[i, 0].set_xticklabels(bin_labels, fontsize=10)  # Set the corresponding bin range labels
        ax_distr[i, 0].tick_params(axis='x', which='both', direction='inout')
        ax_distr[i, 0].set_xlabel("Energy [GeV]", fontsize=SMALL_SIZE)
        ax_distr[i, 0].grid()
        ax_distr[i, 0].legend()
        ax_distr[i, 0].set_title(pid_names[pids[i]])
        ax_distr[i, 1].plot(dic[default_key]["energy_resolutions_p"], dic[default_key]["variance_om_p"] / dic[default_key]["energy_resolutions_p"], ".--", c="blue", label="Pandora")
        ax_distr[i, 1].plot(dic[default_key]["energy_resolutions"], dic[default_key]["variance_om_baseline"] / dic[default_key]["energy_resolutions"], ".--", c="k", label="Baseline")
        for key in dic:
            ax_distr[i, 1].plot(dic[key]["energy_resolutions"], dic[key]["variance_om"] / dic[key]["energy_resolutions"], ".--", c=colors[key], label=key)
        ax_distr[i, 1].set_xlabel("Energy [GeV]", fontsize=SMALL_SIZE)
        ax_distr[i, 1].set_title(pid_names[pids[i]])
        ax_distr[i, 1].set_ylabel("$\sigma_E / E$")
        ax_distr[i, 1].set_xticks([2.5, 10.0, 33.0])
        ax_distr[i, 1].set_xticklabels(bin_labels, fontsize=10)  # Set the corresponding bin range labels
        ax_distr[i, 1].tick_params(axis='x', which='both', direction='inout')
        ax_distr[i, 0].set_ylabel("$\sigma_{E_{reco}} / E_{reco}$")
        ax_distr[i, 1].grid()
        ax_distr[i, 1].legend()
    for j, angle in enumerate(["theta", "phi"]):
        if len(photons_dic[default_key]["distr_phi"]) > 0:
            ax_angle = ax_distr[pids.index("22"), j+2]
            plot_sigma_angle_vs_energy(photons_dic, path, "photons", angle, "Photons", ax=ax_angle)
            ax_angle.grid(1)
        if len(neutrons[default_key]["distr_phi"]) > 0:
            ax_angle = ax_distr[pids.index("2112"), j+2]
            plot_sigma_angle_vs_energy(neutrons, path, "neutrons", angle, "Neutrons", ax=ax_angle)
            ax_angle.grid(1)
        if len(hadrons_dic[default_key]["distr_phi"]) > 0:
            ax_angle = ax_distr[pids.index("130"), j+2]
            plot_sigma_angle_vs_energy(hadrons_dic, path, "KL", angle, "$K_L$", ax=ax_angle)
            ax_angle.grid(1)
        if len(hadrons_dic2[default_key]["distr_phi"]) > 0:
            ax_angle = ax_distr[pids.index("211"), j+2]
            plot_sigma_angle_vs_energy(hadrons_dic2, path, "Pions", angle, "$\pi^\pm$", ax=ax_angle)
            ax_angle.grid(1)
        if len(electrons_dic[default_key]["distr_phi"]) > 0:
            ax_angle = ax_distr[pids.index("11"), j+2]
            plot_sigma_angle_vs_energy(electrons_dic, path, "electrons", angle, "$e^\pm$", ax=ax_angle)
            ax_angle.grid(1)
        if len(protons[default_key]["distr_phi"]) > 0:
            ax_angle = ax_distr[pids.index("2212"), j+2]
            plot_sigma_angle_vs_energy(protons, path, "protons", angle, "Protons", ax=ax_angle)
            ax_angle.grid(1)
    
    SMALL_SIZE = 22
    MEDIUM_SIZE = 22
    BIGGER_SIZE = 22

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)
    fig_distr.tight_layout()
    fig_distr.savefig(path, bbox_inches="tight")
