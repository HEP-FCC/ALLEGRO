import matplotlib

matplotlib.rc("font", size=35)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
import seaborn as sns
from scipy.optimize import curve_fit

hep.style.use("CMS")
# colors_list = ["#fff7bc", "#fec44f", "#d95f0e"]
colors_list = ["#fde0dd", "#c994c7", "#dd1c77"]  # color list poster neurips
marker_size = 20


def plot_for_jan(neutrals_only, dic1, dic2, dict_1, dict_2, dict_3, colors_list):
    marker_size = 15
    log_scale = False
    fig = plt.figure()
    if dic1:
        plt.scatter(
            dict_1["energy_resolutions"],
            dict_1["variance_om"],
            facecolors=colors_list[0],
            edgecolors=colors_list[0],
            label="HDBSCAN",
            s=70,
        )
    if dic2:
        plt.scatter(
            dict_2["energy_resolutions"],
            dict_2["variance_om"],
            facecolors=colors_list[1],
            edgecolors=colors_list[1],
            label="GNN",
            s=70,
        )
        plt.scatter(
            dict_3["energy_resolutions"],
            dict_3["variance_om"],
            facecolors=colors_list[2],
            edgecolors=colors_list[2],
            label="Pandora",
            marker="^",
            s=70,
        )
    if log_scale:
        plt.xlabel("Log True Energy [GeV]")
    else:
        plt.xlabel("True Energy [GeV]")
    plt.ylabel("Resolution")
    plt.grid()
    plt.legend(loc="upper right")
    plt.xlim([0, 21])
    plt.yscale("log")
    fig.tight_layout(pad=2.0)
    if neutrals_only:
        fig.savefig(
            "/afs/cern.ch/work/m/mgarciam/private/mlpf/summ_results/Pandora_mix/histograms_energy_neutrals_only_jan.png",
            bbox_inches="tight",
        )
    else:
        fig.savefig(
            "/afs/cern.ch/work/m/mgarciam/private/mlpf/summ_results/Pandora_mix/histograms_energy_jan.png",
            bbox_inches="tight",
        )


def plot_eff(ax, dic1, dict_1, dic2, dict_2, dic3, dict_3, log_scale, i, j, idx_start):
    if dic1:
        ax[i, j].scatter(
            dict_1["energy_eff"][idx_start:],
            dict_1["eff"][idx_start:],
            facecolors=colors_list[0],
            edgecolors=colors_list[0],
            marker="o",
            label="HDBSCAN",
            s=80,
        )
    if dic2:
        ax[i, j].scatter(
            dict_2["energy_eff"][idx_start:],
            dict_2["eff"][idx_start:],
            facecolors=colors_list[1],
            edgecolors=colors_list[1],
            label="GNN",
            s=70,
        )
    if dic3:
        ax[i, j].scatter(
            dict_3["energy_eff"][idx_start:],
            dict_3["eff"][idx_start:],
            facecolors=colors_list[2],
            edgecolors=colors_list[2],
            label="Pandora",
            marker="^",
            s=70,
        )
        # ax[i, j].axvline(x=0.5, color="b")
    if log_scale:
        ax[i, j].set_xlabel("Log True Energy [GeV]")
        ax[i, j].set_xscale("log")
    else:
        ax[i, j].set_xlabel("True Energy [GeV]")
    ax[i, j].set_ylabel("Efficiency")
    ax[i, j].grid()
    ax[i, j].legend(loc="lower right")
    return ax


def plot_fakes(
    ax, dic1, dict_1, dic2, dict_2, dic3, dict_3, log_scale, i, j, idx_start
):

    if dic1:
        ax[i, j].scatter(
            dict_1["energy_fakes"][idx_start:],
            dict_1["fake_rate"][idx_start:],
            facecolors=colors_list[0],
            edgecolors=colors_list[0],
            label="HDBSCAN",
            marker="o",
            s=80,
        )
    if dic2:
        ax[i, j].scatter(
            dict_2["energy_fakes"][idx_start:],
            dict_2["fake_rate"][idx_start:],
            facecolors=colors_list[1],
            edgecolors=colors_list[1],
            label="GNN",
            s=70,
        )
    if dic3:
        ax[i, j].scatter(
            dict_3["energy_fakes"][idx_start:],
            dict_3["fake_rate"][idx_start:],
            facecolors=colors_list[2],
            edgecolors=colors_list[2],
            label="Pandora",
            s=70,
            marker="^",
        )
        # ax[i, j].axvline(x=0.5, color="b", label="axvline")
    if log_scale:
        ax[i, j].set_xlabel("Log Reconstructed Energy [GeV]")
        ax[i, j].set_xscale("log")
    else:
        ax[i, j].set_xlabel("Reconstructed Energy [GeV]")
    ax[i, j].set_ylabel("Fake rate")
    ax[i, j].grid()
    ax[i, j].set_yscale("log")
    ax[i, j].legend(loc="upper right")
    return ax


def plot_response(
    ax, dic1, dict_1, dic2, dict_2, dic3, dict_3, log_scale, i, j, idx_start
):
    # Energy resolution is parametrized
    if dic1:
        ax[i, j].scatter(
            dict_1["energy_resolutions_reco"][idx_start:],
            dict_1["mean_true_rec"][idx_start:],
            facecolors=colors_list[0],
            edgecolors=colors_list[0],
            marker="o",
            label="HDBSCAN",
            s=80,
        )
    if dic2:
        ax[i, j].scatter(
            dict_2["energy_resolutions_reco"][idx_start:],
            dict_2["mean_true_rec"][idx_start:],
            facecolors=colors_list[1],
            edgecolors=colors_list[1],
            label="GNN",
            s=70,
        )
    if dic3:
        ax[i, j].scatter(
            dict_3["energy_resolutions_reco"][idx_start:],
            dict_3["mean_true_rec"][idx_start:],
            facecolors=colors_list[2],
            edgecolors=colors_list[2],
            label="Pandora",
            marker="^",
            s=70,
        )
    if log_scale:
        ax[i, j].set_xlabel("Log Reco Energy [GeV]")
        ax[i, j].set_xscale("log")
    else:
        ax[i, j].set_xlabel("Reco Energy [GeV]")
    ax[i, j].set_ylabel("Response")
    ax[i, j].grid()
    ax[i, j].legend(loc="lower right")
    return ax


def plot_fit_energy_resolution(
    ax, dic1, dict_1, dic2, dict_2, dic3, dict_3, log_scale, i, j, reco=False
):
    if reco:
        energies = np.array(dict_1["energy_resolutions_reco"])
        errors = np.array(dict_1["variance_om_true_rec"])
    else:
        energies = np.array(dict_1["energy_resolutions"])
        errors = np.array(dict_1["variance_om"])
    mask = (energies > 1.0) * (errors < 0.38)
    energies = energies[mask]
    errors = errors[mask]
    popt, pcov = curve_fit(resolution, energies, errors)
    xdata = np.arange(1, 51, 0.1)
    ax[i, j].plot(
        xdata,
        resolution(xdata, *popt),
        "-",
        c=colors_list[1],
        label="fit GNN: a=%5.3f, b=%5.3f, c=%5.3f" % tuple(popt),
    )

    if reco:
        energies = np.array(dict_3["energy_resolutions_reco"])
        errors = np.array(dict_3["variance_om_true_rec"])
    else:
        energies = np.array(dict_3["energy_resolutions"])
        errors = np.array(dict_3["variance_om"])
    mask = (energies > 1.0) * (errors < 0.38)
    energies = energies[mask]
    errors = errors[mask]
    popt, pcov = curve_fit(resolution, energies, errors)
    xdata = np.arange(1, 51, 0.1)
    ax[i, j].plot(
        xdata,
        resolution(xdata, *popt),
        "-",
        c=colors_list[2],
        label="fit Pandora: a=%5.3f, b=%5.3f, c=%5.3f" % tuple(popt),
    )
    ax[i, j].legend(loc="upper right")
    return ax


def resolution(E, a, b, c):
    return (a**2 / E + c**2 + b**2 / E**2) ** 0.5


def plot_resolution(
    ax, dic1, dict_1, dic2, dict_2, dic3, dict_3, log_scale, i, j, idx_start
):
    if dic1:
        ax[i, j].scatter(
            dict_1["energy_resolutions_reco"][idx_start:],
            dict_1["variance_om_true_rec"][idx_start:],
            facecolors=colors_list[0],
            edgecolors=colors_list[0],
            label="HDBSCAN",
            marker="o",
            s=80,
        )
    if dic2:
        ax[i, j].scatter(
            dict_2["energy_resolutions_reco"][idx_start:],
            dict_2["variance_om_true_rec"][idx_start:],
            facecolors=colors_list[1],
            edgecolors=colors_list[1],
            label="GNN",
            s=70,
        )
    if dic3:
        ax[i, j].scatter(
            dict_3["energy_resolutions_reco"][idx_start:],
            dict_3["variance_om_true_rec"][idx_start:],
            facecolors=colors_list[2],
            edgecolors=colors_list[2],
            label="Pandora",
            marker="^",
            s=70,
        )
        # ax[i, j].axvline(x=0.5, color="b", label="axvline")
    if log_scale:
        ax[i, j].set_xlabel("Log Reco Energy [GeV]")
        ax[i, j].set_xscale("log")
    else:
        ax[i, j].set_xlabel("Reco Energy [GeV]")
    ax[i, j].set_ylabel("Resolution")
    ax[i, j].grid()
    ax[i, j].set_yscale("log")
    ax[i, j].legend(loc="upper right")
    # ax[1, 1].set_ylim([0, 1])
    return ax


def plot_response_trueE(
    ax, dic1, dict_1, dic2, dict_2, dic3, dict_3, log_scale, i, j, idx_start
):
    if dic1:
        ax[i, j].scatter(
            dict_1["energy_resolutions"][idx_start:],
            dict_1["mean"][idx_start:],
            facecolors=colors_list[0],
            edgecolors=colors_list[0],
            label="HDBSCAN",
            marker="o",
            s=80,
        )
    if dic2:
        ax[i, j].scatter(
            dict_2["energy_resolutions"][idx_start:],
            dict_2["mean"][idx_start:],
            facecolors=colors_list[1],
            edgecolors=colors_list[1],
            label="GNN",
            s=70,
        )
    if dic3:
        ax[i, j].scatter(
            dict_3["energy_resolutions"][idx_start:],
            dict_3["mean"][idx_start:],
            facecolors=colors_list[2],
            edgecolors=colors_list[2],
            label="Pandora",
            marker="^",
            s=70,
        )
        # ax[i, j].axvline(x=0.5, color="b", label="axvline")
    if log_scale:
        ax[i, j].set_xlabel("Log True Energy [GeV]")
        ax[i, j].set_xscale("log")
    else:
        ax[i, j].set_xlabel("True Energy [GeV]")
    ax[i, j].set_ylabel("Resolution")
    ax[i, j].set_ylabel("Response")
    ax[i, j].grid()
    ax[i, j].legend(loc="lower right")
    # ax[2, 0].set_ylim([0, 1.5])
    return ax


def plot_resolution_trueE(
    ax, dic1, dict_1, dic2, dict_2, dic3, dict_3, log_scale, i, j, idx_start
):
    if dic1:
        ax[i, j].scatter(
            dict_1["energy_resolutions"][idx_start:],
            dict_1["variance_om"][idx_start:],
            facecolors=colors_list[0],
            edgecolors=colors_list[0],
            label="HDBSCAN",
            marker="o",
            s=80,
        )
    if dic2:
        ax[i, j].scatter(
            dict_2["energy_resolutions"][idx_start:],
            dict_2["variance_om"][idx_start:],
            facecolors=colors_list[1],
            edgecolors=colors_list[1],
            label="GNN",
            s=70,
        )
    if dic3:
        ax[i, j].scatter(
            dict_3["energy_resolutions"][idx_start:],
            dict_3["variance_om"][idx_start:],
            facecolors=colors_list[2],
            edgecolors=colors_list[2],
            label="Pandora",
            marker="^",
            s=70,
        )
        # ax[i, j].axvline(x=0.5, color="b", label="axvline")
    if log_scale:
        ax[i, j].set_xlabel("Log True Energy [GeV]")
        ax[i, j].set_xscale("log")
    else:
        ax[i, j].set_xlabel("True Energy [GeV]")
    ax[i, j].set_ylabel("Resolution")
    ax[i, j].set_ylabel("Resolution")
    ax[i, j].set_yscale("log")
    ax[i, j].grid()
    ax[i, j].legend(loc="upper right")
    return ax


def plot_containment(
    ax, dic1, dict_1, dic2, dict_2, dic3, dict_3, log_scale, i, j, idx_start
):
    if dic1:
        ax[i, j].errorbar(
            np.array(dict_1["energy_ms"])[idx_start:],
            np.array(dict_1["fce_energy"])[idx_start:],
            np.array(dict_1["fce_var_energy"])[idx_start:],
            marker="o",
            mec=colors_list[0],
            mfc=colors_list[0],
            ecolor=colors_list[0],
            ms=marker_size,
            mew=4,
            linestyle="",
        )
    if dic2:
        ax[i, j].errorbar(
            np.array(dict_2["energy_ms"])[idx_start:],
            np.array(dict_2["fce_energy"])[idx_start:],
            np.array(dict_2["fce_var_energy"])[idx_start:],
            marker=".",
            mfc=colors_list[1],
            mec=colors_list[1],
            ecolor=colors_list[1],
            ms=marker_size,
            mew=4,
            linestyle="",
        )
    if dic3:
        # ax[i, j].axvline(x=0.5, color="b", label="axvline")
        ax[i, j].errorbar(
            np.array(dict_3["energy_ms"])[idx_start:],
            np.array(dict_3["fce_energy"])[idx_start:],
            np.array(dict_3["fce_var_energy"])[idx_start:],
            marker="^",
            mfc=colors_list[2],
            mec=colors_list[2],
            ecolor=colors_list[2],
            ms=marker_size,
            mew=4,
            linestyle="",
        )
    if log_scale:
        ax[i, j].set_xlabel("Log Reco Energy [GeV]")
        ax[i, j].set_xscale("log")
    else:
        ax[i, j].set_xlabel("Reco Energy [GeV]")
    ax[i, j].set_ylabel("Containment")
    ax[i, j].grid()
    # ax[3, 0].set_yscale("log")
    return ax


def plot_purity(
    ax, dic1, dict_1, dic2, dict_2, dic3, dict_3, log_scale, i, j, idx_start
):
    if dic1:
        ax[i, j].errorbar(
            np.array(dict_1["energy_ms"])[idx_start:],
            np.array(dict_1["purity_energy"])[idx_start:],
            np.array(dict_1["purity_var_energy"])[idx_start:],
            marker="o",
            mec=colors_list[0],
            mfc=colors_list[0],
            ecolor=colors_list[0],
            ms=marker_size,
            mew=4,
            linestyle="",
        )
    if dic2:
        ax[i, j].errorbar(
            np.array(dict_2["energy_ms"])[idx_start:],
            np.array(dict_2["purity_energy"])[idx_start:],
            np.array(dict_2["purity_var_energy"])[idx_start:],
            marker=".",
            mec=colors_list[1],
            mfc=colors_list[1],
            ecolor=colors_list[1],
            ms=marker_size,
            mew=4,
            linestyle="",
        )
    if dic3:
        # ax[i, j].axvline(x=0.5, color="b", label="axvline")
        ax[i, j].errorbar(
            np.array(dict_3["energy_ms"])[idx_start:],
            np.array(dict_3["purity_energy"])[idx_start:],
            np.array(dict_3["purity_var_energy"])[idx_start:],
            marker="^",
            mec=colors_list[2],
            mfc=colors_list[2],
            ecolor=colors_list[2],
            ms=marker_size,
            mew=4,
            linestyle="",
        )

    if log_scale:
        ax[i, j].set_xlabel("Log Reco Energy [GeV]")
        ax[i, j].set_xscale("log")
    else:
        ax[i, j].set_xlabel("Reco Energy [GeV]")
    ax[i, j].set_ylabel("Purity")
    ax[i, j].grid()
    # ax[3, 1].set_yscale("log")
    # ax[3, 1].set_ylim([0, 1.5])
    return ax


def plot_metrics(
    neutrals_only,
    dic1,
    dic2,
    dic3,
    dict_1,
    dict_2,
    dict_3,
    colors_list,
    PATH_store,
    log_scale,
):
    marker_size = 15
    idx_start = 5
    fig, ax = plt.subplots(4, 4, figsize=(9 * 4, 8 * 4))
    # efficiency plot
    ax = plot_eff(ax, dic1, dict_1, dic2, dict_2, dic3, dict_3, True, 0, 0, idx_start)
    ax = plot_eff(ax, dic1, dict_1, dic2, dict_2, dic3, dict_3, False, 0, 2, idx_start)

    # fake rates
    ax = plot_fakes(ax, dic1, dict_1, dic2, dict_2, dic3, dict_3, True, 0, 1, idx_start)
    ax = plot_fakes(
        ax, dic1, dict_1, dic2, dict_2, dic3, dict_3, False, 0, 3, idx_start
    )

    # response
    ax = plot_response(
        ax, dic1, dict_1, dic2, dict_2, dic3, dict_3, True, 1, 0, idx_start
    )
    ax = plot_response(
        ax, dic1, dict_1, dic2, dict_2, dic3, dict_3, False, 1, 2, idx_start
    )

    # resolution
    ax = plot_resolution(
        ax, dic1, dict_1, dic2, dict_2, dic3, dict_3, True, 1, 1, idx_start
    )
    ax = plot_resolution(
        ax, dic1, dict_1, dic2, dict_2, dic3, dict_3, False, 1, 3, idx_start
    )
    # ax = plot_fit_energy_resolution(
    #     ax, dic1, dict_1, dic2, dict_2, dic3, dict_3, log_scale, 1, 1, reco=True
    # )
    # ax = plot_fit_energy_resolution(
    #     ax, dic1, dict_1, dic2, dict_2, dic3, dict_3, log_scale, 1, 3, reco=True
    # )

    # response true_e
    ax = plot_response_trueE(
        ax, dic1, dict_1, dic2, dict_2, dic3, dict_3, True, 2, 0, idx_start
    )
    ax = plot_response_trueE(
        ax, dic1, dict_1, dic2, dict_2, dic3, dict_3, False, 2, 2, idx_start
    )

    # resolution true_e
    ax = plot_resolution_trueE(
        ax, dic1, dict_1, dic2, dict_2, dic3, dict_3, True, 2, 1, idx_start
    )
    ax = plot_resolution_trueE(
        ax, dic1, dict_1, dic2, dict_2, dic3, dict_3, False, 2, 3, idx_start
    )
    # ax = plot_fit_energy_resolution(
    #     ax, dic1, dict_1, dic2, dict_2, dic3, dict_3, log_scale, 2, 1, reco=False
    # )
    # ax = plot_fit_energy_resolution(
    #     ax, dic1, dict_1, dic2, dict_2, dic3, dict_3, log_scale, 2, 3, reco=False
    # )
    # containment
    ax = plot_containment(
        ax, dic1, dict_1, dic2, dict_2, dic3, dict_3, True, 3, 0, idx_start
    )
    ax = plot_containment(
        ax, dic1, dict_1, dic2, dict_2, dic3, dict_3, False, 3, 2, idx_start
    )
    # containment
    ax = plot_purity(
        ax, dic1, dict_1, dic2, dict_2, dic3, dict_3, True, 3, 1, idx_start
    )
    ax = plot_purity(
        ax, dic1, dict_1, dic2, dict_2, dic3, dict_3, False, 3, 3, idx_start
    )

    if neutrals_only:
        fig.savefig(
            PATH_store + "testeq_rec_comp_MVP_68_calibrated_neutrals_compare_to_V8.png",
            bbox_inches="tight",
        )
    else:
        fig.savefig(
            PATH_store + "testeq_rec_comp_MVP_68_calibrated_compare_to_V8.png",
            bbox_inches="tight",
        )


def plot_histograms_energy(
    dic1, dic2, dict_1, dict_2, dict_3, neutrals_only=False, PATH_store=None
):
    dict_2 = dict_1
    bins_plot_histogram = [5, 6, 10, 20]
    bins = np.exp(np.arange(np.log(0.1), np.log(80), 0.3))
    bins = np.arange(0, 51, 2)
    fig, ax = plt.subplots(4, 2, figsize=(18, 25))

    for i in range(0, 4):
        bin_name = bins_plot_histogram[i]
        sns.histplot(
            data=np.array(dict_2["dic_histograms"][str(bin_name) + "reco"]),
            label="MLPF",
            stat="percent",
            color=colors_list[1],
            element="step",
            binwidth=0.02,
            fill=False,
            ax=ax[i, 0],
            linewidth=2,
        )
        sns.histplot(
            dict_3["dic_histograms"][str(bin_name) + "reco"],
            label="Pandora",
            stat="percent",
            color=colors_list[2],
            element="step",
            fill=False,
            ax=ax[i, 0],
            binwidth=0.02,
            linewidth=2,
        )
        sns.histplot(
            dict_2["dic_histograms"][str(bin_name) + "true"],
            label="MLPF",
            stat="percent",
            color=colors_list[1],
            element="step",
            fill=False,
            ax=ax[i, 1],
            binwidth=0.02,
            linewidth=2,
        )
        sns.histplot(
            dict_3["dic_histograms"][str(bin_name) + "true"],
            label="Pandora",
            stat="percent",
            color=colors_list[2],
            element="step",
            fill=False,
            ax=ax[i, 1],
            binwidth=0.02,
            linewidth=2,
        )

        sns.histplot(
            dict_2["dic_histograms"][str(bin_name) + "reco_showers"],
            label="Reco",
            stat="percent",
            color=colors_list[0],
            element="step",
            fill=False,
            ax=ax[i, 1],
            binwidth=0.02,
            linewidth=2,
        )
        ax[i, 1].set_xlabel("E pred / True Energy [GeV]")
        # ax[i, 1].set_xlim([0, 2])
        ax[i, 1].grid()
        ax[i, 1].legend(loc="upper right")
        ax[i, 1].set_title(
            "["
            + str(np.round(bins[bin_name], 2))
            + ", "
            + str(np.round(bins[bin_name + 1], 2))
            + "]"
            + " GeV"
        )
        ax[i, 0].set_xlabel("E pred / Reco Energy [GeV]")
        ax[i, 0].grid()
        ax[i, 0].set_xlim([0, 2])
        ax[i, 0].legend(loc="upper right")
        ax[i, 0].set_title(
            "["
            + str(np.round(bins[bin_name], 2))
            + ", "
            + str(np.round(bins[bin_name + 1], 2))
            + "]"
            + " GeV"
        )
        # ax[i, 0].set_yscale("log")
    fig.tight_layout(pad=2.0)
    if neutrals_only:
        fig.savefig(
            PATH_store + "histograms_energy_neutrals_only_iou_th.png",
            bbox_inches="tight",
        )
    else:
        fig.savefig(
            PATH_store + "histograms_energy_iou_th.png",
            bbox_inches="tight",
        )


def plot_correction(
    dic1, dic2, dict_1, dict_2, dict_3, neutrals_only=False, PATH_store=None
):
    bins = np.exp(np.arange(np.log(0.1), np.log(80), 0.3))
    bins_plot_histogram = [5, 6, 10, 20]
    fig, ax = plt.subplots(4, 3, figsize=(9 * 3, 25))

    for i in range(0, 4):
        bin_name = bins_plot_histogram[i]
        ax[i, 0].scatter(
            dict_3["dic_histograms"][str(bin_name) + "pred_e"],
            dict_3["dic_histograms"][str(bin_name) + "pred_corr_e"],
            label="Pandora",
            color=colors_list[2],
        )
        ax[i, 1].scatter(
            dict_2["dic_histograms"][str(bin_name) + "pred_e"],
            dict_2["dic_histograms"][str(bin_name) + "pred_corr_e"],
            label="MLPF",
            c=colors_list[1],
            marker="^",
        )
        ax[i, 2].scatter(
            dict_2["dic_histograms"][str(bin_name) + "reco_baseline"],
            dict_2["dic_histograms"][str(bin_name) + "true_baseline"],
            label="True",
            color=colors_list[0],
        )

        ax[i, 0].set_xlabel("E pred [GeV]")
        ax[i, 0].set_ylabel("E calibrated[GeV]")
        ax[i, 0].grid()
        # ax[i, 0].set_xlim([0, 2])
        ax[i, 0].legend(loc="upper right")
        ax[i, 0].set_title(
            "["
            + str(np.round(bins[bin_name], 2))
            + ", "
            + str(np.round(bins[bin_name + 1], 2))
            + "]"
            + " GeV"
        )
        ax[i, 1].set_xlabel("E pred [GeV]")
        ax[i, 1].set_ylabel("E calibrated[GeV]")
        ax[i, 1].grid()
        # ax[i, 0].set_xlim([0, 2])
        ax[i, 1].legend(loc="upper right")
        ax[i, 1].set_title(
            "["
            + str(np.round(bins[bin_name], 2))
            + ", "
            + str(np.round(bins[bin_name + 1], 2))
            + "]"
            + " GeV"
        )
        ax[i, 2].set_xlabel("E pred [GeV]")
        ax[i, 2].set_ylabel("E calibrated[GeV]")
        ax[i, 2].grid()
        # ax[i, 0].set_xlim([0, 2])
        ax[i, 2].legend(loc="upper right")
        ax[i, 2].set_title(
            "["
            + str(np.round(bins[bin_name], 2))
            + ", "
            + str(np.round(bins[bin_name + 1], 2))
            + "]"
            + " GeV"
        )
        # ax[i, 0].set_yscale("log")
    fig.tight_layout(pad=2.0)
    if neutrals_only:
        fig.savefig(
            PATH_store + "correction_energy_neutrals_only.png",
            bbox_inches="tight",
        )
    else:
        fig.savefig(
            PATH_store + "correction_energy.png",
            bbox_inches="tight",
        )
