
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib
matplotlib.rc('font', size=15)
import matplotlib.pyplot as plt


def track_cluser_eff_(sd_hgb):
    bins = [0, 5, 15,  51]
    pids_with_tracks = []
    track_cluser_eff = []
    track_cluster_bad_assignation = []
    neutrals_with_tracks = []
    for i in range(len(bins) - 1):
        current_pids_with_tracks = {} # e.g. 11 -> 50% have tracks
        bin_i = bins[i]
        bin_i1 = bins[i + 1]
        mask_above = sd_hgb["true_showers_E"] <= bin_i1
        mask_below = sd_hgb["true_showers_E"] > bin_i
        mask_check = ~pd.isna(sd_hgb["pred_showers_E"])
        mask = mask_below * mask_above * mask_check
        correct_track_found = sd_hgb["is_track_correct"][mask*(sd_hgb.distance_to_cluster_MC<0.21)].values/sd_hgb["is_track_in_MC"][mask*(sd_hgb.distance_to_cluster_MC<0.21)].values
        correct_track_found = correct_track_found[~np.isnan(correct_track_found)]>0
        correct_track_found_percent = np.mean(correct_track_found)
        track_cluser_eff.append(correct_track_found_percent)
        track_found = (sd_hgb["is_track_correct"][mask].values+1*(sd_hgb["is_track_in_cluster"][mask].values>0))==1
        track_in_MC = sd_hgb["is_track_in_MC"][mask].values
        track_found_ = (track_found[~np.isnan(track_in_MC)]>0)*(track_in_MC[~np.isnan(track_in_MC)]>0)
        track_found_percent = np.mean(track_found_)
        track_cluster_bad_assignation.append(track_found_percent)

        track_found = (sd_hgb["is_track_in_cluster"][mask].values)==1
        track_in_MC = sd_hgb["is_track_in_MC"][mask].values
        track_found = (track_found[~np.isnan(track_in_MC)]>0)*(track_in_MC[~np.isnan(track_in_MC)]==0)
        track_found_percent = np.mean(track_found)
        for pid in sd_hgb.pid[mask].unique():
            mask_pid = sd_hgb.pid[mask] == pid
            track_in_MC = sd_hgb["is_track_in_MC"][mask].values[mask_pid]
            current_pids_with_tracks[pid] = np.mean(track_in_MC>0)*100
        neutrals_with_tracks.append(track_found_percent)
        pids_with_tracks.append(current_pids_with_tracks)
        print("Energy range ", bin_i, bin_i1, " - % of each PID that have MC track:", current_pids_with_tracks)
    return track_cluser_eff, track_cluster_bad_assignation, neutrals_with_tracks


def plot_track_assignation_eval(sd_hgb, sd_hgb_p, path):
    matplotlib.rcParams["font.size"] = 11
    track_cluser_eff, track_cluster_bad_assignation, neutrals_with_tracks = track_cluser_eff_(sd_hgb)
    track_cluser_eff_p, track_cluster_bad_assignation_p, neutrals_with_tracks_p = track_cluser_eff_(sd_hgb_p)
    # Put ML and pandora on the same histogram
    fig_distr, ax_distr = plt.subplots(3, 1, figsize=(3, 7))
    # Define x-axis labels and positions
    categories = ["[0, 5] GeV", "[5, 15] GeV", "[15, 51] GeV"]
    x = np.arange(len(categories))  # Positions for the groups
    width = 0.35  # Width of the bars

    # Create subplots
    fig_distr, ax_distr = plt.subplots(3, 1, figsize=(6, 10))

    # Plot for track_cluser_eff
    ax_distr[0].bar(x - width / 2, track_cluser_eff, width, label="ML", color="red")
    ax_distr[0].bar(x + width / 2, track_cluser_eff_p, width, label="Pandora", color="blue")
    ax_distr[0].set_title("track_cluser_eff")
    ax_distr[0].set_xticks(x)
    ax_distr[0].set_xticklabels(categories)
    ax_distr[0].legend()

    # Plot for track_cluster_bad_assignation
    ax_distr[1].bar(x - width / 2, track_cluster_bad_assignation, width, label="ML", color="red")
    ax_distr[1].bar(x + width / 2, track_cluster_bad_assignation_p, width, label="Pandora", color="blue")
    ax_distr[1].set_title("track_cluster_bad_assignation")
    ax_distr[1].set_xticks(x)
    ax_distr[1].set_xticklabels(categories)
    ax_distr[1].legend()

    # Plot for neutrals_with_tracks
    ax_distr[2].bar(x - width / 2, neutrals_with_tracks, width, label="ML", color="red")
    ax_distr[2].bar(x + width / 2, neutrals_with_tracks_p, width, label="Pandora", color="blue")
    ax_distr[2].set_title("neutrals_with_tracks")
    ax_distr[2].set_xticks(x)
    ax_distr[2].set_xticklabels(categories)
    ax_distr[2].legend()
    '''fig_distr, ax_distr = plt.subplots(3,2,figsize=(8, 12))
    ax_distr[0,0].bar(["[0,5]", "[5,15]",  "[15,51]"], track_cluser_eff,  label="ML")
    ax_distr[0,1].bar(["[0,5]", "[5,15]",  "[15,51]"], track_cluser_eff_p,  label="Pandora")
    ax_distr[0,0].set_title("track_cluser_eff")
    ax_distr[1,0].bar(["[0,5]", "[5,15]",  "[15,51]"], track_cluster_bad_assignation,  label="ML")
    ax_distr[1,1].bar(["[0,5]", "[5,15]",  "[15,51]"], track_cluster_bad_assignation_p,  label="Pandora")
    ax_distr[1,0].set_title("track_cluster_bad_assignation")
    ax_distr[2,0].bar(["[0,5]", "[5,15]",  "[15,51]"], neutrals_with_tracks,  label="ML")
    ax_distr[2,1].bar(["[0,5]", "[5,15]",  "[15,51]"], neutrals_with_tracks_p,  label="Pandora")
    ax_distr[2,0].set_title("neutrals_with_tracks")'''
    fig_distr.tight_layout()
    fig_distr.savefig(os.path.join(path, "track_cluster_eval.pdf"), bbox_inches="tight")
