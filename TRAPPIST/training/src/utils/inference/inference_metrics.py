import matplotlib
import torch
#matplotlib.rc("font", size=25)
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from scipy import asarray as ar, exp
from src.utils.pid_conversion import our_to_pandora_mapping, pandora_to_our_mapping
import pandas as pd

def calculate_eff(sd, log_scale=False, pandora=False):
    if log_scale:
        bins = np.exp(np.arange(np.log(0.1), np.log(80), 0.3))
    else:
        bins = [0, 5, 10, 35, 50]
    eff = []
    energy_eff = []
    errors = []
    for i in range(len(bins) - 1):
        bin_i = bins[i]
        bin_i1 = bins[i + 1]
        mask_above = sd.reco_showers_E.values <= bin_i1
        mask_below = sd.reco_showers_E.values > bin_i
        mask = mask_below * mask_above
        number_of_non_reconstructed_showers = np.sum(
            np.isnan(sd.pred_showers_E.values)[mask]
        )
        total_showers = len(sd.true_showers_E.values[mask])
        if pandora:
            number_of_non_reconstructed_showers = np.sum(
                np.isnan(sd.pandora_calibrated_E.values)[mask]
            )
            total_showers = len(sd.pandora_calibrated_E.values[mask])
        if total_showers > 0:
            eff.append(
                (total_showers - number_of_non_reconstructed_showers) / total_showers
            )
            energy_eff.append((bin_i1 + bin_i) / 2)
            n_total = total_showers
            n_r = total_showers-number_of_non_reconstructed_showers
            error = (n_r/(n_total**2)*np.sqrt(n_total))**2+(1/n_total*np.sqrt(n_r))**2
            error = np.sqrt(error)
            errors.append(error)
    return eff, energy_eff, errors


def calculate_fakes(sd, matched, log_scale=False, pandora=False, id=None):
    if log_scale:
        bins_fakes = np.exp(np.arange(np.log(0.1), np.log(80), 0.3))
    else:
        bins_fakes = [0, 5, 15, 35, 50]
    fake_rate = []
    energy_fakes = []
    fake_percent_energy = []
    fake_percent_reco_energy = []
    id_our = pandora_to_our_mapping[id]
    fake_errors = []
    for i in range(len(bins_fakes) - 1):
        bin_i = bins_fakes[i]
        bin_i1 = bins_fakes[i + 1]
        if pandora:
            mask_above = sd.pred_showers_E.values <= bin_i1
            mask_below = sd.pred_showers_E.values > bin_i
            mask_pid = sd.pandora_pid.isin(our_to_pandora_mapping[id_our])
            mask_pid_truth = mask_above * mask_below * sd.pid.isin(our_to_pandora_mapping[id_our]) # The matched ones
            mask = mask_below * mask_above * mask_pid
            fakes = np.sum(np.isnan(sd.pid)[mask])
            non_fakes_mask = ~np.isnan(sd.pid)[mask]
            fakes_mask = np.isnan(sd.pid)[mask]
            energy_in_fakes = np.sum(sd.pandora_calibrated_pfo[mask].values[fakes_mask])
            reco_in_fakes = np.sum(sd.pred_showers_E[mask].values[fakes_mask])
            total_E_meas = np.sum(sd.pandora_calibrated_pfo.values[mask])
            total_E_reco = np.sum(sd.pred_showers_E.values[mask])
            total_showers = len(sd.pred_showers_E.values[mask]) # The true showers
        else:
            mask_above = sd.pred_showers_E.values <= bin_i1
            mask_below = sd.pred_showers_E.values > bin_i
            mask_pid = sd.pred_pid_matched == id_our
            mask_pid_truth = sd.pid.isin(our_to_pandora_mapping[id_our]) # The matched ones!
            mask = mask_below * mask_above * mask_pid
            fakes = np.sum(np.isnan(sd.pid)[mask])
            total_showers_true = len(sd.pred_showers_E.values[mask_pid_truth])
            total_showers = sum(~np.isnan(sd.pred_showers_E.values[mask]))
            fakes_mask = np.isnan(sd.pid)[mask]
            energy_in_fakes = np.sum(sd.calibrated_E[mask].values[fakes_mask])
            reco_in_fakes = np.sum(sd.pred_showers_E[mask].values[fakes_mask])

            #non_fakes_mask = ~np.isnan(sd.pid)[mask]
            total_E_meas = np.sum(sd.calibrated_E.values[mask])
            total_E_reco = np.sum(sd.pred_showers_E.values[mask])
        if total_showers > 0:
            # print(fakes, np.mean(sd.pred_energy_hits_raw[mask]))
            fake_rate.append(fakes / total_showers)
            n_r = fakes
            n_total = total_showers
            error = (n_r/(n_total**2)*np.sqrt(n_total))**2+(1/n_total*np.sqrt(n_r))**2
            error = np.sqrt(error)
            print(fakes, n_total, pandora, error)
            fake_errors.append(error)
            energy_fakes.append((bin_i1 + bin_i) / 2)
            fake_percent_energy.append(energy_in_fakes / total_E_meas)
            fake_percent_reco_energy.append(reco_in_fakes / total_E_reco)
    return fake_rate, energy_fakes, fake_percent_energy, fake_percent_reco_energy, fake_errors 


def calculate_response(matched, pandora, log_scale=False):
    if log_scale:
        bins = np.exp(np.arange(np.log(0.1), np.log(80), 0.3))
    else:
        bins = np.arange(0, 51, 2)

    bins_plot_histogram = [5, 6, 10, 20]
    bins_per_binned_E = np.arange(0, 3, 0.001)

    mean = []
    variance_om = []
    mean_true_rec = []
    variance_om_true_rec = []
    energy_resolutions = []
    energy_resolutions_reco = []
    dic_histograms = {}
    for i in range(len(bins) - 1):
        bin_i = bins[i]
        bin_i1 = bins[i + 1]
        mask_above = (
            matched["reco_showers_E"] <= bin_i1
        )  # true_showers_E, reco_showers_E
        mask_below = matched["reco_showers_E"] > bin_i
        mask_check = matched["pred_showers_E"] > 0
        mask = mask_below * mask_above * mask_check

        pred_e = matched.calibrated_E[mask]
        true_rec = matched.reco_showers_E[mask]
        true_e = matched.true_showers_E[mask]
        if pandora:
            pred_e_corrected = matched.pandora_calibrated_E[mask]
        else:
            pred_e_corrected = matched.calibrated_E[mask]
        if np.sum(mask) > 0:  # if the bin is not empty
            e_over_rec = pred_e / true_rec
            if i in bins_plot_histogram:
                dic_histograms[str(i) + "reco"] = e_over_rec
                dic_histograms[str(i) + "reco_baseline"] = true_rec
                dic_histograms[str(i) + "pred_corr_e"] = pred_e_corrected
                dic_histograms[str(i) + "true_baseline"] = true_e
                dic_histograms[str(i) + "pred_e"] = pred_e
            mean_predtored, variance_om_true_rec_ = obtain_MPV_and_68(
                e_over_rec, bins_per_binned_E
            )
            # mean_predtored = np.mean(e_over_rec)
            # variance_om_true_rec_ = np.var(e_over_rec) / mean_predtored
            mean_true_rec.append(mean_predtored)
            variance_om_true_rec.append(variance_om_true_rec_)
            energy_resolutions_reco.append((bin_i1 + bin_i) / 2)
    # TODO change the pred_showers_E to the pandora calibrated E and the calibrated E for the model pandora_calibrated_E
    if pandora:
        bins_per_binned_E = np.arange(0, 3, 0.005)
    else:
        bins_per_binned_E = np.arange(0, 3, 0.005)
    for i in range(len(bins) - 1):
        bin_i = bins[i]
        bin_i1 = bins[i + 1]
        mask_above = matched["true_showers_E"] <= bin_i1
        mask_below = matched["true_showers_E"] > bin_i
        mask_check = matched["pred_showers_E"] > 0
        mask = mask_below * mask_above * mask_check
        true_e = matched.true_showers_E[mask]
        true_rec = matched.reco_showers_E[mask]
        if pandora:
            pred_e = matched.pandora_calibrated_E[mask]
        else:
            pred_e = matched.calibrated_E[mask]
        if np.sum(mask) > 0:  # if the bin is not empty
            e_over_true = pred_e / true_e
            e_rec_over_true = true_rec / true_e
            if i in bins_plot_histogram:
                dic_histograms[str(i) + "true"] = e_over_true
                dic_histograms[str(i) + "reco_showers"] = e_rec_over_true
            mean_predtotrue, var_predtotrue = obtain_MPV_and_68(
                e_over_true, bins_per_binned_E
            )
            # mean_predtotrue, var_predtotrue = get_sigma_gaussian(e_over_true,bins_per_binned_E)
            # mean_predtotrue = np.mean(e_over_true)
            # var_predtotrue = np.var(e_over_true) / mean_predtotrue
            print(
                "bin i ",
                bins[i],
                mean_predtotrue,
                var_predtotrue,
                np.mean(e_over_true),
                np.var(e_over_true) / np.mean(e_over_true),
            )
            mean.append(mean_predtotrue)
            variance_om.append(var_predtotrue)
            energy_resolutions.append((bin_i1 + bin_i) / 2)

    return (
        mean,
        variance_om,
        mean_true_rec,
        variance_om_true_rec,
        energy_resolutions,
        energy_resolutions_reco,
        dic_histograms,
    )


def get_sigma_gaussian(e_over_reco, bins_per_binned_E, epsilon=0.01, return_gaussian=False, return_divided=True):
    #mpv, std = obtain_MPV_and_68(e_over_reco, bins_per_binned_E)
    #return mpv, std, None, None
    hist, bin_edges = np.histogram(e_over_reco, bins=bins_per_binned_E, density=True)

    if not return_gaussian:
        mu, sigma_over_mu = obtain_MPV_and_68(e_over_reco, bins_per_binned_E, epsilon=epsilon, no_divide=not return_divided)
        return mu, sigma_over_mu, 0,0
    # Calculating the Gaussian PDF values given Gaussian parameters and random variable X
    def gaus(X, C, X_mean, sigma):
        return C * exp(-((X - X_mean) ** 2) / (2 * sigma**2))
    n = len(hist)
    x_hist = np.zeros((n), dtype=float)
    for ii in range(n):
        x_hist[ii] = (bin_edges[ii + 1] + bin_edges[ii]) / 2
    y_hist = hist
    if (torch.tensor(hist) == 0).all():
        return 0,0
    mean = sum(x_hist * y_hist) / sum(y_hist)
    sigma = sum(y_hist * (x_hist - mean) ** 2) / sum(y_hist)
    # cut 1% of highest vals
    #e_over_reco_filtered = np.sort(e_over_reco)
    #e_over_reco_filtered = e_over_reco_filtered[:int(len(e_over_reco_filtered) * 0.99)]
    #mean = np.mean(e_over_reco_filtered)
    #sigma = np.std(e_over_reco_filtered)
    try:
        param_optimised, param_covariance_matrix = curve_fit(
            gaus, x_hist, y_hist, p0=[max(y_hist), mean, sigma], maxfev=10000
        )
    except:
        print("Error! Using this")
        return mean, sigma/mean, 0.001, 0.001 # dummy errors temporarily
    if param_optimised[2] < 0:
        param_optimised[2] = sigma
    if param_optimised[1] < 0:
       param_optimised[1] = mean  # due to some weird fitting errors
    #assert param_optimised[1] >= 0
    #assert param_optimised[2] >= 0
    errors = np.sqrt(np.diag(param_covariance_matrix))
    # sigma_over_E_error = errors[2] / param_optimised[1]
    if return_divided:
        return param_optimised[1], param_optimised[2] / param_optimised[1], errors[1], errors[2] / param_optimised[1]
    return param_optimised[1], param_optimised[2], errors[1], errors[2] / param_optimised[1]

def obtain_MPV_and_68(data_for_hist, bins_per_binned_E, epsilon=0.01, no_divide=False):
    hist, bin_edges = np.histogram(data_for_hist, bins=bins_per_binned_E, density=True)
    ind_max_hist = np.argmax(hist)
    MPV = (bin_edges[ind_max_hist] + bin_edges[ind_max_hist + 1]) / 2
    std68, low, high = get_std68(hist, bin_edges, epsilon=epsilon)
    if std68 == 0.4 and low == 0.2 and high == 1.0:
        # It didn't fit correctly as it's too close to a delta function
        if type(data_for_hist) == pd.Series:
            data_for_hist = data_for_hist.values
        MPV, std68 = torch.mean(torch.tensor(data_for_hist)).item(), torch.std(torch.tensor(data_for_hist)).item()
    if no_divide:
        return MPV, std68
    return MPV, std68 / MPV


def get_std68(theHist, bin_edges, percentage=0.683, epsilon=0.005):
    # theHist, bin_edges = np.histogram(data_for_hist, bins=bins, density=True)
    wmin = 0.2
    wmax = 1.0

    weight = 0.0
    points = []
    sums = []

    # fill list of bin centers and the integral up to those point
    for i in range(len(bin_edges) - 1):
        weight += theHist[i] * (bin_edges[i + 1] - bin_edges[i])
        points.append([(bin_edges[i + 1] + bin_edges[i]) / 2, weight])
        sums.append(weight)
    low = wmin
    high = wmax
    width = 100
    for i in range(len(points)):
        for j in range(i, len(points)):
            wy = points[j][1] - points[i][1]
            if abs(wy - percentage) < epsilon:
                wx = points[j][0] - points[i][0]
                if wx < width:
                    low = points[i][0]
                    high = points[j][0]
                    width = wx
                    # ii = i
                    # jj = j

    return 0.5 * (high - low), low, high


def calculate_purity_containment(matched, log_scale=False):
    if log_scale:
        bins = np.exp(np.arange(np.log(0.1), np.log(80), 0.3))
    else:
        bins = np.arange(0, 51, 2)
    fce_energy = []
    fce_var_energy = []
    energy_ms = []

    purity_energy = []
    purity_var_energy = []
    fce = matched["e_pred_and_truth"] / matched["reco_showers_E"]
    purity = matched["e_pred_and_truth"] / matched["pred_showers_E"]
    for i in range(len(bins) - 1):
        bin_i = bins[i]
        bin_i1 = bins[i + 1]
        mask_above = matched["reco_showers_E"] <= bin_i1
        mask_below = matched["reco_showers_E"] > bin_i
        mask_check = matched["pred_showers_E"] > 0
        mask = mask_below * mask_above * mask_check
        fce_e = np.mean(fce[mask])
        fce_var = np.var(fce[mask])
        purity_e = np.mean(purity[mask])
        purity_var = np.var(purity[mask])
        if np.sum(mask) > 0:
            fce_energy.append(fce_e)
            fce_var_energy.append(fce_var)
            energy_ms.append((bin_i1 + bin_i) / 2)
            purity_energy.append(purity_e)
            purity_var_energy.append(purity_var)
    return (
        fce_energy,
        fce_var_energy,
        energy_ms,
        purity_energy,
        purity_var_energy,
    )


def obtain_metrics(sd, matched, pandora=False, log_scale=False):
    eff, energy_eff = calculate_eff(sd, log_scale)
    fake_rate, energy_fakes = calculate_fakes(sd, matched, log_scale)

    (
        mean,
        variance_om,
        mean_true_rec,
        variance_om_true_rec,
        energy_resolutions,
        energy_resolutions_reco,
        dic_histograms,
    ) = calculate_response(matched, pandora, log_scale)

    (
        fce_energy,
        fce_var_energy,
        energy_ms,
        purity_energy,
        purity_var_energy,
    ) = calculate_purity_containment(matched, log_scale)

    dict = {
        "energy_eff": energy_eff,
        "eff": eff,
        "energy_fakes": energy_fakes,
        "fake_rate": fake_rate,
        "mean": mean,
        "variance_om": variance_om,
        "mean_true_rec": mean_true_rec,
        "variance_om_true_rec": variance_om_true_rec,
        "fce_energy": fce_energy,
        "fce_var_energy": fce_var_energy,
        "energy_ms": energy_ms,
        "purity_energy": purity_energy,
        "purity_var_energy": purity_var_energy,
        "energy_resolutions": energy_resolutions,
        "energy_resolutions_reco": energy_resolutions_reco,
        "dic_histograms": dic_histograms,
    }
    return dict
