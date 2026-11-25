
def obtain_metrics_hgcal(sd, matched, ms):
    true_e = matched.truthHitAssignedEnergies
    bins = np.arange(0, 51, 2)
    eff = []
    fake_rate = []
    energy_eff = []
    for i in range(len(bins) - 1):
        bin_i = bins[i]
        bin_i1 = bins[i + 1]
        mask_above = sd.truthHitAssignedEnergies.values <= bin_i1
        mask_below = sd.truthHitAssignedEnergies.values > bin_i
        mask = mask_below * mask_above
        number_of_non_reconstructed_showers = np.sum(
            np.isnan(sd.pred_energy_hits_raw.values)[mask]
        )
        total_showers = len(sd.t_rec_energy.values[mask])
        if total_showers > 0:
            eff.append(
                (total_showers - number_of_non_reconstructed_showers) / total_showers
            )
            energy_eff.append((bin_i1 + bin_i) / 2)
    # fake rate per energy with a binning of 1
    true_e = matched.truthHitAssignedEnergies
    bins_fakes = np.arange(0, 51, 2)
    fake_rate = []
    energy_fakes = []
    total_true_showers = np.sum(
        ~np.isnan(sd.truthHitAssignedEnergies.values)
    )  # the ones where truthHitAssignedEnergies is not nan
    for i in range(len(bins_fakes) - 1):
        bin_i = bins_fakes[i]
        bin_i1 = bins_fakes[i + 1]
        mask_above = sd.pred_energy_hits_raw.values <= bin_i1
        mask_below = sd.pred_energy_hits_raw.values > bin_i
        mask = mask_below * mask_above
        fakes = np.sum(np.isnan(sd.truthHitAssignedEnergies)[mask])
        total_showers = len(sd.pred_energy_hits_raw.values[mask])

        if total_showers > 0:
            # print(fakes, np.mean(sd.pred_energy_hits_raw[mask]))
            fake_rate.append((fakes) / total_true_showers)
            energy_fakes.append((bin_i1 + bin_i) / 2)

    # plot 2 for each energy bin calculate the mean and the variance of the distribution
    mean = []
    variance_om = []
    mean_true_rec = []
    variance_om_true_rec = []
    energy_resolutions = []
    for i in range(len(bins) - 1):
        bin_i = bins[i]
        bin_i1 = bins[i + 1]
        mask_above = ms["e_truth"] <= bin_i1
        mask_below = ms["e_truth"] > bin_i
        mask = mask_below * mask_above
        pred_e = matched.pred_energy_hits_raw[mask]
        true_e = matched.truthHitAssignedEnergies[mask]
        true_rec = ms.e_truth[mask]

        if np.sum(mask) > 0:
            mean_predtotrue = np.mean(pred_e / true_e)
            mean_predtored = np.mean(pred_e / true_rec)
            var_predtotrue = np.var(pred_e / true_e) / mean_predtotrue
            variance_om_true_rec_ = np.var(pred_e / true_rec) / mean_predtored
            mean.append(mean_predtotrue)
            mean_true_rec.append(mean_predtored)
            variance_om.append(var_predtotrue)
            variance_om_true_rec.append(variance_om_true_rec_)
            energy_resolutions.append((bin_i1 + bin_i) / 2)

    bins = np.arange(0, 51, 2)
    fce_energy = []
    fce_var_energy = []
    energy_ms = []

    purity_energy = []
    purity_var_energy = []
    fce = ms["e_pred_and_truth"] / ms["e_truth"]
    purity = ms["e_pred_and_truth"] / ms["e_pred"]
    for i in range(len(bins) - 1):
        bin_i = bins[i]
        bin_i1 = bins[i + 1]
        mask_above = ms["e_truth"] <= bin_i1
        mask_below = ms["e_truth"] > bin_i
        mask = mask_below * mask_above
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

    dict = {
        "energy_eff": energy_eff,
        "eff": eff,
        "energy_fakes": energy_fakes,
        "fake_rate": fake_rate,
        "mean_true_rec": mean,
        "variance_om_true_rec": variance_om,
        "fce_energy": fce_energy,
        "fce_var_energy": fce_var_energy,
        "energy_ms": energy_ms,
        "purity_energy": purity_energy,
        "purity_var_energy": purity_var_energy,
        "energy_resolutions": energy_resolutions,
    }
    return dict

