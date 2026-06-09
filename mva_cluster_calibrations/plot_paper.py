#!/usr/bin/env python3
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Nimbus Sans"],   # falls back if not available
    
    "font.size": 14,                    # global base size
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "legend.title_fontsize": 13,
})

def resol_curve_no_noise(E, a, c):
    """
    Resolution in percent:
        sigma/E = a/sqrt(E) ⊕ c
    E in GeV
    """
    return np.sqrt((a / np.sqrt(E))**2 + c**2)

def resol_curve(E, noise, a, c):
    """
    Resolution in percent:
        sigma/E = noise/E ⊕ a/sqrt(E) ⊕ c
    E in GeV
    """
    return np.sqrt((noise / E)**2 + (a / np.sqrt(E))**2 + c**2)

def load_json(fname):
    with open(fname) as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json1", default="EMB_calo_topo_wo_noise_with_fitparams.json", help="First *_with_fitparams.json file")
    parser.add_argument("--json2", default="EMB_calo_topo_wo_noise_with_fitparams_LKrW.json", help="Second *_with_fitparams.json file")
    parser.add_argument("--label1", default="LAr/Pb", help="Label for 1st dataset")
    parser.add_argument("--label2", default="LKr/W", help="Label for 2nd dataset")
    parser.add_argument(
        "--collection",
        default="EMBCaloClusters",
        help="Cluster collection to plot "
             "(EMBCaloClusters, EMBCaloTopoClusters, ...)"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="comparison.pdf"
    )
    args = parser.parse_args()
    data1 = load_json(args.json1)
    data2 = load_json(args.json2)
    coll = args.collection
    label1 = args.label1
    label2 = args.label2
    E1 = np.asarray(data1["energies"]) / 1000.0  # MeV -> GeV
    E2 = np.asarray(data2["energies"]) / 1000.0
    R1 = np.asarray(data1["resolutions_cal"][coll])
    R2 = np.asarray(data2["resolutions_cal"][coll])
    err1 = np.asarray(data1["resolutions_cal_err"][coll])
    err2 = np.asarray(data2["resolutions_cal_err"][coll])
    fit1 = data1["fitparams"][coll]
    fit2 = data2["fitparams"][coll]
    xfit = np.exp(
        np.linspace(
            np.log(min(E1.min(), E2.min())),
            np.log(max(E1.max(), E2.max())),
            500,
        )
    )
    plt.figure(figsize=(8, 6))
    # Data points
    eb1 = plt.errorbar(
        E1, R1, yerr=err1,
        fmt="o",
        label=label1,
        ms=8
    )
    color1 = eb1[0].get_color()
    eb2 = plt.errorbar(
        E2, R2, yerr=err2,
        fmt="s",
        label=label2,
        ms=8
    )
    color2 = eb2[0].get_color()
    # Fit curves
    if len(fit1) == 2:
        yfit1 = resol_curve_no_noise(xfit, *fit1)
        fitlabel1 = (
            rf"{fit1[0]:.1f}%/$\sqrt{{E}}$"
            rf" $\oplus$ {fit1[1]:.1f}%"
        )
    else:
        yfit1 = resol_curve(xfit, *fit1)
        fitlabel1 = (
            rf"{fit1[0]:.3f}/E"
            rf" $\oplus$ {fit1[1]:.1f}%/$\sqrt{{E}}$"
            rf" $\oplus$ {fit1[2]:.1f}%"
        )
    if len(fit2) == 2:
        yfit2 = resol_curve_no_noise(xfit, *fit2)
        fitlabel2 = (
            rf"{fit2[0]:.1f}%/$\sqrt{{E}}$"
            rf" $\oplus$ {fit2[1]:.1f}%"
        )
    else:
        yfit2 = resol_curve(xfit, *fit2)
        fitlabel2 = (
            rf"{fit2[0]:.3f}/E"
            rf" $\oplus$ {fit2[1]:.1f}%/$\sqrt{{E}}$"
            rf" $\oplus$ {fit2[2]:.1f}%"
        )
    fit1_line, = plt.plot(xfit, yfit1, linewidth=2, label=fitlabel1, color=color1)
    fit2_line, = plt.plot(xfit, yfit2, linewidth=2, label=fitlabel2, color=color2)
    plt.xscale("log")
    plt.grid(True, which="both", alpha=0.3)
    # plt.xlabel(r"$E_{\rm true}$ [GeV]")
    plt.xlabel(r"$E$ [GeV]")
    plt.ylabel("Resolution [%]")
    # plt.title(coll)
    plt.legend(handles=[eb1, fit1_line, eb2, fit2_line])
    plt.tight_layout()
    plt.savefig("plots/"+args.output)
    print(f"Saved plots/{args.output}")

if __name__ == "__main__":
    main()
