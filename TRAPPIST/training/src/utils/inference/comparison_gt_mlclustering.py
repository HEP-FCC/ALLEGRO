from src.utils.inference.event_metrics import get_response_for_event_energy
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

def plot_mass(sd_hgb, sd_hgb_gt, sd_pandora, PATH_store, label_ML="ML", label_ML_GTC="ML GTC",
              color_ML_GTC="green", filename="mass_resolution_comp.pdf"):
    perfect_pid = False
    mass_zero = False
    ML_pid = True
    matched_all = {label_ML: sd_hgb, label_ML_GTC: sd_hgb_gt}
    matched_pandora = sd_pandora
    event_res_dic = {} 
    for key in matched_all:
            matched_ = matched_all[key]
            event_res_dic[key] = get_response_for_event_energy(
                    matched_pandora, matched_, perfect_pid=perfect_pid, mass_zero=mass_zero, ML_pid=ML_pid
                )
    old_font_size = matplotlib.rcParams['font.size']
    matplotlib.rcParams.update({'font.size': 22})
    fig, ax = plt.subplots(1, 2,figsize=(16, 8))
    # set fontsize to 20
    ax[0].set_xlabel(r"$m_{pred}/m_{true}$")
    bins = np.linspace(0, 2, 200)
    ax[0].hist(
        event_res_dic[label_ML]["mass_over_true_model"],
        bins=bins,
        histtype="step",
        label=label_ML + " $\mu$={}".format(
            round((event_res_dic[label_ML]["mean_mass_model"]), 4)
        )+"\n"+"$\sigma/\mu$={}".format(round((event_res_dic[label_ML]["var_mass_model"]), 4),
        ),
        color="red",
        density=True,
    )
    # ax[0].hist(
    #     event_res_dic[label_ML_GTC]["mass_over_true_model"],
    #     bins=bins,
    #     histtype="step",
    #     label=label_ML_GTC + " $\mu$={}".format(
    #         round((event_res_dic[label_ML_GTC]["mean_mass_model"]), 4)
    #     )+"\n"+"$\sigma/\mu$={}".format(round((event_res_dic[label_ML_GTC]["var_mass_model"]), 4),
    #     ),
    #     color=color_ML_GTC,
    #     density=True,
    # )
    ax[0].hist(
        event_res_dic[label_ML]["mass_over_true_pandora"],
        bins=bins,
        histtype="step",
        label="Pandora $\mu$={}".format(
            round((event_res_dic[label_ML]["mean_mass_pandora"]), 4)
        )+"\n"+"$\sigma/\mu$={}".format(round((event_res_dic[label_ML_GTC]["var_mass_pandora"]), 4),
        ),
        color="blue",
        density=True,
    )
    ax[0].grid()
    ax[0].legend(loc='upper left')
    #ax.set_xlim([0, 10])
    mean_e_over_true_pandora, sigma_e_over_true_pandora = round(event_res_dic[label_ML]["mean_energy_over_true_pandora"], 4), round(
        event_res_dic[label_ML]["var_energy_over_true_pandora"], 4)
    mean_e_over_true, sigma_e_over_true = round(event_res_dic[label_ML]["mean_energy_over_true"], 4), round(
        event_res_dic[label_ML]["var_energy_over_true"], 4)
    mean_e_over_true_gtc, sigma_e_over_true_gtc = round(event_res_dic[label_ML_GTC]["mean_energy_over_true"], 4), round(
        event_res_dic[label_ML_GTC]["var_energy_over_true"], 4)
    ax[1].hist(event_res_dic[label_ML]["energy_over_true"], bins=bins, histtype="step",
                # label=r"ML $\mu$={} $\sigma / \mu$={}".format(mean_e_over_true, sigma_e_over_true),
                color="red",
                label=label_ML + " $\mu$={}".format(
                    mean_e_over_true
                )+"\n"+"$\sigma/\mu$={}".format(sigma_e_over_true
                ),
                density=True)
    # ax[1].hist(event_res_dic[label_ML_GTC]["energy_over_true"], bins=bins, histtype="step",
    #             # label=r"ML $\mu$={} $\sigma / \mu$={}".format(mean_e_over_true, sigma_e_over_true),
    #             color=color_ML_GTC,
    #             label=label_ML_GTC + " $\mu$={}".format(
    #                 mean_e_over_true_gtc
    #             )+"\n"+"$\sigma/\mu$={}".format(sigma_e_over_true_gtc
    #             ),
    #             density=True)
    ax[1].hist(event_res_dic[label_ML]["energy_over_true_pandora"], bins=bins, histtype="step",
                        # label=r"Pandora $\mu$={} $\sigma / \mu$={}".format(mean_e_over_true_pandora,
                        #                                                     sigma_e_over_true_pandora),
                        label="Pandora  $\mu$={}".format(
                            mean_e_over_true_pandora
                        )+"\n"+"$\sigma/\mu$={}".format(sigma_e_over_true_pandora
                        ),
                        color="blue",
                        density=True)

    ax[1].grid(1)
    ax[1].set_xlabel(r"$E_{vis,pred} / E_{vis,true}$")
    ax[1].legend(loc='upper left')

    fig.tight_layout()

    import os
    fig.savefig(os.path.join(PATH_store, filename), bbox_inches="tight")
    matplotlib.rcParams.update({'font.size': old_font_size})



def plot_mass_note(sd_hgb, sd_hgb_gt, sd_pandora, PATH_store):
    perfect_pid=False
    mass_zero=False
    ML_pid=True
    matched_all = {"ML": sd_hgb, "ML GTC": sd_hgb_gt}
    matched_pandora = sd_pandora
    event_res_dic = {} 
    for key in matched_all:
            matched_ = matched_all[key]
            event_res_dic[key] = get_response_for_event_energy(
                    matched_pandora, matched_, perfect_pid=perfect_pid, mass_zero=mass_zero, ML_pid=ML_pid
                )
    # old_font_size = matplotlib.rcParams['font.size']
    # matplotlib.rcParams.update({'font.size': 30})
    fig= plt.figure( figsize=(8,8))
    ax = fig.add_subplot(111)
    # set fontsize to 20
    ax.set_xlabel(r"$M_{pred}/M_{true}$")
    bins = np.linspace(0, 2, 100)
    ax.hist(
        event_res_dic["ML"]["mass_over_true_model"],
        bins=bins,
        histtype="step",
        label="ML "+"\n"+"$\sigma/\mu$={}".format(round((event_res_dic["ML"]["var_mass_model"]), 2),
        ),
        color="red",
        density=True,
        linewidth=2,
    )
    ax.hist(
        event_res_dic["ML GTC"]["mass_over_true_model"],
        bins=bins,
        histtype="step",
        label="ML GT "+"\n"+"$\sigma/\mu$={}".format(round((event_res_dic["ML GTC"]["var_mass_model"]), 2),
        ),
        color="green",
        density=True,
        linewidth=2,
    )
    
    ax.hist(
        event_res_dic["ML"]["mass_over_true_pandora"],
        bins=bins,
        histtype="step",
        label="Pandora "+"\n"+"$\sigma/\mu$={}".format(round((event_res_dic["ML GTC"]["var_mass_pandora"]), 2),
        ),
        color="blue",
        density=True,
        linewidth=2, 
    )
    ax.grid()
    from matplotlib.lines import Line2D
    custom_line1 = Line2D([0], [0], color="red",label="ML "+"\n"+"$\sigma/\mu$={}".format(round((event_res_dic["ML"]["var_mass_model"]), 2),
        ))
    custom_line_gt = Line2D([0], [0], color="green",label="ML GT "+"\n"+"$\sigma/\mu$={}".format(round((event_res_dic["ML GTC"]["var_mass_model"]), 2),
        ))
    custom_line_pandora = Line2D([0], [0], color="blue",label="Pandora "+"\n"+"$\sigma/\mu$={}".format(round((event_res_dic["ML GTC"]["var_mass_pandora"]), 2),
        ))
    ax.legend(handles=[custom_line1, custom_line_pandora, custom_line_gt],loc='upper left')
    # ax.legend(loc='upper left')
    ax.set_xlim([0.5, 1.5])
    ax.set_ylabel("Event fraction")
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    size_font=30
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = size_font
    plt.rcParams['axes.labelsize'] = size_font
    plt.rcParams['xtick.labelsize'] = size_font
    plt.rcParams['ytick.labelsize'] = size_font
    plt.rcParams['legend.fontsize'] = 30
    fig.tight_layout()

    import os
    fig.savefig(os.path.join(PATH_store, "mass_resolution_comp.pdf"), bbox_inches="tight")
    # matplotlib.rcParams.update({'font.size': old_font_size})
