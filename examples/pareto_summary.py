"""Optimize the heat sink."""

# python libraries
import os

# 3rd party libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# own libraries
import hct
import dct


# specify input parameters
project_name = "2025-03-12_debug"
circuit_study_name = "circuit_01"
inductor_study_name_list = ["inductor_01"]
stacked_transformer_study_name_list = ["transformer_01"]
heat_sink_study_name = "3_dimensions"

# task = "generate_pareto_summary"
task = "plot_pareto_summary"

# Allocate instance CircuitOptimization
circuit_optimization: dct.CircuitOptimization = dct.CircuitOptimization()

# load project file paths
filepaths = circuit_optimization.load_filepaths(os.path.abspath(os.path.join(os.curdir, project_name)))

if task == "generate_pareto_summary":

    # load heat sink
    hs_config_filepath = os.path.join(filepaths.heat_sink, f"{heat_sink_study_name}.pkl")
    hs_config = hct.Optimization.load_config(hs_config_filepath)
    df_hs = hct.Optimization.study_to_df(hs_config)
    df_hs.to_csv(f"{filepaths.heat_sink}/df_hs.csv")

    # load summarized results without heat sink
    df_wo_hs = pd.read_csv(f"{filepaths.heat_sink}/result_df.csv")

    # generate full summary
    print(df_hs.loc[df_hs["values_1"] < 1]["values_0"].nsmallest(n=1).values[0])
    df_wo_hs["heat_sink_volume"] = df_wo_hs["r_th_heat_sink"].apply(
        lambda r_th_max: df_hs.loc[df_hs["values_1"] < r_th_max]["values_0"].nsmallest(n=1).values[0] \
        if np.any(df_hs.loc[df_hs["values_1"] < r_th_max]["values_0"].nsmallest(n=1).values) else None)

    df_wo_hs["total_volume"] = df_wo_hs["transformer_volume"] + df_wo_hs["inductor_volume"] + df_wo_hs["heat_sink_volume"]

    # save full summary
    df_wo_hs.to_csv(f"{filepaths.heat_sink}/df_summary.csv")

elif task == "plot_pareto_summary":
    df = pd.read_csv(f"{filepaths.heat_sink}/df_summary.csv")

    df_filtered = circuit_optimization.filter_df(df, x="total_volume", y="total_mean_loss",
                                                 factor_min_dc_losses=0.001, factor_max_dc_losses=10)

    dct.global_plot_settings_font_latex()
    fig = plt.figure(figsize=(80/25.4, 60/25.4), dpi=1000)
    plt.scatter(df["total_volume"] * 1e6, df["total_mean_loss"], s=10, c=dct.colors()["black"], alpha=0.1, label="Design")
    plt.scatter(df_filtered["total_volume"] * 1e6, df_filtered["total_mean_loss"], s=10, c=dct.colors()["red"], alpha=1, label="Best designs")

    plt.xlabel(r"$V_\mathrm{DAB}$ / cm³")
    plt.ylabel(r"$P_\mathrm{DAB,mean}$ / W")
    plt.grid()
    # plt.xlim(140, 200)
    # plt.ylim(45, 55)
    # plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()
