"""Optimize the heat sink."""

# python libraries
import os

# 3rd party libraries
import pandas as pd

# own libraries
import hct
import dct


# specify input parameters
project_name = "2025-01-31_example"

# load project file paths
filepaths = dct.Optimization.load_filepaths(os.path.abspath(os.path.join(os.curdir, project_name)))

for (_, _, file_name_list) in os.walk('../../HCT_heat_sink_computation_toolbox/hct/data'):
    fan_list = file_name_list

config = hct.OptimizationParameters(
    heat_sink_study_name="3_dimensions",
    heat_sink_optimization_directory=filepaths.heat_sink,
    height_c_list=[0.02, 0.08],
    width_b_list=[0.02, 0.08],
    length_l_list=[0.08, 0.20],
    height_d_list=[0.001, 0.003],
    number_fins_n_list=[5, 20],
    thickness_fin_t_list=[1e-3, 5e-3],
    fan_list=fan_list,
    t_ambient=40,
    area_min=None,
    number_directions=3
)

hct.Optimization.start_proceed_study(config=config, number_trials=1000)

df_heat_sink = hct.Optimization.study_to_df(config)
hct.Optimization.df_plot_pareto_front(df_heat_sink, (50, 60))

# df_heat_sink["heat_sink_volume"] = df_heat_sink["r_th_heat_sink"].apply(lambda r_th_max: df_heat_sink.loc[df_heat_sink["values_1"] < r_th_max]["values_0"].nsmallest(n=1))
# df_heat_sink["total_volume"] = df_heat_sink["transformer_volume"] + df_heat_sink["inductor_volume"] + df_heat_sink["heat_sink_volume"]
