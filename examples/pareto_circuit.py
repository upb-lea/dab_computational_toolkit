"""Example how to use the design_check for the DAB converter."""
# python libraries
import os

# 3rd party libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# own libraries
import dct

import logging
logging.basicConfig(format='%(levelname)s,%(asctime)s:%(message)s', encoding='utf-8')
logging.getLogger('pygeckocircuits2').setLevel(logging.DEBUG)

design_space = dct.CircuitParetoDesignSpace(
    f_s_min_max_list=[50e3, 300e3],
    l_s_min_max_list=[20e-6, 900e-6],
    l_1_min_max_list=[10e-6, 10e-3],
    l_2__min_max_list=[10e-6, 1e-3],
    n_min_max_list=[3, 7],
    transistor_1_name_list=['CREE_C3M0065100J', 'CREE_C3M0120100J'],
    transistor_2_name_list=['CREE_C3M0060065J', 'CREE_C3M0120065J'],
    c_par_1=16e-12,
    c_par_2=16e-12,
)

output_range = dct.CircuitOutputRange(
    v_1_min_nom_max_list=[690, 700, 710],
    v_2_min_nom_max_list=[175, 235, 295],
    p_min_nom_max_list=[0, 2000, 2200],
    steps_per_direction=19,
)

dab_config = dct.CircuitParetoDabDesign(
    # circuit_study_name='circuit_paper_trial_1',

    # circuit_study_name='fix_white_area_trial_5',
    circuit_study_name='2024-12-17_trial',
    project_directory=os.path.abspath(os.path.join(os.curdir, "2024-12-17_trial")),

    design_space=design_space,
    output_range=output_range
)

action = 'run_new_study'
# action = 'show_study_results'
# action = 'filter_study_results_and_run_gecko'
# action = 'custom'

if action == 'run_new_study':
    dct.Optimization.start_proceed_study(dab_config, 1000)
    dct.Optimization.show_study_results(dab_config)

elif action == 'show_study_results':
    dab_config = dct.Optimization.load_config(dab_config.project_directory, dab_config.circuit_study_name)
    print(f"{dab_config.project_directory=}")
    dab_config.project_directory = dab_config.project_directory.replace("@uni-paderborn.de", "")
    print(f"{dab_config.project_directory=}")
    dct.Optimization.show_study_results(dab_config)
    df = dct.Optimization.study_to_df(dab_config)

elif action == 'filter_study_results_and_run_gecko':
    df = dct.Optimization.study_to_df(dab_config)
    df = df[df["values_0"] == 100]

    df_original = df.copy()

    smallest_dto_list = []
    df_smallest_all = df.nsmallest(n=1, columns=["values_1"])
    df_smallest = df.nsmallest(n=1, columns=["values_1"])

    smallest_dto_list.append(dct.Optimization.df_to_dab_dto_list(dab_config, df_smallest))
    print(f"{np.shape(df)=}")

    for count in np.arange(0, 20):
        print("------------------")
        print(f"{count=}")
        n_suggest = df_smallest['params_n_suggest'].item()
        f_s_suggest = df_smallest['params_f_s_suggest'].item()
        l_s_suggest = df_smallest['params_l_s_suggest'].item()
        l_1_suggest = df_smallest['params_l_1_suggest'].item()
        l_2__suggest = df_smallest['params_l_2__suggest'].item()
        transistor_1_name_suggest = df_smallest['params_transistor_1_name_suggest'].item()
        transistor_2_name_suggest = df_smallest['params_transistor_2_name_suggest'].item()

        # make sure to use parameters with minimum x % difference.
        difference = 0.05

        df = df.loc[
            ~((df["params_n_suggest"].ge(n_suggest * (1 - difference)) & df["params_n_suggest"].le(n_suggest * (1 + difference))) & \
              (df["params_f_s_suggest"].ge(f_s_suggest * (1 - difference)) & df["params_f_s_suggest"].le(f_s_suggest * (1 + difference))) & \
              (df["params_l_s_suggest"].ge(l_s_suggest * (1 - difference)) & df["params_l_s_suggest"].le(l_s_suggest * (1 + difference))) & \
              (df["params_l_1_suggest"].ge(l_1_suggest * (1 - difference)) & df["params_l_1_suggest"].le(l_1_suggest * (1 + difference))) & \
              (df["params_l_2__suggest"].ge(l_2__suggest * (1 - difference)) & df["params_l_2__suggest"].le(l_2__suggest * (1 + difference))) & \
              df["params_transistor_1_name_suggest"].isin([transistor_1_name_suggest]) & \
              df["params_transistor_2_name_suggest"].isin([transistor_2_name_suggest])
              )]

        df_smallest = df.nsmallest(n=1, columns=["values_1"])
        df_smallest_all = pd.concat([df_smallest_all, df_smallest], axis=0)

    smallest_dto_list = dct.Optimization.df_to_dab_dto_list(dab_config, df_smallest_all)

    dct.global_plot_settings_font_latex()
    figure = "zoom"  # "zoom"

    fig = plt.figure(figsize=(80/25.4, 80/25.4), dpi=350)

    plt.scatter(df_original["values_0"], df_original["values_1"], color=dct.colors()["blue"], label="Possible designs")
    plt.scatter(df_smallest_all["values_0"], df_smallest_all["values_1"], color=dct.colors()["red"], label="Non-similar designs")
    plt.xlabel(r"ZVS coverage / \%")
    plt.ylabel(r"$i_\mathrm{cost}$ / A²")

    if figure == "zoom":
        plt.ylim(91.37, 91.47)
    else:
        plt.ylim(50, 200)
    plt.xticks(ticks=[100], labels=["100"])
    plt.legend()
    plt.grid()
    plt.tight_layout()
    if figure == "zoom":
        plt.savefig("/home/nikolasf/Dokumente/12_Paper/14_2024_DMC/03_final_paper_git/figures/circuit_zvs_current_zoom.png")
    else:
        plt.savefig("/home/nikolasf/Dokumente/12_Paper/14_2024_DMC/03_final_paper_git/figures/circuit_zvs_current.png")
    plt.show()

    # folders = dct.Optimization.load_filepaths(dab_config.project_directory)
    # for dto in smallest_dto_list:
    #     print(f"{dto.name=}")
    #     dto_directory = os.path.join(folders.circuit, dab_config.circuit_study_name, "filtered_results")
    #     os.makedirs(dto_directory, exist_ok=True)
    #     dto = dct.HandleDabDto.add_gecko_simulation_results(dto, get_waveforms=True)
    #     dct.HandleDabDto.save(dto, dto.name, comment="", directory=dto_directory, timestamp=False)


elif action == 'custom':
    dab_config = dct.Optimization.load_config(dab_config.project_directory, dab_config.circuit_study_name)
    # dab_dto = dct.Optimization.load_dab_dto_from_study(dab_config, 99999)
    df = dct.Optimization.load_csv_to_df(os.path.join(dab_config.project_directory, "01_circuit", dab_config.circuit_study_name,
                                         f"{dab_config.circuit_study_name}.csv"))
    # df = df[df["number"] == 99999]
    df = df[df["number"] == 79030]
    print(df.head())

    [dab_dto] = dct.Optimization.df_to_dab_dto_list(dab_config, df)

    i_cost = dab_dto.calc_currents.i_hf_1_rms ** 2 + dab_dto.calc_currents.i_hf_2_rms ** 2

    print(f"{np.mean(i_cost)=}")

    i_cost_matrix = dab_dto.calc_currents.i_hf_1_rms ** 2 + dab_dto.calc_currents.i_hf_2_rms ** 2
    i_cost_new = np.mean(i_cost_matrix)
    print(f"{i_cost_new=}")
    i_cost_original = np.mean(i_cost_matrix[~np.isnan(i_cost_matrix)])
    print(f"{i_cost_original=}")

    dab_dto = dct.HandleDabDto.add_gecko_simulation_results(dab_dto, False)

    i_cost_matrix_gecko = dab_dto.gecko_results.i_HF1 ** 2 + dab_dto.gecko_results.i_HF2 ** 2
    i_cost_gecko = np.mean(i_cost_matrix_gecko)
    print(f"{i_cost_gecko=}")

    error_matrix_hf_1 = np.abs((dab_dto.calc_currents.i_hf_1_rms - dab_dto.gecko_results.i_HF1) / dab_dto.calc_currents.i_hf_1_rms)
    print(f"{np.mean(error_matrix_hf_1)=}")
    error_matrix_hf_2 = np.abs((dab_dto.calc_currents.i_hf_2_rms - dab_dto.gecko_results.i_HF2) / dab_dto.calc_currents.i_hf_2_rms)
    print(f"{np.mean(error_matrix_hf_2)=}")

    dct.HandleDabDto.save(dab_dto, "results", "", "~/Downloads", False)

    # loaded_dto = dct.HandleDabDto.load_from_file("~/Downloads/results.pkl")
    #
    # print(loaded_dto)
    #
    # loaded_dto = dct.HandleDabDto.add_gecko_simulation_results(loaded_dto)
