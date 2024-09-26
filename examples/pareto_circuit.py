"""Example how to use the design_check for the DAB converter."""
# python libraries
import os

# 3rd party libraries
import numpy as np

# own libraries
import paretodab

import logging
logging.basicConfig(format='%(levelname)s,%(asctime)s:%(message)s', encoding='utf-8')
logging.getLogger('pygeckocircuits2').setLevel(logging.DEBUG)

# design_space = paretodab.CircuitParetoDesignSpace(
#     f_s_min_max_list=[200e3, 200e3],
#     l_s_min_max_list=[1e-6, 200e-6],
#     l_1_min_max_list=[100e-6, 100e-3],
#     l_2__min_max_list=[30e-6, 1e-3],
#     n_min_max_list=[3, 7],
#     transistor_1_list=['CREE_C3M0065100J'],
#     transistor_2_list=['CREE_C3M0060065J']
# )

design_space = paretodab.CircuitParetoDesignSpace(
    f_s_min_max_list=[200e3, 200e3],
    l_s_min_max_list=[120e-6, 125e-6],
    l_1_min_max_list=[120e-7, 120e-4],
    l_2__min_max_list=[120e-7 / 4.2 ** 2, 120e-4 / 4.2 ** 2],
    n_min_max_list=[4.2, 4.2],
    transistor_1_name_list=['CREE_C3M0065100J'],
    transistor_2_name_list=['CREE_C3M0060065J'],
    c_par_1=6e-12,
    c_par_2=6e-12,
)

output_range = paretodab.CircuitOutputRange(
    v_1_min_nom_max_list=[690, 700, 710],
    v_2_min_nom_max_list=[175, 235, 295],
    p_min_nom_max_list=[0, 2000, 2200],
    steps_per_direction=1,
)

dab_config = paretodab.CircuitParetoDabDesign(
    circuit_study_name='circuit_trial_11_workflow_steps_1',
    project_directory=os.path.abspath(os.path.join(os.curdir, "2024-09-12_project_dab_paper")),

    design_space=design_space,
    output_range=output_range
)

# action = 'run_new_study'
# action = 'show_study_results'
action = 'filter_study_results_and_run_gecko'
# action = 'custom'

if action == 'run_new_study':
    paretodab.Optimization.start_proceed_study(dab_config, 5000)

elif action == 'show_study_results':
    dab_config = paretodab.Optimization.load_config(dab_config.project_directory, dab_config.circuit_study_name)
    paretodab.Optimization.show_study_results(dab_config)

elif action == 'filter_study_results_and_run_gecko':
    df = paretodab.Optimization.study_to_df(dab_config)
    df = df[df["values_0"] == 100]
    print(df.head())
    # df["l_2__"] = df["params_l_2_suggest"] * df["params_n_suggest"] * df["params_n_suggest"]
    # print(df["l_2__"])
    # df = df[df["params_l_2__suggest"] < 1e-3]

    df = df.nsmallest(n=10, columns=["values_1"])
    print(df.head())
    smallest_dto_list = paretodab.Optimization.df_to_dab_dto_list(dab_config, df)

    folders = paretodab.Optimization.load_filepaths(dab_config.project_directory)

    for dto in smallest_dto_list:
        print(f"{dto.name=}")
        dto_directory = os.path.join(folders.circuit, dab_config.circuit_study_name, "filtered_results")
        os.makedirs(dto_directory, exist_ok=True)
        dto = paretodab.HandleDabDto.add_gecko_simulation_results(dto, get_waveforms=True)
        paretodab.HandleDabDto.save(dto, dto.name, comment="", directory=dto_directory, timestamp=False)
elif action == 'custom':
    dab_config = paretodab.Optimization.load_config(dab_config.project_directory, dab_config.circuit_study_name)
    # dab_dto = paretodab.Optimization.load_dab_dto_from_study(dab_config, 99999)
    df = paretodab.Optimization.load_csv_to_df(os.path.join(dab_config.project_directory, "01_circuit", dab_config.circuit_study_name,
                                               f"{dab_config.circuit_study_name}.csv"))
    # df = df[df["number"] == 99999]
    df = df[df["number"] == 79030]
    print(df.head())

    [dab_dto] = paretodab.Optimization.df_to_dab_dto_list(dab_config, df)

    i_cost = dab_dto.calc_currents.i_hf_1_rms ** 2 + dab_dto.calc_currents.i_hf_2_rms ** 2

    print(f"{np.mean(i_cost)=}")

    i_cost_matrix = dab_dto.calc_currents.i_hf_1_rms ** 2 + dab_dto.calc_currents.i_hf_2_rms ** 2
    i_cost_new = np.mean(i_cost_matrix)
    print(f"{i_cost_new=}")
    i_cost_original = np.mean(i_cost_matrix[~np.isnan(i_cost_matrix)])
    print(f"{i_cost_original=}")

    dab_dto = paretodab.HandleDabDto.add_gecko_simulation_results(dab_dto, False)

    i_cost_matrix_gecko = dab_dto.gecko_results.i_HF1 ** 2 + dab_dto.gecko_results.i_HF2 ** 2
    i_cost_gecko = np.mean(i_cost_matrix_gecko)
    print(f"{i_cost_gecko=}")

    error_matrix_hf_1 = np.abs((dab_dto.calc_currents.i_hf_1_rms - dab_dto.gecko_results.i_HF1) / dab_dto.calc_currents.i_hf_1_rms)
    print(f"{np.mean(error_matrix_hf_1)=}")
    error_matrix_hf_2 = np.abs((dab_dto.calc_currents.i_hf_2_rms - dab_dto.gecko_results.i_HF2) / dab_dto.calc_currents.i_hf_2_rms)
    print(f"{np.mean(error_matrix_hf_2)=}")

    paretodab.HandleDabDto.save(dab_dto, "results", "", "~/Downloads", False)

    # loaded_dto = paretodab.HandleDabDto.load_from_file("~/Downloads/results.pkl")
    #
    # print(loaded_dto)
    #
    # loaded_dto = paretodab.HandleDabDto.add_gecko_simulation_results(loaded_dto)
