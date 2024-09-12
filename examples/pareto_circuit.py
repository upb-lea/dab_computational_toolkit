"""Example how to use the design_check for the DAB converter."""
# python libraries
import os

# 3rd party libraries

# own libraries
import dct

design_space = dct.CircuitParetoDesignSpace(
    f_s_min_max_list=[200e3, 200e3],
    l_s_min_max_list=[1e-6, 200e-6],
    l_1_min_max_list=[100e-6, 100e-3],
    l_2_min_max_list=[100e-6, 100e-3],
    n_min_max_list=[4, 7],
    transistor_1_list=['CREE_C3M0065100J'],
    transistor_2_list=['CREE_C3M0060065J']
)

output_range = dct.CircuitOutputRange(
    v_1_min_nom_max_list=[600, 700, 800],
    v_2_min_nom_max_list=[175, 235, 295],
    p_min_nom_max_list=[0, 2000, 2200],
    steps_per_direction=2,
)

dab_config = dct.CircuitParetoDabDesign(
    dab_study_name='circuit_trial_1',
    project_directory=os.path.abspath(os.path.join(os.curdir, "2024-09-12_project_dab_paper")),

    design_space=design_space,
    output_range=output_range
)

# action = 'run_new_study'
action = 'show_study_results'
# action = 'filter_study_results_and_run_gecko'

if action == 'run_new_study':
    dct.Optimization.start_proceed_study(dab_config, 50)

elif action == 'show_study_results':
    dab_config = dct.Optimization.load_config(dab_config.project_directory, dab_config.dab_study_name)
    dct.Optimization.show_study_results(dab_config)

elif action == 'filter_study_results_and_run_gecko':
    df = dct.Optimization.study_to_df(dab_config)
    df = df[df["values_0"] == 100]
    df = df.nsmallest(n=3, columns=["values_1"])
    smallest_dto_list = dct.Optimization.df_to_dab_dto_list(dab_config, df)

    folders = dct.Optimization.load_filepaths(dab_config.project_directory)

    for dto in smallest_dto_list:
        print(f"{dto.name=}")
        dto_directory = os.path.join(folders.circuit, dab_config.dab_study_name, "filtered_results")
        os.makedirs(dto_directory, exist_ok=True)
        dto = dct.HandleDabDto.add_gecko_simulation_results(dto, get_waveforms=True)
        dct.HandleDabDto.save(dto, dto.name, comment="", directory=dto_directory, timestamp=False)
