"""Stacked transformer Pareto optimization."""

# python libraries
import logging
import os.path
import pickle
import sys


# own libraries
import paretodab
import femmt as fmt

# 3rd party libraries
import numpy as np
import pandas as pd
import tqdm

# configure root logger
logging.basicConfig(format='%(levelname)s,%(asctime)s:%(message)s', encoding='utf-8')
logging.getLogger().setLevel(logging.ERROR)

insulations = fmt.StoInsulation(
    # insulation for top core window
    iso_window_top_core_top=1.3e-3,
    iso_window_top_core_bot=1.3e-3,
    iso_window_top_core_left=1.3e-3,
    iso_window_top_core_right=1.3e-3,
    # insulation for bottom core window
    iso_window_bot_core_top=1.3e-3,
    iso_window_bot_core_bot=1.3e-3,
    iso_window_bot_core_left=1.3e-3,
    iso_window_bot_core_right=1.3e-3,
    # winding-to-winding insulation
    iso_primary_to_primary=0.2e-3,
    iso_secondary_to_secondary=0.2e-3,
    iso_primary_to_secondary=0.2e-3,
)

material_data_sources = fmt.StackedTransformerMaterialDataSources(
    permeability_datasource=fmt.MaterialDataSource.Measurement,
    permeability_datatype=fmt.MeasurementDataType.ComplexPermeability,
    permeability_measurement_setup=fmt.MeasurementSetup.MagNet,
    permittivity_datasource=fmt.MaterialDataSource.ManufacturerDatasheet,
    permittivity_datatype=fmt.MeasurementDataType.ComplexPermittivity,
    permittivity_measurement_setup=fmt.MeasurementSetup.LEA_LK
)
def simulation(circuit_trial_numbers: list, process_number: int, target_number_trials: int, filter_factor: float = 1.0,
               re_simulate: bool = False, debug: bool = False):
    """
    Simulate.

    :param circuit_trial_numbers: List of circuit trial numbers to perform inductor optimization
    :type circuit_trial_numbers: list
    :param process_number: Process number (in case of parallel computing)
    :type process_number: int
    :param target_number_trials: Number of trials for the reluctance model optimization
    :type target_number_trials: int
    :param filter_factor: Pareto filter, tolerance band = Multiplication of minimum losses
    :type filter_factor: flot
    :param re_simulate: True to re-simulate all waveforms
    :type re_simulate: bool
    :param debug: True to debug, defaults to False
    :type debug: bool
    """
    filepaths = paretodab.Optimization.load_filepaths(os.path.abspath(os.path.join(os.curdir, project_name)))

    for circuit_trial_number in circuit_trial_numbers:
        circuit_filepath = os.path.join(filepaths.circuit, circuit_study_name, "filtered_results", f"{circuit_trial_number}.pkl")

        circuit_dto = paretodab.HandleDabDto.load_from_file(circuit_filepath)
        # get the peak current waveform
        sorted_max_angles, i_l_s_max_current_waveform, i_hf_2_max_current_waveform = paretodab.HandleDabDto.get_max_peak_waveform_transformer(
            circuit_dto, False)

        time = sorted_max_angles / 2 / np.pi / circuit_dto.input_config.fs

        transformer_target_params = paretodab.HandleDabDto.export_transformer_target_parameters_dto(dab_dto=circuit_dto)

        target_param_ls12 = float(transformer_target_params.l_s12_target)
        target_param_lh = float(transformer_target_params.l_h_target)
        target_param_n = float(transformer_target_params.n_target)
        time_current_1_vec = transformer_target_params.time_current_1_vec
        time_current_2_vec = transformer_target_params.time_current_2_vec
        temperature = 100

        sto_config = fmt.StoSingleInputConfig(

            stacked_transformer_study_name=sto_study_name,
            # target parameters
            l_s12_target=target_param_ls12,
            l_h_target=target_param_lh,
            n_target=target_param_n,

            # operating point: current waveforms and temperature
            time_current_1_vec=time_current_1_vec,
            time_current_2_vec=time_current_2_vec,
            temperature=temperature,

            # sweep parameters: geometry and materials
            n_p_top_min_max_list=[1, 30],
            n_p_bot_min_max_list=[10, 80],
            material_list=['3C95'],
            core_name_list=["PQ 40/40", "PQ 40/30", "PQ 35/35", "PQ 32/30", "PQ 32/20", "PQ 26/25", "PQ 26/20", "PQ 20/20", "PQ 20/16"],
            core_inner_diameter_min_max_list=[15e-3, 30e-3],
            window_w_min_max_list=[10e-3, 40e-3],
            window_h_bot_min_max_list=[10e-3, 50e-3],
            primary_litz_wire_list=['1.1x60x0.1'],
            secondary_litz_wire_list=['1.35x200x0.071', '1.1x60x0.1'],

            # maximum limitation for transformer total height and core volume
            max_transformer_total_height=60e-3,
            max_core_volume=50e-3 ** 2 * np.pi,

            # fix parameters: insulations
            insulations=insulations,

            # misc
            stacked_transformer_optimization_directory=os.path.join(filepaths.transformer, circuit_study_name, circuit_dto.name, sto_study_name),
            fft_filter_value_factor=0.01,
            mesh_accuracy=0.8,

            # data sources
            material_data_sources=material_data_sources
        )

        if target_number_trials != 0:
            if debug:
                # overwrite input number of trials with 100 for short simulation times
                target_number_trials = 100 if target_number_trials > 100 else target_number_trials
                fmt.optimization.StackedTransformerOptimization.ReluctanceModel.start_proceed_study(sto_config, number_trials=target_number_trials)
            else:
                fmt.optimization.StackedTransformerOptimization.ReluctanceModel.start_proceed_study(sto_config, target_number_trials=target_number_trials)

        # perform FEM simulations
        if filter_factor != 0:
            df = fmt.optimization.StackedTransformerOptimization.ReluctanceModel.study_to_df(sto_config)
            df_filtered = fmt.optimization.StackedTransformerOptimization.ReluctanceModel.filter_loss_list_df(df,
                                                                                                              factor_min_dc_losses=20, factor_max_dc_losses=30)
            if debug:
                # reduce dataset to the fist 5 entries
                df_filtered = df_filtered.iloc[:5]

            fmt.optimization.StackedTransformerOptimization.FemSimulation.fem_simulations_from_reluctance_df(df_filtered, sto_config,
                                                                                                             process_number=process_number)

        # plot options
        # fmt.optimization.StackedTransformerOptimization.ReluctanceModel.show_study_results(sto_config)
        # fmt.optimization.StackedTransformerOptimization.ReluctanceModel.df_plot_pareto_front(df, df_filtered, label_list=["all", "front"], interactive=False)

        if re_simulate:
            fem_results_folder_path = os.path.join(filepaths.transformer, circuit_study_name, circuit_dto.name, sto_study_name, "02_fem_simulation_results")
            df = fmt.optimization.StackedTransformerOptimization.ReluctanceModel.study_to_df(sto_config)
            df_filtered = fmt.optimization.StackedTransformerOptimization.ReluctanceModel.filter_loss_list_df(df, factor_min_dc_losses=20,
                                                                                                              factor_max_dc_losses=30)
            df_fem_reluctance = fmt.StackedTransformerOptimization.FemSimulation.fem_logs_to_df(df_filtered, fem_results_folder_path)

            config_filepath = os.path.join(filepaths.transformer, circuit_study_name, str(circuit_trial_number), sto_study_name, f"{sto_study_name}.pkl")
            config_on_disk = fmt.StackedTransformerOptimization.ReluctanceModel.load_config(config_filepath)

            # workaround for comma problem. Read a random csv file and set back the delimiter.
            pd.read_csv('~/Downloads/Pandas_trial.csv', header=0, index_col=0, delimiter=';')

            # sweep through all current waveforms
            i_l1_sorted = np.transpose(circuit_dto.calc_currents.i_l_1_sorted, (1, 2, 3, 0))
            angles_rad_sorted = np.transpose(circuit_dto.calc_currents.angles_rad_sorted, (1, 2, 3, 0))

            re_simulate_numbers = df_fem_reluctance["number"].to_numpy()

            for re_simulate_number in re_simulate_numbers:
                print(f"{re_simulate_number=}")
                df_geometry_re_simulation_number = df_fem_reluctance[df_fem_reluctance["number"] == re_simulate_number]

                result_array = np.full_like(circuit_dto.calc_modulation.phi, np.nan)

                new_circuit_dto_directory = os.path.join(sto_config.stacked_transformer_optimization_directory, "09_circuit_dtos_incl_transformer_losses")
                if not os.path.exists(new_circuit_dto_directory):
                    os.makedirs(new_circuit_dto_directory)

                if os.path.exists(os.path.join(new_circuit_dto_directory, f"{re_simulate_number}.pkl")):
                    print(f"Re-simulation of {circuit_dto.name} already exists. Skip.")
                else:
                    for vec_vvp in tqdm.tqdm(np.ndindex(circuit_dto.calc_modulation.phi.shape), total=len(circuit_dto.calc_modulation.phi.flatten())):
                        if debug:
                            print("----------------------")
                            print("Re-simulation of:")
                            print(f"   * Circuit study: {circuit_study_name}")
                            print(f"   * Circuit trial: {circuit_trial_number}")
                            print(f"   * Inductor study: {sto_study_name}")
                            print(f"   * Inductor re-simulation trial: {re_simulate_number}")
                        time = paretodab.functions_waveforms.full_angle_waveform_from_angles(
                            angles_rad_sorted[vec_vvp]) / 2 / np.pi / circuit_dto.input_config.fs
                        current = paretodab.functions_waveforms.full_current_waveform_from_currents(i_l1_sorted[vec_vvp])

                        current_waveform = np.array([time, current])

                        print(f"{current_waveform=}")
                        # workaround for comma problem. Read a random csv file and set back the delimiter.
                        pd.read_csv('~/Downloads/Pandas_trial.csv', header=0, index_col=0, delimiter=';')

                        volume, combined_losses, area_to_heat_sink = fmt.StackedTransformerOptimization.FemSimulation.full_simulation(
                            df_geometry_re_simulation_number, current_waveform, config_filepath, show_visual_outputs=False,
                            process_number=process_number)
                        result_array[vec_vvp] = combined_losses

                    results_dto = paretodab.StackedTransformerResults(
                        p_combined_losses=result_array,
                        volume=volume,
                        area_to_heat_sink=area_to_heat_sink,
                        circuit_trial_number=circuit_trial_number,
                        stacked_transformer_trial_number=re_simulate_number
                    )

                    pickle_file = os.path.join(new_circuit_dto_directory, f"{re_simulate_number}.pkl")
                    with open(pickle_file, 'wb') as output:
                        pickle.dump(results_dto, output, pickle.HIGHEST_PROTOCOL)

                        if debug:
                            # stop after one successful re-simulation run
                            break

        if debug:
            # stop after one circuit run
            break


if __name__ == '__main__':
    process_number = int(sys.argv[1]) if len(sys.argv) == 2 else 1
    total_processes = 1

    # project name, circuit study name and inductor study name
    project_name = "2024-10-04_dab_paper"
    circuit_study_name = "circuit_paper_trial_1"
    sto_study_name = "transformer_trial_1_test"

    # inductor optimization
    process_circuit_trial_numbers = []

    filepaths = paretodab.Optimization.load_filepaths(os.path.abspath(os.path.join(os.curdir, project_name)))
    circuit_filepath = os.path.join(filepaths.circuit, circuit_study_name, "filtered_results")
    objects = os.scandir(circuit_filepath)
    all_circuit_trial_numbers = [entity.name.replace(".pkl", "") for entity in objects]

    # check for empty list
    if not process_circuit_trial_numbers:
        # define circuit numbers per process
        process_circuit_trial_numbers = [all_circuit_trial_numbers[index] for index in range(0, len(all_circuit_trial_numbers))
                                         if (index + 1 - process_number) % total_processes == 0]

    simulation(process_circuit_trial_numbers, target_number_trials=200, filter_factor=0, re_simulate=True, debug=False, process_number=process_number)
