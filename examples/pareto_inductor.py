"""Example for inductor optimization."""

# python libraries
import os
import pickle
import logging
import sys

# 3rd party libraries
import pandas as pd
import numpy as np

# own libraries
import femmt as fmt
import paretodab

# configure root logger
logging.basicConfig(format='%(levelname)s,%(asctime)s:%(message)s', encoding='utf-8')
logging.getLogger().setLevel(logging.ERROR)

insulations = fmt.InductorInsulationDTO(
    core_bot=1e-3,
    core_top=1e-3,
    core_right=1e-3,
    core_left=1e-3,
    primary_to_primary=0.2e-3
)

material_data_sources = fmt.InductorMaterialDataSources(
    permeability_datasource=fmt.MaterialDataSource.Measurement,
    permeability_datatype=fmt.MeasurementDataType.ComplexPermeability,
    permeability_measurement_setup=fmt.MeasurementSetup.MagNet,
    permittivity_datasource=fmt.MaterialDataSource.ManufacturerDatasheet,
    permittivity_datatype=fmt.MeasurementDataType.ComplexPermittivity,
    permittivity_measurement_setup=fmt.MeasurementSetup.LEA_MTB_small_signal
)


def simulation(circuit_trial_numbers: list, number_trials: int, filter_factor: float = 1.0, re_simulate: bool = False, debug: bool = False):
    filepaths = paretodab.Optimization.load_filepaths(os.path.abspath(os.path.join(os.curdir, project_name)))

    for circuit_trial_number in circuit_trial_numbers:
        circuit_filepath = os.path.join(filepaths.circuit, circuit_study_name, "filtered_results", f"{circuit_trial_number}.pkl")

        circuit_dto = paretodab.HandleDabDto.load_from_file(circuit_filepath)
        # get the peak current waveform
        sorted_max_angles, i_l_1_max_current_waveform = paretodab.HandleDabDto.get_max_peak_waveform_inductor(circuit_dto, False)
        time = sorted_max_angles / 2 / np.pi / circuit_dto.input_config.fs

        io_config = fmt.InductorOptimizationDTO(
            inductor_study_name=inductor_study_name,
            core_name_list=["PQ 50/50", "PQ 50/40", "PQ 40/40", "PQ 40/30", "PQ 35/35", "PQ 32/30", "PQ 32/20", "PQ 26/25", "PQ 26/20", "PQ 20/20", "PQ 20/16"],
            material_name_list=["3C95"],
            core_inner_diameter_list=[],
            window_h_list=[],
            window_w_list=[],
            litz_wire_list=["1.5x105x0.1", "1.4x200x0.071", "1.1x60x0.1"],
            insulations=insulations,
            target_inductance=circuit_dto.input_config.Lc1,
            temperature=100,
            time_current_vec=np.array([time, i_l_1_max_current_waveform]),
            inductor_optimization_directory=os.path.join(filepaths.inductor, circuit_study_name, circuit_dto.name, inductor_study_name),
            material_data_sources=material_data_sources
        )

        if number_trials != 0:
            # overwrite input number of trials with 100 for short simulation times
            number_trials = 100 if debug and number_trials > 100 else number_trials
            fmt.optimization.InductorOptimization.ReluctanceModel.start_proceed_study(io_config, number_trials)

        # Plot options
        # fmt.optimization.InductorOptimization.ReluctanceModel.show_study_results(io_config)
        # fmt.optimization.InductorOptimization.ReluctanceModel.df_plot_pareto_front(df, df_filtered, label_list=["all", "front"], interactive=False)

        # perform FEM simulations
        if filter_factor != 0:
            df = fmt.optimization.InductorOptimization.ReluctanceModel.study_to_df(io_config)
            df_filtered = fmt.optimization.InductorOptimization.ReluctanceModel.filter_loss_list_df(df, factor_min_dc_losses=0.01)
            if debug:
                # reduce dataset to the fist 5 entries
                df_filtered = df_filtered.iloc[:5]

            fmt.InductorOptimization.FemSimulation.fem_simulations_from_reluctance_df(df_filtered, io_config, process_number=process_number)

        if re_simulate:
            fem_results_folder_path = os.path.join(filepaths.inductor, circuit_study_name, circuit_dto.name, inductor_study_name, "02_fem_simulation_results")
            df = fmt.optimization.InductorOptimization.ReluctanceModel.study_to_df(io_config)
            df_filtered = fmt.optimization.InductorOptimization.ReluctanceModel.filter_loss_list_df(df, factor_min_dc_losses=0.01)
            df_fem_reluctance = fmt.InductorOptimization.FemSimulation.fem_logs_to_df(df_filtered, fem_results_folder_path)
            # fmt.InductorOptimization.FemSimulation.fem_vs_reluctance_pareto(df_fem_reluctance)

            config_filepath = os.path.join(filepaths.inductor, circuit_study_name, str(circuit_trial_number), inductor_study_name, f"{inductor_study_name}.pkl")

            # workaround for comma problem. Read a random csv file and set back the delimiter.
            pd.read_csv('~/Downloads/Pandas_trial.csv', header=0, index_col=0, delimiter=';')

            # sweep through all current waveforms
            i_l1_sorted = np.transpose(circuit_dto.calc_currents.i_l_1_sorted, (1, 2, 3, 0))
            angles_rad_sorted = np.transpose(circuit_dto.calc_currents.angles_rad_sorted, (1, 2, 3, 0))

            print(df_fem_reluctance["number"])

            re_simulate_numbers = df_fem_reluctance["number"].to_numpy()

            for re_simulate_number in re_simulate_numbers:
                print(f"{re_simulate_number=}")
                df_geometry_re_simulation_number = df_fem_reluctance[df_fem_reluctance["number"] == float(re_simulate_number)]

                print(df_geometry_re_simulation_number.head())

                result_array = np.full_like(circuit_dto.calc_modulation.phi, np.nan)

                new_circuit_dto_directory = os.path.join(io_config.inductor_optimization_directory, "09_circuit_dtos_incl_inductor_losses")
                if not os.path.exists(new_circuit_dto_directory):
                    os.makedirs(new_circuit_dto_directory)

                if os.path.exists(os.path.join(new_circuit_dto_directory, f"{re_simulate_number}.pkl")):
                    print(f"Re-simulation of {circuit_dto.name} already exists. Skip.")
                else:
                    for vec_vvp in np.ndindex(circuit_dto.calc_modulation.phi.shape):
                        time, unique_indices = np.unique(
                            paretodab.functions_waveforms.full_angle_waveform_from_angles(angles_rad_sorted[vec_vvp]) / 2 / np.pi / circuit_dto.input_config.fs,
                            return_index=True)
                        current = paretodab.functions_waveforms.full_current_waveform_from_currents(i_l1_sorted[vec_vvp])[unique_indices]

                        current_waveform = np.array([time, current])
                        print(f"{current_waveform=}")
                        print("----------------------")
                        print("Re-simulation of:")
                        print(f"   * Circuit study: {circuit_study_name}")
                        print(f"   * Circuit trial: {circuit_trial_number}")
                        print(f"   * Inductor study: {inductor_study_name}")
                        print(f"   * Inductor re-simulation trial: {re_simulate_number}")

                        volume, combined_losses, area_to_heat_sink = fmt.InductorOptimization.FemSimulation.full_simulation(
                            df_geometry_re_simulation_number, current_waveform, config_filepath)
                        result_array[vec_vvp] = combined_losses

                    inductor_losses = paretodab.InductorResults(
                        p_combined_losses=result_array,
                        volume=volume,
                        area_to_heat_sink=area_to_heat_sink,
                        circuit_trial_number=circuit_trial_number,
                        inductor_trial_number=re_simulate_number,
                    )

                    pickle_file = os.path.join(new_circuit_dto_directory, f"{re_simulate_number}.pkl")
                    with open(pickle_file, 'wb') as output:
                        pickle.dump(inductor_losses, output, pickle.HIGHEST_PROTOCOL)

                    if debug:
                        # stop after one successful re-simulation run
                        break

        if debug:
            # stop after one circuit run
            break


if __name__ == '__main__':
    process_number = int(sys.argv[1]) if len(sys.argv) == 2 else 1
    total_processes = 3

    # project name, circuit study name and inductor study name
    project_name = "2024-10-04_dab_paper"
    circuit_study_name = "circuit_paper_trial_1"
    inductor_study_name = "inductor_trial_1"

    # inductor optimization
    process_circuit_trial_numbers = []
    re_simulate = False

    filepaths = paretodab.Optimization.load_filepaths(os.path.abspath(os.path.join(os.curdir, project_name)))
    circuit_filepath = os.path.join(filepaths.circuit, circuit_study_name, "filtered_results")
    objects = os.scandir(circuit_filepath)
    all_circuit_trial_numbers = [entity.name.replace(".pkl", "") for entity in objects]

    # check for empty list
    if not process_circuit_trial_numbers:
        # define circuit numbers per process
        process_circuit_trial_numbers = [all_circuit_trial_numbers[index] for index in range(0, len(all_circuit_trial_numbers)) if (index + 1 - process_number) % total_processes == 0]

    simulation(process_circuit_trial_numbers, number_trials=1, filter_factor=1, re_simulate=False, debug=False)
