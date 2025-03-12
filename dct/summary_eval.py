"""Summary pareto optimization."""
# python libraries
import os
import pickle

# 3rd party libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# own libraries
import dct


class Dct_result_summary:

    # Variable declaration

    @staticmethod
    def generate_result_database(act_ginfo: dct.GeneralInformation):
        """Generate a database df by summaries the calculation results.

        The results of circuit-optimization, inductor-optimization and transformer-optimization are to summarize

        # specify input parameters
        project_name = "2024-10-04_dab_paper"
        circuit_study_name = "circuit_paper_trial_1"
        inductor_study_name_list = ["inductor_trial_1"]
        stacked_transformer_study_name_list = ["transformer_trial_1"]
        heat_sink_study_name = "heat_sink_trial_3_dimensions"

        # load project file paths
        filepaths = dct.Optimization.load_filepaths(os.path.abspath(os.path.join(os.curdir, project_name)))

        circuit_filepath_results = os.path.join(filepaths.circuit, circuit_study_name, "filtered_results")
        circuit_objects = os.scandir(circuit_filepath_results)
        circuit_numbers = [entry.name.split(os.extsep)[0] for entry in circuit_objects]

        """
        # Variable declaration


        # iterate circuit numbers
        for circuit_number in act_ginfo.filtered_list_id:
            # Assemble pkl-filename
            circuit_filepath_number = os.path.join(act_ginfo.circuit_study_path, "filtered_results",f"{circuit_number}.pkl")

            # Get circuit results
            circuit_dto = dct.HandleDabDto.load_from_file(circuit_filepath_number)

            a=act_ginfo.circuit_study_path

            print(f"{circuit_number=}")

            # iterate inductor study
            for inductor_study_name in inductor_study_name_list:
                inductor_filepath_results = os.path.join(filepaths.inductor, circuit_study_name, circuit_number,
                                                         inductor_study_name, "09_circuit_dtos_incl_inductor_losses")
                if os.path.exists(inductor_filepath_results):

                    inductor_objects = os.scandir(inductor_filepath_results)
                    inductor_numbers = [entry.name.split(os.extsep)[0] for entry in inductor_objects]

                    # iterate inductor numbers
                    for inductor_number in inductor_numbers:
                        inductor_filepath_number = os.path.join(inductor_filepath_results, f"{inductor_number}.pkl")

                        # Get inductor results
                        with open(inductor_filepath_number, 'rb') as pickle_file_data:
                            inductor_dto = pickle.load(pickle_file_data)

                        if int(inductor_dto.circuit_trial_number) != int(circuit_number):
                            raise ValueError(f"{inductor_dto.circuit_trial_number=} != {circuit_number}")
                        if int(inductor_dto.inductor_trial_number) != int(inductor_number):
                            raise ValueError(f"{inductor_dto.inductor_trial_number=} != {inductor_number}")

                        inductance_loss_matrix = inductor_dto.p_combined_losses

                        for stacked_transformer_study_name in stacked_transformer_study_name_list:
                            stacked_transformer_filepath_results = os.path.join(filepaths.transformer, circuit_study_name, circuit_number,
                                                                                stacked_transformer_study_name, "09_circuit_dtos_incl_transformer_losses")

                            if os.path.exists(stacked_transformer_filepath_results):

                                stacked_transformer_objects = os.scandir(stacked_transformer_filepath_results)
                                stacked_transformer_numbers = [entry.name.split(os.extsep)[0] for entry in stacked_transformer_objects]

                                for stacked_transformer_number in stacked_transformer_numbers:
                                    stacked_transformer_filepath_number = os.path.join(stacked_transformer_filepath_results, f"{stacked_transformer_number}.pkl")

                                    # get transformer results
                                    with open(stacked_transformer_filepath_number, 'rb') as pickle_file_data:
                                        transformer_dto = pickle.load(pickle_file_data)

                                    if int(transformer_dto.circuit_trial_number) != int(circuit_number):
                                        raise ValueError(f"{transformer_dto.circuit_trial_number=} != {circuit_number}")
                                    if int(transformer_dto.stacked_transformer_trial_number) != int(stacked_transformer_number):
                                        raise ValueError(f"{transformer_dto.stacked_transformer_trial_number=} != {stacked_transformer_number}")

                                    transformer_loss_matrix = transformer_dto.p_combined_losses

                                    # get transistor results
                                    total_transistor_cond_loss_matrix = 2 * (circuit_dto.gecko_results.S11_p_cond + circuit_dto.gecko_results.S12_p_cond + \
                                                                             circuit_dto.gecko_results.S23_p_cond + circuit_dto.gecko_results.S24_p_cond)

                                    max_b1_transistor_cond_loss_matrix = circuit_dto.gecko_results.S11_p_cond
                                    max_b1_transistor_cond_loss_matrix[
                                        np.greater(circuit_dto.gecko_results.S12_p_cond, circuit_dto.gecko_results.S11_p_cond)] = (
                                        circuit_dto.gecko_results.S12_p_cond)[
                                        np.greater(circuit_dto.gecko_results.S12_p_cond, circuit_dto.gecko_results.S11_p_cond)]

                                    max_b2_transistor_cond_loss_matrix = circuit_dto.gecko_results.S23_p_cond
                                    max_b2_transistor_cond_loss_matrix[
                                        np.greater(circuit_dto.gecko_results.S24_p_cond, circuit_dto.gecko_results.S23_p_cond)] = \
                                        circuit_dto.gecko_results.S24_p_cond[
                                        np.greater(circuit_dto.gecko_results.S24_p_cond, circuit_dto.gecko_results.S23_p_cond)]

                                    total_loss_matrix = (inductor_dto.p_combined_losses + total_transistor_cond_loss_matrix + \
                                                         transformer_dto.p_combined_losses)

                                    # maximum loss indices
                                    max_loss_all_index = np.unravel_index(total_loss_matrix.argmax(), np.shape(total_loss_matrix))
                                    max_loss_circuit_1_index = np.unravel_index(max_b1_transistor_cond_loss_matrix.argmax(),
                                                                                np.shape(max_b1_transistor_cond_loss_matrix))
                                    max_loss_circuit_2_index = np.unravel_index(max_b2_transistor_cond_loss_matrix.argmax(),
                                                                                np.shape(max_b2_transistor_cond_loss_matrix))
                                    max_loss_inductor_index = np.unravel_index(inductance_loss_matrix.argmax(), np.shape(inductance_loss_matrix))
                                    max_loss_transformer_index = np.unravel_index(transformer_loss_matrix.argmax(), np.shape(transformer_loss_matrix))

                                    # get all the losses in a matrix
                                    r_th_copper_coin_1, copper_coin_area_1 = dct.calculate_r_th_copper_coin(circuit_dto.input_config.transistor_dto_1.cooling_area)
                                    r_th_copper_coin_2, copper_coin_area_2 = dct.calculate_r_th_copper_coin(circuit_dto.input_config.transistor_dto_2.cooling_area)

                                    circuit_r_th_tim_1 = dct.calculate_r_th_tim(copper_coin_area_1, transistor_b1_cooling)
                                    circuit_r_th_tim_2 = dct.calculate_r_th_tim(copper_coin_area_2, transistor_b2_cooling)

                                    circuit_r_th_1_jhs = circuit_dto.input_config.transistor_dto_1.r_th_jc + r_th_copper_coin_1 + circuit_r_th_tim_1
                                    circuit_r_th_2_jhs = circuit_dto.input_config.transistor_dto_2.r_th_jc + r_th_copper_coin_2 + circuit_r_th_tim_2

                                    circuit_heat_sink_max_1_matrix = (
                                        circuit_dto.input_config.transistor_dto_1.t_j_max_op - circuit_r_th_1_jhs * max_b1_transistor_cond_loss_matrix)
                                    circuit_heat_sink_max_2_matrix = (
                                        circuit_dto.input_config.transistor_dto_2.t_j_max_op - circuit_r_th_2_jhs * max_b2_transistor_cond_loss_matrix)

                                    r_th_ind_heat_sink = 1 / inductor_cooling.tim_conductivity * inductor_cooling.tim_thickness / inductor_dto.area_to_heat_sink
                                    temperature_inductor_heat_sink_max_matrix = 125 - r_th_ind_heat_sink * inductance_loss_matrix

                                    r_th_xfmr_heat_sink = (1 / transformer_cooling.tim_conductivity * \
                                                           transformer_cooling.tim_thickness / transformer_dto.area_to_heat_sink)
                                    temperature_xfmr_heat_sink_max_matrix = 125 - r_th_xfmr_heat_sink * transformer_loss_matrix

                                    # maximum heat sink temperatures (minimum of all the maximum temperatures of single components)
                                    t_min_matrix = np.minimum(circuit_heat_sink_max_1_matrix, circuit_heat_sink_max_2_matrix)
                                    t_min_matrix = np.minimum(t_min_matrix, temperature_inductor_heat_sink_max_matrix)
                                    t_min_matrix = np.minimum(t_min_matrix, temperature_xfmr_heat_sink_max_matrix)
                                    t_min_matrix = np.minimum(t_min_matrix, heat_sink.t_hs_max)

                                    # maximum delta temperature over the heat sink
                                    delta_t_max_heat_sink_matrix = t_min_matrix - heat_sink.t_ambient

                                    r_th_heat_sink_target_matrix = delta_t_max_heat_sink_matrix / total_loss_matrix

                                    r_th_target = r_th_heat_sink_target_matrix.min()

                                    data = {
                                        # circuit
                                        "circuit_number": circuit_number,
                                        "circuit_mean_loss": np.mean(total_transistor_cond_loss_matrix),
                                        "circuit_max_all_loss": total_transistor_cond_loss_matrix[max_loss_all_index],
                                        "circuit_max_circuit_ib_loss": total_transistor_cond_loss_matrix[max_loss_circuit_1_index],
                                        "circuit_max_circuit_ob_loss": total_transistor_cond_loss_matrix[max_loss_circuit_2_index],
                                        "circuit_max_inductor_loss": total_transistor_cond_loss_matrix[max_loss_inductor_index],
                                        "circuit_max_transformer_loss": total_transistor_cond_loss_matrix[max_loss_transformer_index],
                                        "circuit_t_j_max_1": circuit_dto.input_config.transistor_dto_1.t_j_max_op,
                                        "circuit_t_j_max_2": circuit_dto.input_config.transistor_dto_2.t_j_max_op,
                                        "circuit_r_th_ib_jhs_1": circuit_r_th_1_jhs,
                                        "circuit_r_th_ib_jhs_2": circuit_r_th_2_jhs,
                                        "circuit_heat_sink_temperature_max_1": circuit_heat_sink_max_1_matrix[max_loss_circuit_1_index],
                                        "circuit_heat_sink_temperature_max_2": circuit_heat_sink_max_2_matrix[max_loss_circuit_2_index],
                                        "circuit_area": 4 * (copper_coin_area_1 + copper_coin_area_2),
                                        # inductor
                                        "inductor_study_name": inductor_study_name,
                                        "inductor_number": inductor_number,
                                        "inductor_volume": inductor_dto.volume,
                                        "inductor_mean_loss": np.mean(inductance_loss_matrix),
                                        "inductor_max_all_loss": inductance_loss_matrix[max_loss_all_index],
                                        "inductor_max_circuit_ib_loss": inductance_loss_matrix[max_loss_circuit_1_index],
                                        "inductor_max_circuit_ob_loss": inductance_loss_matrix[max_loss_circuit_2_index],
                                        "inductor_max_inductor_loss": inductance_loss_matrix[max_loss_inductor_index],
                                        "inductor_max_transformer_loss": inductance_loss_matrix[max_loss_transformer_index],
                                        "inductor_t_max": 0,
                                        "inductor_heat_sink_temperature_max": temperature_inductor_heat_sink_max_matrix[max_loss_inductor_index],
                                        "inductor_area": inductor_dto.area_to_heat_sink,
                                        # transformer
                                        "transformer_study_name": stacked_transformer_study_name,
                                        "transformer_number": stacked_transformer_number,
                                        "transformer_volume": transformer_dto.volume,
                                        "transformer_mean_loss": np.mean(transformer_dto.p_combined_losses),
                                        "transformer_max_all_loss": transformer_loss_matrix[max_loss_all_index],
                                        "transformer_max_circuit_ib_loss": transformer_loss_matrix[max_loss_circuit_1_index],
                                        "transformer_max_circuit_ob_loss": transformer_loss_matrix[max_loss_circuit_2_index],
                                        "transformer_max_inductor_loss": transformer_loss_matrix[max_loss_inductor_index],
                                        "transformer_max_transformer_loss": transformer_loss_matrix[max_loss_transformer_index],
                                        "transformer_t_max": 0,
                                        "transformer_heat_sink_temperature_max": temperature_xfmr_heat_sink_max_matrix[max_loss_transformer_index],
                                        "transformer_area": transformer_dto.area_to_heat_sink,

                                        # summary
                                        "total_losses": total_loss_matrix[max_loss_all_index],

                                        # heat sink
                                        "r_th_heat_sink": r_th_target
                                    }
                                    local_df = pd.DataFrame([data])

                                    df = pd.concat([df, local_df], axis=0)

        # Calculate tthe total area as sum of circuit,  inductor and transformer area df-comand is like vector sum v1[:]=v2[:]+v3[:])
        df["total_area"] = df["circuit_area"] + df["inductor_area"] + df["transformer_area"]
        df["total_mean_loss"] = df["circuit_mean_loss"] + df["inductor_mean_loss"] + df["transformer_mean_loss"]
        df["volume_wo_heat_sink"] = df["transformer_volume"] + df["inductor_volume"]
        # Save results to file (ASA : later to store only on demand)
        df.to_csv(f"{filepaths.heat_sink}/result_df.csv")

    @staticmethod
    def select_heatsink_configuration():
        """Select the heatsink configuration from calculated heatsink pareto front.

        Based on the summary results the suitable heatsink configuration is to select from paretofront.
        """
        # Variable declaration

        # load heat sink
        hs_config_filepath = os.path.join(filepaths.heat_sink, f"{heat_sink_study_name}.pkl")
        hs_config = hct.Optimization.load_config(hs_config_filepath)
        df_hs = hct.Optimization.study_to_df(hs_config)

        # load summarized results from database (ASA : xTodo: Take it directly form memory)
        df_wo_hs = pd.read_csv(f"{filepaths.heat_sink}/result_df.csv")

        # generate full summary as panda database operation
        print(df_hs.loc[df_hs["values_1"] < 1]["values_0"].nsmallest(n=1).values[0])
        df_wo_hs["heat_sink_volume"] = df_wo_hs["r_th_heat_sink"].apply(
            lambda r_th_max: df_hs.loc[df_hs["values_1"] < r_th_max]["values_0"].nsmallest(n=1).values[0] \
            if df_hs.loc[df_hs["values_1"] < r_th_max]["values_0"].nsmallest(n=1).values else None)

        df_wo_hs["total_volume"] = df_wo_hs["transformer_volume"] + df_wo_hs["inductor_volume"] + df_wo_hs["heat_sink_volume"]

        # save full summary
        df_wo_hs.to_csv(f"{filepaths.heat_sink}/df_summary.csv")
