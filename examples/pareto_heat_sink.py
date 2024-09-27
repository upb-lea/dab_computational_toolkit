"""Heat sink Pareto optimization."""
# python libraries
import os
import pickle

# 3rd party libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# own libraries
import paretodab

# specify input parameters
project_name = "2024-09-12_project_dab_paper"
circuit_study_name = "circuit_trial_11_workflow_steps_1"
inductor_study_name_list = ["inductor_trial_1_workflow"]
stacked_transformer_study_name_list = ["transformer_trial_1_workflow"]

global_df = pd.DataFrame()

# load project file paths
filepaths = paretodab.Optimization.load_filepaths(os.path.abspath(os.path.join(os.curdir, project_name)))

circuit_filepath_results = os.path.join(filepaths.circuit, circuit_study_name, "filtered_results")
circuit_objects = os.scandir(circuit_filepath_results)
circuit_numbers = [entry.name.split(os.extsep)[0] for entry in circuit_objects]

# iterate circuit numbers
for circuit_number in circuit_numbers:
    circuit_filepath_results = os.path.join(filepaths.circuit, circuit_study_name, "filtered_results")
    circuit_filepath_number = os.path.join(circuit_filepath_results, f"{circuit_number}.pkl")
    # iterate inductor study
    for inductor_study_name in inductor_study_name_list:
        inductor_filepath_results = os.path.join(filepaths.inductor, circuit_study_name, circuit_number,
                                                 inductor_study_name, "09_circuit_dtos_incl_inductor_losses")
        if os.path.exists(inductor_filepath_results):

            inductor_objects = os.scandir(inductor_filepath_results)
            inductor_numbers = [entry.name.split(os.extsep)[0] for entry in inductor_objects]
            print(inductor_numbers)

            # iterate inductor numbers
            for inductor_number in inductor_numbers:
                inductor_filepath_number = os.path.join(inductor_filepath_results, f"{inductor_number}.pkl")

                for stacked_transformer_study_name in stacked_transformer_study_name_list:
                    stacked_transformer_filepath_results = os.path.join(filepaths.transformer, circuit_study_name, circuit_number,
                                                                        stacked_transformer_study_name, "09_circuit_dtos_incl_transformer_losses")

                    stacked_transformer_objects = os.scandir(stacked_transformer_filepath_results)
                    stacked_transformer_numbers = [entry.name.split(os.extsep)[0] for entry in stacked_transformer_objects]

                    for stacked_transformer_number in stacked_transformer_numbers:
                        stacked_transformer_filepath_number = os.path.join(stacked_transformer_filepath_results, f"{stacked_transformer_number}.pkl")
                        # Get inductor results
                        with open(inductor_filepath_number, 'rb') as pickle_file_data:
                            inductance_dto_results = pickle.load(pickle_file_data)
                        inductance_loss_matrix = inductance_dto_results.p_combined_losses

                        # Get circuit results
                        circuit_dto_results = paretodab.HandleDabDto.load_from_file(circuit_filepath_number)

                        # get transistor results
                        total_transistor_cond_loss_matrix = 2 * (circuit_dto_results.gecko_results.S11_p_cond + circuit_dto_results.gecko_results.S12_p_cond + \
                                                                 circuit_dto_results.gecko_results.S23_p_cond + circuit_dto_results.gecko_results.S24_p_cond)

                        max_input_bridge_transistor_cond_loss_matrix = circuit_dto_results.gecko_results.S11_p_cond
                        max_input_bridge_transistor_cond_loss_matrix[
                            np.greater(circuit_dto_results.gecko_results.S12_p_cond, circuit_dto_results.gecko_results.S11_p_cond)] = (
                            circuit_dto_results.gecko_results.S12_p_cond)[
                            np.greater(circuit_dto_results.gecko_results.S12_p_cond, circuit_dto_results.gecko_results.S11_p_cond)]

                        max_output_bridge_transistor_cond_loss_matrix = circuit_dto_results.gecko_results.S23_p_cond
                        max_output_bridge_transistor_cond_loss_matrix[
                            np.greater(circuit_dto_results.gecko_results.S24_p_cond, circuit_dto_results.gecko_results.S23_p_cond)] = \
                            circuit_dto_results.gecko_results.S24_p_cond[
                            np.greater(circuit_dto_results.gecko_results.S24_p_cond, circuit_dto_results.gecko_results.S23_p_cond)]

                        # get transformer results
                        with open(stacked_transformer_filepath_number, 'rb') as pickle_file_data:
                            transformer_dto_results = pickle.load(pickle_file_data)
                        transformer_loss_matrix = transformer_dto_results.p_combined_losses

                        total_loss_matrix = (inductance_dto_results.p_combined_losses + total_transistor_cond_loss_matrix + \
                                             transformer_dto_results.p_combined_losses)

                        # get all the maximum losses
                        max_loss_all_index = np.unravel_index(total_loss_matrix.argmax(), np.shape(total_loss_matrix))
                        max_loss_circuit_ib_index = np.unravel_index(max_input_bridge_transistor_cond_loss_matrix.argmax(),
                                                                     np.shape(max_input_bridge_transistor_cond_loss_matrix))
                        max_loss_circuit_ob_index = np.unravel_index(max_output_bridge_transistor_cond_loss_matrix.argmax(),
                                                                     np.shape(max_output_bridge_transistor_cond_loss_matrix))
                        max_loss_inductor_index = np.unravel_index(inductance_loss_matrix.argmax(), np.shape(inductance_loss_matrix))
                        max_loss_transformer_index = np.unravel_index(transformer_loss_matrix.argmax(), np.shape(transformer_loss_matrix))

                        data = {
                            # circuit
                            "circuit_number": circuit_number,
                            "circuit_mean_loss": np.mean(total_transistor_cond_loss_matrix),
                            "circuit_max_all_loss": total_transistor_cond_loss_matrix[max_loss_all_index],
                            "circuit_max_circuit_ib_loss": total_transistor_cond_loss_matrix[max_loss_circuit_ib_index],
                            "circuit_max_circuit_ob_loss": total_transistor_cond_loss_matrix[max_loss_circuit_ob_index],
                            "circuit_max_inductor_loss": total_transistor_cond_loss_matrix[max_loss_inductor_index],
                            "circuit_max_transformer_loss": total_transistor_cond_loss_matrix[max_loss_transformer_index],
                            "circuit_ib_t_j_max": circuit_dto_results.input_config.transistor_dto_1.t_j,
                            "circuit_ob_t_j_max": circuit_dto_results.input_config.transistor_dto_2.t_j,
                            # inductor
                            "inductor_study_name": inductor_study_name,
                            "inductor_number": inductor_number,
                            "inductor_volume": inductance_dto_results.volume,
                            "inductor_mean_loss": np.mean(inductance_loss_matrix),
                            "inductor_max_all_loss": inductance_loss_matrix[max_loss_all_index],
                            "inductor_max_circuit_ib_loss": inductance_loss_matrix[max_loss_circuit_ib_index],
                            "inductor_max_circuit_ob_loss": inductance_loss_matrix[max_loss_circuit_ob_index],
                            "inductor_max_inductor_loss": inductance_loss_matrix[max_loss_inductor_index],
                            "inductor_max_transformer_loss": inductance_loss_matrix[max_loss_transformer_index],
                            "inductor_t_max": 0,
                            # transformer
                            "transformer_study_name": stacked_transformer_study_name,
                            "transformer_number": stacked_transformer_number,
                            "transformer_volume": transformer_dto_results.volume,
                            "transformer_mean_loss": np.mean(transformer_dto_results.p_combined_losses),
                            "transformer_max_all_loss": transformer_loss_matrix[max_loss_all_index],
                            "transformer_max_circuit_ib_loss": transformer_loss_matrix[max_loss_circuit_ib_index],
                            "transformer_max_circuit_ob_loss": transformer_loss_matrix[max_loss_circuit_ob_index],
                            "transformer_max_inductor_loss": transformer_loss_matrix[max_loss_inductor_index],
                            "transformer_max_transformer_loss": transformer_loss_matrix[max_loss_transformer_index],
                            "transformer_t_max": 0,
                        }
                        local_df = pd.DataFrame([data])
                        print(local_df.head())

                        global_df = pd.concat([global_df, local_df], axis=0)


global_df["total_volume"] = global_df["transformer_volume"] + global_df["inductor_volume"]
global_df["total_mean_loss"] = global_df["circuit_mean_loss"] + global_df["inductor_mean_loss"] + global_df["transformer_mean_loss"]

plt.scatter(global_df["total_volume"], global_df["total_mean_loss"])
plt.grid()
plt.xlabel("Volume in m³")
plt.ylabel("Mean total losses in W")
plt.show()

print(global_df.head())
