"""Summary pareto optimizations."""
# python libraries
import os
import pickle

# 3rd party libraries
import pandas as pd
import numpy as np

# own libraries
import dct
from heat_sink_optimization import ThermalCalcSupport as thr_sup
import hct


class DctSummmaryProcessing:
    """Perform the summary calculation based on optimization results."""

    # Variable declaration

    # Areas and transistor cooling parameter
    copper_coin_area_1: float
    transistor_b1_cooling: float
    copper_coin_area_2: float
    transistor_b2_cooling: float

    # Thermal resistance
    r_th_ind_heat_sink_a: float
    r_th_xfmr_heat_sink_a: float

    # Heat sink parameter
    heat_sink = None

    @staticmethod
    def init_thermal_configuration(act_thermal_data: dct.TomlHeatSinkSummaryData) -> bool:
        """Initialize the thermal parameter of the connection points for the transistors, inductor and transformer.

        :param act_thermal_data : toml file with configuration data
        :type  act_thermal_data : dct.TomlHeatSinkSummaryData

        :return: True, if the thermal parameter of the connection points was successful initialized
        :rtype: bool
        """
        # Variable declaration
        # Return variable initialized to True
        successful_init = True
        transformer_cooling: list[float]
        # Thermal parameter for bridge transistor 1: List [tim_thickness, tim_conductivity]
        DctSummmaryProcessing.transistor_b1_cooling = dct.TransistorCooling(
            tim_thickness=act_thermal_data.transistor_b1_cooling[0],
            tim_conductivity=act_thermal_data.transistor_b1_cooling[1])

        # Thermal parameter for bridge transistor 2: List [tim_thickness, tim_conductivity]
        DctSummmaryProcessing.transistor_b2_cooling = dct.TransistorCooling(
            tim_thickness=act_thermal_data.transistor_b2_cooling[0],
            tim_conductivity=act_thermal_data.transistor_b2_cooling[1])

        # Thermal parameter for inductor: rth per area: List [tim_thickness, tim_conductivity]
        inductor_tim_thickness = act_thermal_data.inductor_cooling[0]
        inductor_tim_conductivity = act_thermal_data.inductor_cooling[1]

        # Check on zero
        if inductor_tim_conductivity > 0:
            # Calculate the thermal resistance area product
            DctSummmaryProcessing.r_th_ind_heat_sink_a = inductor_tim_thickness / inductor_tim_conductivity
        else:
            print(f"inductor cooling tim conductivity value must be greater zero, but is {inductor_tim_conductivity}!")
            successful_init = False

        # Thermal parameter for inductor: rth per area: List [tim_thickness, tim_conductivity]
        # ASA: Rename database class from InductiveElementCooling to MagneticElementCooling
        transformer_tim_thickness = act_thermal_data.transformer_cooling[0]
        transformer_tim_conductivity = act_thermal_data.transformer_cooling[1]

        transformer_cooling = dct.InductiveElementCooling(
            tim_thickness=transformer_tim_thickness,
            tim_conductivity=transformer_tim_conductivity
        )
        # Check on zero ( ASA: Maybe in general all configurtation files are to check for validity in advanced. In this case the check can be removed.)
        if transformer_tim_conductivity > 0:
            # Calculate the thermal resistance area product
            DctSummmaryProcessing.r_th_xfmr_heat_sink_a = transformer_tim_thickness / transformer_tim_conductivity
        else:
            print(f"transformer cooling tim conductivity value must be greater zero, but is {transformer_tim_conductivity}!")
            successful_init = False

        # Heat sink parameter:  List [t_ambient, t_hs_max]
        DctSummmaryProcessing.heat_sink = dct.HeatSinkBoundaryConditions(t_ambient=act_thermal_data.heat_sink[0],
                                                                         t_hs_max=act_thermal_data.heat_sink[1])
        # Return if initialisation was successful performed (True)
        return successful_init

    @staticmethod
    def _generate_magnetic_number_list(act_dir_name: str) -> tuple[bool, list[str]]:
        """Generate a list of the numbers from filenames.

        :param act_dir_name : Name of the directory containing the files
        :type  act_dir_name : str

        :return: tuple of bool and result list. True, if the directory exists and contains minimum one file
        :rtype: tuple
        """
        # Variable deklaration
        magnetic_result_numbers: list[str] = []
        is_magnetic_list_generated = False

        # Check if target folder 09_circuit_dtos_incl_inductor_losses is created
        if os.path.exists(act_dir_name):
            # Create list of filespath
            file_list = os.listdir(act_dir_name)
            # Filter basename without extension
            for file_name in file_list:
                # Create file path
                file_path = os.path.join(act_dir_name, file_name)
                # Check if it is a file
                if os.path.isfile(file_path):
                    device_number = os.path.splitext(os.path.basename(file_name))[0]
                    # Check file type
                    extension = os.path.splitext(os.path.basename(file_name))[1]
                    if extension == '.pkl':
                        magnetic_result_numbers.append(device_number)
                    else:
                        print(f"File {device_number}{extension} has no extension '.pkl'!")
                else:
                    print(f"File'{file_path}' does not exists!")
        else:
            print(f"Path {act_dir_name} does not exists!")

        if magnetic_result_numbers:
            is_magnetic_list_generated = True

        return is_magnetic_list_generated, magnetic_result_numbers

    @staticmethod
    def generate_result_database(act_ginfo: dct.GeneralInformation, act_inductor_study_names: list[str],
                                 act_stacked_transformer_study_names: list[str]) -> pd.DataFrame:
        """Generate a database df by summaries the calculation results.

        :param act_ginfo : General information about the study
        :type  act_ginfo : dct.GeneralInformation
        :param act_inductor_study_names : List of names with inductor studies which are to process
        :type  act_inductor_study_names : list[str]
        :param act_stacked_transformer_study_names : List of names with transformer studies which are to process
        :type  act_stacked_transformer_study_names : list[str]

        :return: dataframe with result information of the pareto front
        :rtype:  pd.DataFrame
        """
        # Variable declaration
        # Result dataframe
        df = pd.DataFrame()

        # iterate circuit numbers
        for circuit_number in act_ginfo.filtered_list_id:
            # Assemble pkl-filename
            circuit_filepath_number = os.path.join(act_ginfo.circuit_study_path, act_ginfo.circuit_study_name, "filtered_results", f"{circuit_number}.pkl")

            # Get circuit results
            circuit_dto = dct.HandleDabDto.load_from_file(circuit_filepath_number)

            # Calculate the thermal values

            # Begin: ASA: No influence by inductor or transformer ################################
            # get transistor results
            total_transistor_cond_loss_matrix \
                = 4 * (circuit_dto.calc_losses.p_m1_conduction + circuit_dto.calc_losses.p_m2_conduction)

            b1_transistor_cond_loss_matrix = circuit_dto.calc_losses.p_m1_conduction
            b2_transistor_cond_loss_matrix = circuit_dto.calc_losses.p_m2_conduction
            # End: ASA: No influence by inductor or transformer ################################
            # Begin: ASA: No influence by inductor or transformer ################################
            # get all the losses in a matrix
            r_th_copper_coin_1, copper_coin_area_1 = thr_sup.calculate_r_th_copper_coin(
                circuit_dto.input_config.transistor_dto_1.cooling_area)
            r_th_copper_coin_2, copper_coin_area_2 = thr_sup.calculate_r_th_copper_coin(
                circuit_dto.input_config.transistor_dto_2.cooling_area)

            circuit_r_th_tim_1 = thr_sup.calculate_r_th_tim(copper_coin_area_1, DctSummmaryProcessing.transistor_b1_cooling)
            circuit_r_th_tim_2 = thr_sup.calculate_r_th_tim(copper_coin_area_2, DctSummmaryProcessing.transistor_b2_cooling)

            circuit_r_th_1_jhs = circuit_dto.input_config.transistor_dto_1.r_th_jc + r_th_copper_coin_1 + circuit_r_th_tim_1
            circuit_r_th_2_jhs = circuit_dto.input_config.transistor_dto_2.r_th_jc + r_th_copper_coin_2 + circuit_r_th_tim_2

            circuit_heat_sink_max_1_matrix = (
                circuit_dto.input_config.transistor_dto_1.t_j_max_op - circuit_r_th_1_jhs * b1_transistor_cond_loss_matrix)
            circuit_heat_sink_max_2_matrix = (
                circuit_dto.input_config.transistor_dto_2.t_j_max_op - circuit_r_th_2_jhs * b2_transistor_cond_loss_matrix)
            # End: ASA: No influence by inductor or transformer ################################

            print(f"{circuit_number=}")

            # iterate inductor study
            for inductor_study_name in act_inductor_study_names:

                # Create update listfile for inductor and transformer
                # Initialise inductor and transformer list
                inductor_numbers = []
                stacked_transformer_numbers = []

                # Assemble directory name for inductor results:.../09_circuit_dtos_incl_inductor_losseslosses
                inductor_filepath_results = os.path.join(act_ginfo.inductor_study_path, circuit_number,
                                                         inductor_study_name,
                                                         "09_circuit_dtos_incl_inductor_losses")

                # Generate magnetic list
                is_inductor_list_generated, inductor_full_operating_range_list = (
                    DctSummmaryProcessing._generate_magnetic_number_list(inductor_filepath_results))
                if not is_inductor_list_generated:
                    print(f"Path {inductor_filepath_results} does not exists or does not contains any pkl-files!")
                    # Next circuit
                    continue

                # iterate inductor numbers
                for inductor_number in inductor_full_operating_range_list:
                    inductor_filepath_number = os.path.join(inductor_filepath_results, f"{inductor_number}.pkl")

                    # Get inductor results
                    with open(inductor_filepath_number, 'rb') as pickle_file_data:
                        inductor_dto = pickle.load(pickle_file_data)

                    if int(inductor_dto.circuit_trial_number) != int(circuit_number):
                        raise ValueError(f"{inductor_dto.circuit_trial_number=} != {circuit_number}")
                    if int(inductor_dto.inductor_trial_number) != int(inductor_number):
                        raise ValueError(f"{inductor_dto.inductor_trial_number=} != {inductor_number}")

                    inductance_loss_matrix = inductor_dto.p_combined_losses

                    # iterate transformer study
                    for stacked_transformer_study_name in act_stacked_transformer_study_names:

                        # Assemble directory name for transformer  results:.../09_circuit_dtos_incl_transformer_losseslosses
                        stacked_transformer_filepath_results = os.path.join(act_ginfo.transformer_study_path,
                                                                            circuit_number,
                                                                            stacked_transformer_study_name,
                                                                            "09_circuit_dtos_incl_transformer_losses")

                        # Check, if stacked transformer number list cannot be generated
                        is_transformer_list_generated, stacked_transformer_full_operating_range_list = (
                            DctSummmaryProcessing._generate_magnetic_number_list(stacked_transformer_filepath_results))
                        if not is_transformer_list_generated:
                            print(f"Path {stacked_transformer_filepath_results} does not exists or does not contains any pkl-files!")
                            # Next circuit
                            continue

                        # iterate transformer numbers
                        for stacked_transformer_number in stacked_transformer_full_operating_range_list:
                            stacked_transformer_filepath_number = os.path.join(stacked_transformer_filepath_results, f"{stacked_transformer_number}.pkl")

                            # get transformer results
                            with open(stacked_transformer_filepath_number, 'rb') as pickle_file_data:
                                transformer_dto = pickle.load(pickle_file_data)

                            if int(transformer_dto.circuit_trial_number) != int(circuit_number):
                                raise ValueError(f"{transformer_dto.circuit_trial_number=} != {circuit_number}")
                            if int(transformer_dto.stacked_transformer_trial_number) != int(stacked_transformer_number):
                                raise ValueError(f"{transformer_dto.stacked_transformer_trial_number=} != {stacked_transformer_number}")

                            transformer_loss_matrix = transformer_dto.p_combined_losses

                            total_loss_matrix = (inductor_dto.p_combined_losses + total_transistor_cond_loss_matrix + \
                                                 transformer_dto.p_combined_losses)

                            # maximum loss indices
                            max_loss_all_index = np.unravel_index(total_loss_matrix.argmax(), np.shape(total_loss_matrix))
                            # Calculate losses of circuit1 and 2
                            max_loss_circuit_1_index = np.unravel_index(b1_transistor_cond_loss_matrix.argmax(),
                                                                        np.shape(b1_transistor_cond_loss_matrix))
                            max_loss_circuit_2_index = np.unravel_index(b2_transistor_cond_loss_matrix.argmax(),
                                                                        np.shape(b2_transistor_cond_loss_matrix))

                            max_loss_inductor_index = np.unravel_index(inductance_loss_matrix.argmax(), np.shape(inductance_loss_matrix))
                            max_loss_transformer_index = np.unravel_index(transformer_loss_matrix.argmax(), np.shape(transformer_loss_matrix))

                            r_th_ind_heat_sink = DctSummmaryProcessing.r_th_ind_heat_sink_a / inductor_dto.area_to_heat_sink
                            temperature_inductor_heat_sink_max_matrix = 125 - r_th_ind_heat_sink * inductance_loss_matrix

                            r_th_xfmr_heat_sink = DctSummmaryProcessing.r_th_xfmr_heat_sink_a / transformer_dto.area_to_heat_sink
                            temperature_xfmr_heat_sink_max_matrix = 125 - r_th_xfmr_heat_sink * transformer_loss_matrix

                            # maximum heat sink temperatures (minimum of all the maximum temperatures of single components)
                            t_min_matrix = np.minimum(circuit_heat_sink_max_1_matrix, circuit_heat_sink_max_2_matrix)
                            t_min_matrix = np.minimum(t_min_matrix, temperature_inductor_heat_sink_max_matrix)
                            t_min_matrix = np.minimum(t_min_matrix, temperature_xfmr_heat_sink_max_matrix)
                            t_min_matrix = np.minimum(t_min_matrix, DctSummmaryProcessing.heat_sink.t_hs_max)

                            # maximum delta temperature over the heat sink
                            delta_t_max_heat_sink_matrix = t_min_matrix - DctSummmaryProcessing.heat_sink.t_ambient

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

        # Calculate the total area as sum of circuit,  inductor and transformer area df-comand is like vector sum v1[:]=v2[:]+v3[:])
        df["total_area"] = df["circuit_area"] + df["inductor_area"] + df["transformer_area"]
        df["total_mean_loss"] = df["circuit_mean_loss"] + df["inductor_mean_loss"] + df["transformer_mean_loss"]
        df["volume_wo_heat_sink"] = df["transformer_volume"] + df["inductor_volume"]
        # Save results to file (ASA : later to store only on demand)
        df.to_csv(f"{act_ginfo.heat_sink_study_path}/result_df.csv")

        # return the data base
        return df

    @staticmethod
    def select_heat_sink_configuration(act_ginfo: dct.GeneralInformation, act_df_for_hs: pd.DataFrame):
        """Select the heatsink configuration from calculated heatsink pareto front.

        :param act_ginfo : General information about the study name and study path
        :type  act_ginfo : dct.GeneralInformation:
        :param act_df_for_hs : dataframe with result information of the pareto front for heatsink selection
        :type  act_df_for_hs : pd.DataFrame
        """
        # Variable declaration

        # load heat sink
        hs_config_filepath = os.path.join(act_ginfo.heat_sink_study_path, act_ginfo.heat_sink_study_name, f"{act_ginfo.heat_sink_study_name}.pkl")
        hs_config = hct.Optimization.load_config(hs_config_filepath)
        # Debug ASA Missing true simulations for remaining function

        hs_config.heat_sink_optimization_directory = os.path.join(act_ginfo.heat_sink_study_path, act_ginfo.heat_sink_study_name)
        df_hs = hct.Optimization.study_to_df(hs_config)

        # generate full summary as panda database operation
        print(df_hs.loc[df_hs["values_1"] < 1]["values_0"].nsmallest(n=1).values[0])
        act_df_for_hs["heat_sink_volume"] = act_df_for_hs["r_th_heat_sink"].apply(
            lambda r_th_max: df_hs.loc[df_hs["values_1"] < r_th_max]["values_0"].nsmallest(n=1).values[0] \
            if np.any(df_hs.loc[df_hs["values_1"] < r_th_max]["values_0"].nsmallest(n=1).values) else None)

        act_df_for_hs["total_volume"] = act_df_for_hs["transformer_volume"] + act_df_for_hs["inductor_volume"] 
        + act_df_for_hs["heat_sink_volume"]

        # save full summary
        act_df_for_hs.to_csv(f"{act_ginfo.heat_sink_study_path}/df_summary.csv")
