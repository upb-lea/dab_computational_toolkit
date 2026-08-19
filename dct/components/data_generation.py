"""Data generation for lab prototypes."""

# Python libraries
import os
import logging
import subprocess

# 3rd party libraries
import pandas as pd
import femmt as fmt

# own libraries
import dct.toml_checker as tc
from dct import CapacitorConfiguration, InductorConfiguration, TransformerConfiguration, StudyData, CircuitOptimizationBase
from dct.constant_path import DF_SUMMARY_FINAL_FILTERED
from dct.constants import FACTOR_M_TO_MM

logger = logging.getLogger(__name__)

class DataGeneration:
    """Generate manufacturing data."""

    @staticmethod
    def _run_freecad(freecad_script_file, output_file, variables=None):
        """
        Run a FreeCAD Python script from the command line to export a STEP file.

        :param freecad_script_file: Path to the FreeCAD Python script (.py)
        :type freecad_script_file : str
        :param output_file: output STEP file path
        :type output_file: str
        :param variables: Optional parameters passed to the FreeCAD script as environment variables. Keys are converted to uppercase.
        :type variables: dict | None

        Example:
        {
            "core_inner_diameter_mm": 16.0,
            "l_air_gap_mm": 0.8
        }

        becomes:
        CORE_INNER_DIAMETER_MM=16.0
        L_AIR_GAP_MM=0.8
        """
        # Check whether the FreeCAD script exists.
        if not os.path.isfile(freecad_script_file):
            logger.error("Error: %s not found.", freecad_script_file)
            return False

        # Normalize paths.
        freecad_script_file = os.path.abspath(freecad_script_file)
        output_file = os.path.abspath(output_file)

        # Create the output folder when required.
        output_directory = os.path.dirname(output_file)
        if output_directory:
            os.makedirs(output_directory, exist_ok=True)

        # Copy the current environment to retain PATH, FreeCAD libraries, etc.
        environment = os.environ.copy()

        # This environment variable is read by the FreeCAD script.
        environment["OUTPUT_STEP_FILE"] = output_file

        # Pass optional script parameters through environment variables.
        if variables:
            for key, value in variables.items():
                environment[key.upper()] = str(value)

        # Adjust this if FreeCADCmd is not available in the system PATH.
        cmd = [
            "FreeCADCmd",
            freecad_script_file
        ]

        logger.info("Running: %s", " ".join(cmd))

        if variables:
            logger.info(
                "FreeCAD parameters: %s",
                ", ".join(
                    f"{key.upper()}={value}"
                    for key, value in variables.items()
                )
            )

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                env=environment
            )

            if result.stdout:
                logger.debug("FreeCAD output:\n%s", result.stdout)

            if result.stderr:
                # FreeCAD can write non-fatal messages to stderr.
                logger.warning("FreeCAD messages:\n%s", result.stderr)

            if not os.path.isfile(output_file):
                logger.error(
                    "FreeCAD completed without error, but no STEP file was found: %s",
                    output_file
                )
                return False

            logger.info("Success! STEP file saved to: %s", output_file)
            return True

        except FileNotFoundError:
            logger.error(
                "Error: 'FreeCADCmd' command not found. "
                "Install FreeCAD or add FreeCADCmd to your PATH."
            )
            return False

        except subprocess.CalledProcessError as error:
            logger.error(
                "FreeCAD exited with return code %s.",
                error.returncode
            )

            if error.stdout:
                logger.error("FreeCAD stdout:\n%s", error.stdout)

            if error.stderr:
                logger.error("FreeCAD stderr:\n%s", error.stderr)

            return False

        except Exception:
            logger.exception("Unexpected error while running FreeCAD.")
            return False

    @staticmethod
    def _read_summary_parameters(combination_id: int, df: pd.DataFrame) -> tuple[int, list[int], list[int], list[int], int]:
        """
        Read component IDs from the summary file.

        :param combination_id: combination ID
        :type combination_id: int
        :param df: summary dataframe
        :type df: pd.DataFrame
        """
        capacitor_id_list = []
        inductor_id_list = []
        transformer_id_list = []

        circuit_id = df.loc[df['combination_id'] == combination_id, 'circuit_id'].values[0]

        for count in [0, 1, 2, 3, 4, 5]:
            try:
                capacitor_id_list.append(df.loc[df['combination_id'] == combination_id, f'capacitor_id_{count}'].values[0])
            except KeyError:
                pass
            try:
                inductor_id_list.append(df.loc[df['combination_id'] == combination_id, f'inductor_id_{count}'].values[0])
            except KeyError:
                pass
            try:
                transformer_id_list.append(df.loc[df['combination_id'] == combination_id, f'transformer_id_{count}'].values[0])
            except KeyError:
                pass
        heat_sink_id = df.loc[df['combination_id'] == combination_id, 'heat_sink_id'].values[0]

        return circuit_id, capacitor_id_list, inductor_id_list, transformer_id_list, heat_sink_id

    @staticmethod
    def _generate_circuit_data(circuit_id: int, df_circuit: pd.DataFrame, output_filepath: str) -> None:
        """
        Generate circuit manufacturing data.

        :param circuit_id: circuit ID
        :type circuit_id: int
        :param df_circuit: circuit dataframe
        :type df_circuit: pd.DataFrame
        :param output_filepath: output filepath
        :type output_filepath: str
        """
        frequency = df_circuit.loc[df_circuit['number'] == circuit_id, 'params_f_s_suggest'].values[0]
        l_1 = df_circuit.loc[df_circuit['number'] == circuit_id, 'params_l_1_suggest'].values[0]
        l_2_ = df_circuit.loc[df_circuit['number'] == circuit_id, 'params_l_2__suggest'].values[0]
        l_s = df_circuit.loc[df_circuit['number'] == circuit_id, 'params_l_s_suggest'].values[0]
        n = df_circuit.loc[df_circuit['number'] == circuit_id, 'params_n_suggest'].values[0]
        transistor_1 = df_circuit.loc[df_circuit['number'] == circuit_id, 'params_transistor_1_name_suggest'].values[0]
        transistor_2 = df_circuit.loc[df_circuit['number'] == circuit_id, 'params_transistor_2_name_suggest'].values[0]

        circuit_data = (f"{frequency=}\n"
                        f"{l_1=}\n"
                        f"{l_2_=}\n"
                        f"{l_s=}\n"
                        f"{n=}\n"
                        f"{transistor_1=}\n"
                        f"{transistor_2=}\n"
                        )
        with open(f"{output_filepath}/circuit_data.txt", "w", encoding="utf-8") as f:
            f.write(circuit_data)

    @staticmethod
    def _generate_capacitor_data(capacitor_id: int, df_capacitor: pd.DataFrame, output_filepath: str, capacitor_number: int) -> None:
        """
        Generate inductor manufacturing data.

        :param capacitor_id: inductor ID
        :type capacitor_id: int
        :param df_capacitor: inductor dataframe
        :type df_capacitor: pd.DataFrame
        :param output_filepath: output filepath
        :type output_filepath: str
        :param capacitor_number: capacitor number in circuit, e.g. 0 or 1
        :type capacitor_number: int
        """
        ordering_code = df_capacitor.loc[df_capacitor['ordering code'] == capacitor_id, 'ordering code'].values[0]
        in_series_needed = df_capacitor.loc[df_capacitor['ordering code'] == capacitor_id, 'in_series_needed'].values[0]
        in_parallel_needed = df_capacitor.loc[df_capacitor['ordering code'] == capacitor_id, 'in_parallel_needed'].values[0]

        capacitor_data = (f"{ordering_code=}\n"
                          f"{in_series_needed=}\n"
                          f"{in_parallel_needed=}\n"
                          )
        with open(f"{output_filepath}/capacitor_{capacitor_number}_data.txt", "w", encoding="utf-8") as f:
            f.write(capacitor_data)

    @staticmethod
    def _generate_inductor_data(inductor_id: int, df_inductor: pd.DataFrame, output_filepath: str, inductor_number: int) -> None:
        """
        Generate inductor manufacturing data.

        :param inductor_id: inductor ID
        :type inductor_id: int
        :param df_inductor: inductor dataframe
        :type df_inductor: pd.DataFrame
        :param output_filepath: output filepath
        :type output_filepath: str
        :param inductor_number: number of the inductor in circuit
        :type inductor_number: int
        """
        params_core_name = df_inductor.loc[df_inductor['number'] == inductor_id, 'params_core_name'].values[0]
        params_litz_wire_name = df_inductor.loc[df_inductor['number'] == inductor_id, 'params_litz_wire_name'].values[0]
        params_material_name = df_inductor.loc[df_inductor['number'] == inductor_id, 'params_material_name'].values[0]
        params_turns = df_inductor.loc[df_inductor['number'] == inductor_id, 'params_turns'].values[0]
        params_window_h = df_inductor.loc[df_inductor['number'] == inductor_id, 'params_window_h'].values[0]
        user_attrs_core_inner_diameter = df_inductor.loc[df_inductor['number'] == inductor_id, 'user_attrs_core_inner_diameter'].values[0]
        user_attrs_dynamic_mu_r_abs = df_inductor.loc[df_inductor['number'] == inductor_id, 'user_attrs_dynamic_mu_r_abs'].values[0]
        user_attrs_flux_density_peak = df_inductor.loc[df_inductor['number'] == inductor_id, 'user_attrs_flux_density_peak'].values[0]
        user_attrs_l_air_gap = df_inductor.loc[df_inductor['number'] == inductor_id, 'user_attrs_l_air_gap'].values[0]
        user_attrs_window_w = df_inductor.loc[df_inductor['number'] == inductor_id, 'user_attrs_window_w'].values[0]

        inductor_data = (f"{params_core_name=}\n"
                         f"{params_litz_wire_name=}\n"
                         f"{params_material_name=}\n"
                         f"{params_turns=}\n"
                         f"{params_window_h=}\n"
                         f"{user_attrs_core_inner_diameter=}\n"
                         f"{user_attrs_dynamic_mu_r_abs=}\n"
                         f"{user_attrs_flux_density_peak=}\n"
                         f"{user_attrs_l_air_gap=}\n"
                         f"{user_attrs_window_w=}\n"
                         )
        with open(f"{output_filepath}/inductor_{inductor_number}_data.txt", "w", encoding="utf-8") as f:
            f.write(inductor_data)

        # PQ core step file generation
        core = fmt.core_database()[params_core_name]

        core_height_difference = core["window_h"] - params_window_h

        pq_core_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "pq_core_half.py")

        success = DataGeneration._run_freecad(
            freecad_script_file=pq_core_filepath,
            output_file=f"{output_filepath}/inductor_{inductor_number}_core.step",
            variables={
                "core_h_mm": (core["core_h"] - core_height_difference) * FACTOR_M_TO_MM,
                "core_inner_diameter_mm": user_attrs_core_inner_diameter * FACTOR_M_TO_MM,
                "window_h_mm": params_window_h * FACTOR_M_TO_MM,
                "window_w_mm": user_attrs_window_w * FACTOR_M_TO_MM,
                "core_dimension_x_mm": core["core_dimension_x"] * FACTOR_M_TO_MM,
                "core_dimension_y_mm": core["core_dimension_y"] * FACTOR_M_TO_MM,
                "l_air_gap_mm": user_attrs_l_air_gap * FACTOR_M_TO_MM,
            }
        )

        if not success:
            raise RuntimeError("STEP export failed.")

        success = DataGeneration._run_freecad(
            freecad_script_file=pq_core_filepath,
            output_file=f"{output_filepath}/inductor_{inductor_number}_core_original.step",
            variables={
                "core_h_mm": core["core_h"] * FACTOR_M_TO_MM,
                "core_inner_diameter_mm": core["core_inner_diameter"] * FACTOR_M_TO_MM,
                "window_h_mm": core["window_h"] * FACTOR_M_TO_MM,
                "window_w_mm": core["window_w"] * FACTOR_M_TO_MM,
                "core_dimension_x_mm": core["core_dimension_x"] * FACTOR_M_TO_MM,
                "core_dimension_y_mm": core["core_dimension_y"] * FACTOR_M_TO_MM,
                "l_air_gap_mm": 0,
            }
        )

        if not success:
            raise RuntimeError("STEP export failed.")

        bobbin_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "pq_bobbin.py")

        # generate inductor bobbin
        success = DataGeneration._run_freecad(
            freecad_script_file=bobbin_filepath,
            output_file=f"{output_filepath}/inductor_{inductor_number}_bobbin.step",

            variables={
                "window_h_mm": params_window_h * FACTOR_M_TO_MM,
                "window_w_mm": user_attrs_window_w * FACTOR_M_TO_MM,
                "core_inner_diameter_mm": user_attrs_core_inner_diameter * FACTOR_M_TO_MM,

                # Bobbin dimensions
                "flange_thickness_mm": 0.002 * FACTOR_M_TO_MM,

                "clearance": 0.3,
                "inner_edge_radius": 0.6,
                "outer_edge_radius": 0.6,
                "enable_wire_slots": True,
                "wire_slots_position": "both",
                "wire_slot_width": 4.0
            }
        )

        if not success:
            raise RuntimeError("STEP export failed.")

    @staticmethod
    def _generate_transformer_data(transformer_id: int, df_transformer: pd.DataFrame, output_filepath: str, transformer_number: int) -> None:
        """
        Generate transformer manufacturing data.

        :param transformer_id: transformer ID
        :type transformer_id: int
        :param df_transformer: transformer dataframe
        :type df_transformer: pd.DataFrame
        :param output_filepath: output filepath
        :type output_filepath: str
        :param transformer_number: number of the transformer in circuit
        :type transformer_number: int
        """
        params_core_name = df_transformer.loc[df_transformer['number'] == transformer_id, 'params_core_name'].values[0]
        params_material_name = df_transformer.loc[df_transformer['number'] == transformer_id, 'params_material_name'].values[0]
        params_n_p_bot = df_transformer.loc[df_transformer['number'] == transformer_id, 'params_n_p_bot'].values[0]
        params_n_p_top = df_transformer.loc[df_transformer['number'] == transformer_id, 'params_n_p_top'].values[0]
        params_n_s_bot = df_transformer.loc[df_transformer['number'] == transformer_id, 'params_n_s_bot'].values[0]
        params_primary_litz_name = df_transformer.loc[df_transformer['number'] == transformer_id, 'params_primary_litz_name'].values[0]
        params_secondary_litz_name = df_transformer.loc[df_transformer['number'] == transformer_id, 'params_secondary_litz_name'].values[0]
        params_window_h_bot = df_transformer.loc[df_transformer['number'] == transformer_id, 'params_window_h_bot'].values[0]
        user_attrs_core_inner_diameter = df_transformer.loc[df_transformer['number'] == transformer_id, 'user_attrs_core_inner_diameter'].values[0]
        user_attrs_l_bot_air_gap = df_transformer.loc[df_transformer['number'] == transformer_id, 'user_attrs_l_bot_air_gap'].values[0]
        user_attrs_l_top_air_gap = df_transformer.loc[df_transformer['number'] == transformer_id, 'user_attrs_l_top_air_gap'].values[0]
        user_attrs_window_h_bot = df_transformer.loc[df_transformer['number'] == transformer_id, 'user_attrs_window_h_bot'].values[0]
        user_attrs_window_h_top = df_transformer.loc[df_transformer['number'] == transformer_id, 'user_attrs_window_h_top'].values[0]
        user_attrs_window_w = df_transformer.loc[df_transformer['number'] == transformer_id, 'user_attrs_window_w'].values[0]

        transformer_data = (f"{params_core_name=}\n"
                            f"{params_material_name=}\n"
                            f"{params_n_p_bot=}\n"
                            f"{params_n_p_top=}\n"
                            f"{params_n_s_bot=}\n"
                            f"{params_primary_litz_name=}\n"
                            f"{params_secondary_litz_name=}\n"
                            f"{params_window_h_bot=}\n"
                            f"{user_attrs_core_inner_diameter=}\n"
                            f"{user_attrs_l_bot_air_gap=}\n"
                            f"{user_attrs_l_top_air_gap=}\n"
                            f"{user_attrs_window_h_bot=}\n"
                            f"{user_attrs_window_h_top=}\n"
                            f"{user_attrs_window_w=}\n"

                            )
        with open(f"{output_filepath}/transformer_{transformer_number}_data.txt", "w", encoding="utf-8") as f:
            f.write(transformer_data)

        core = fmt.core_database()[params_core_name]

        lower_core_height_difference = core["window_h"] - params_window_h_bot

        pq_core_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "pq_core_half.py")

        success = DataGeneration._run_freecad(
            freecad_script_file=pq_core_filepath,
            output_file=f"{output_filepath}/transformer_{transformer_number}_core_lower.step",
            variables={
                "core_h_mm": (core["core_h"] - lower_core_height_difference) * FACTOR_M_TO_MM,
                "core_inner_diameter_mm": user_attrs_core_inner_diameter * FACTOR_M_TO_MM,
                "window_h_mm": params_window_h_bot * FACTOR_M_TO_MM,
                "window_w_mm": user_attrs_window_w * FACTOR_M_TO_MM,
                "core_dimension_x_mm": core["core_dimension_x"] * FACTOR_M_TO_MM,
                "core_dimension_y_mm": core["core_dimension_y"] * FACTOR_M_TO_MM,
                "l_air_gap_mm": user_attrs_l_bot_air_gap * FACTOR_M_TO_MM,
            }
        )

        if not success:
            raise RuntimeError("STEP export failed.")

        success = DataGeneration._run_freecad(
            freecad_script_file=pq_core_filepath,
            output_file=f"{output_filepath}/transformer_{transformer_number}_core_original.step",
            variables={
                "core_h_mm": core["core_h"] * FACTOR_M_TO_MM,
                "core_inner_diameter_mm": core["core_inner_diameter"] * FACTOR_M_TO_MM,
                "window_h_mm": core["window_h"] * FACTOR_M_TO_MM,
                "window_w_mm": core["window_w"] * FACTOR_M_TO_MM,
                "core_dimension_x_mm": core["core_dimension_x"] * FACTOR_M_TO_MM,
                "core_dimension_y_mm": core["core_dimension_y"] * FACTOR_M_TO_MM,
                "l_air_gap_mm": 0,
            }
        )

        if not success:
            raise RuntimeError("STEP export failed.")

        upper_core_height_difference = core["window_h"] - 2 * user_attrs_window_h_top

        success = DataGeneration._run_freecad(
            freecad_script_file=pq_core_filepath,
            output_file=f"{output_filepath}/transformer_{transformer_number}_core_upper.step",
            variables={
                "core_h_mm": (core["core_h"] - upper_core_height_difference) * FACTOR_M_TO_MM,
                "core_inner_diameter_mm": user_attrs_core_inner_diameter * FACTOR_M_TO_MM,
                "window_h_mm": user_attrs_window_h_top * 2 * FACTOR_M_TO_MM,  # upper core half needs twice the window_h
                "window_w_mm": user_attrs_window_w * FACTOR_M_TO_MM,
                "core_dimension_x_mm": core["core_dimension_x"] * FACTOR_M_TO_MM,
                "core_dimension_y_mm": core["core_dimension_y"] * FACTOR_M_TO_MM,
                "l_air_gap_mm": user_attrs_l_top_air_gap * FACTOR_M_TO_MM * 2,  # upper core half needs the full air gap, not the reduced one
            }
        )

        if not success:
            raise RuntimeError("STEP export failed.")

        # bobbin generation
        bobbin_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "pq_bobbin.py")

        # generate transformer upper bobbin
        success = DataGeneration._run_freecad(
            freecad_script_file=bobbin_filepath,
            output_file=f"{output_filepath}/transformer_{transformer_number}_bobbin_upper.step",

            variables={
                "window_h_mm": user_attrs_window_h_top * FACTOR_M_TO_MM,
                "window_w_mm": user_attrs_window_w * FACTOR_M_TO_MM,
                "core_inner_diameter_mm": user_attrs_core_inner_diameter * FACTOR_M_TO_MM,

                # Bobbin dimensions
                "flange_thickness_mm": 0.002 * FACTOR_M_TO_MM,

                "clearance": 0.3,
                "inner_edge_radius": 0.6,
                "outer_edge_radius": 0.6,
                "enable_wire_slots": True,
                "wire_slots_position": "both",
                "wire_slot_width": 4.0
            }
        )

        if not success:
            raise RuntimeError("STEP export failed.")

        # generate transformer lower bobbin
        success = DataGeneration._run_freecad(
            freecad_script_file=bobbin_filepath,
            output_file=f"{output_filepath}/transformer_{transformer_number}_bobbin_lower.step",

            variables={
                "window_h_mm": params_window_h_bot * FACTOR_M_TO_MM,
                "window_w_mm": user_attrs_window_w * FACTOR_M_TO_MM,
                "core_inner_diameter_mm": user_attrs_core_inner_diameter * FACTOR_M_TO_MM,

                # Bobbin dimensions
                "flange_thickness_mm": 0.002 * FACTOR_M_TO_MM,

                "clearance": 0.3,
                "inner_edge_radius": 0.6,
                "outer_edge_radius": 0.6,
                "enable_wire_slots": True,
                "wire_slots_position": "both",
                "wire_slot_width": 4.0
            }
        )

        if not success:
            raise RuntimeError("STEP export failed.")

    @staticmethod
    def _generate_heat_sink_data(heat_sink_id: int, df_heat_sink: pd.DataFrame, output_filepath: str) -> None:
        """
        Generate heat sink manufacturing data.

        :param heat_sink_id: heat sink ID
        :type heat_sink_id: int
        :param df_heat_sink: heat sink dataframe
        :type df_heat_sink: pd.DataFrame
        :param output_filepath: output filepath
        :type output_filepath: str
        """
        params_fan = df_heat_sink.loc[df_heat_sink['number'] == heat_sink_id, 'params_fan'].values[0]
        params_height_c = df_heat_sink.loc[df_heat_sink['number'] == heat_sink_id, 'params_height_c'].values[0]
        params_height_d = df_heat_sink.loc[df_heat_sink['number'] == heat_sink_id, 'params_height_d'].values[0]
        params_length_l = df_heat_sink.loc[df_heat_sink['number'] == heat_sink_id, 'params_length_l'].values[0]
        params_number_cooling_channels_n = df_heat_sink.loc[df_heat_sink['number'] == heat_sink_id, 'params_number_cooling_channels_n'].values[0]
        params_thickness_fin_t = df_heat_sink.loc[df_heat_sink['number'] == heat_sink_id, 'params_thickness_fin_t'].values[0]
        params_width_b = df_heat_sink.loc[df_heat_sink['number'] == heat_sink_id, 'params_width_b'].values[0]

        heat_sink_data = (
            f"{params_fan=}\n"
            f"{params_height_c=}\n"
            f"{params_height_d=}\n"
            f"{params_length_l=}\n"
            f"{params_number_cooling_channels_n=}\n"
            f"{params_thickness_fin_t=}\n"
            f"{params_width_b=}\n"
        )
        with open(f"{output_filepath}/heat_sink_data.txt", "w", encoding="utf-8") as f:
            f.write(heat_sink_data)

    @staticmethod
    def generate_manufacturing_data(debug: tc.Debug,
                                    circuit_configuration: CircuitOptimizationBase,
                                    inductor_configuration_list: list[InductorConfiguration],
                                    transformer_configuration_list: list[TransformerConfiguration],
                                    capacitor_configuration_list: list[CapacitorConfiguration],
                                    heat_sink_configuration: StudyData,
                                    summary_data: StudyData, data_generation_data: StudyData) -> None:
        """
        Generate data for all components to enable the manufacturing process.

        :param debug: Debug configuration
        :type debug: tc.Debug
        :param circuit_configuration: circuit configuration
        :type circuit_configuration:
        :param inductor_configuration_list: inductor configuration list
        :type inductor_configuration_list: list[InductorConfiguration]
        :param transformer_configuration_list: transformer configuration list
        :type transformer_configuration_list: list[TransformerConfiguration]
        :param capacitor_configuration_list: capacitor configuration list
        :type capacitor_configuration_list: list[CapacitorConfiguration]
        :param heat_sink_configuration: heat sink configuration
        :type heat_sink_configuration: StudyData
        :param summary_data: summary data
        :type summary_data: StudyData
        :param data_generation_data: data generation data
        :type data_generation_data: StudyData
        """
        # read summary parameters
        summary_filepath = os.path.join(summary_data.optimization_directory, DF_SUMMARY_FINAL_FILTERED)

        df_summary = pd.read_csv(summary_filepath)

        if debug.general.is_debug:
            # reduce dataset to the given number from the debug configuration
            df_summary = df_summary.iloc[:debug.data_generation.number_combinations_max]

        for combination_id in df_summary["combination_id"]:
            logger.info(f"Generate manufacturing data for {combination_id=}")

            circuit_id, capacitor_id_list, inductor_id_list, transformer_id_list, heat_sink_id = DataGeneration._read_summary_parameters(
                combination_id, df_summary)

            # read circuit file
            circuit_filepath = os.path.join(circuit_configuration.circuit_study_data.optimization_directory,
                                            f"{circuit_configuration.circuit_study_data.study_name}.csv")

            output_filepath = os.path.join(data_generation_data.optimization_directory, str(combination_id))
            if not os.path.exists(output_filepath):
                os.makedirs(output_filepath)

            df_circuit = pd.read_csv(circuit_filepath)
            DataGeneration._generate_circuit_data(circuit_id, df_circuit, output_filepath)

            # read capacitor file
            for count, capacitor_id in enumerate(capacitor_id_list):
                capacitor_filepath = os.path.join(capacitor_configuration_list[count].study_data.optimization_directory,
                                                  str(circuit_id), capacitor_configuration_list[count].study_data.study_name, "results.csv")
                df_capacitor = pd.read_csv(capacitor_filepath)
                DataGeneration._generate_capacitor_data(capacitor_id, df_capacitor, output_filepath, count)

            # read inductor file
            for count, inductor_id in enumerate(inductor_id_list):
                inductor_filepath = os.path.join(inductor_configuration_list[count].study_data.optimization_directory, str(circuit_id),
                                                 inductor_configuration_list[count].study_data.study_name,
                                                 f"{inductor_configuration_list[count].study_data.study_name}.csv")
                df_inductor = pd.read_csv(inductor_filepath)
                DataGeneration._generate_inductor_data(inductor_id, df_inductor, output_filepath, count)

            # read transformer file
            for count, transformer_id in enumerate(transformer_id_list):
                transformer_filepath = os.path.join(transformer_configuration_list[count].study_data.optimization_directory, str(circuit_id),
                                                    transformer_configuration_list[count].study_data.study_name,
                                                    f"{transformer_configuration_list[count].study_data.study_name}.csv")
                df_transformer = pd.read_csv(transformer_filepath)
                DataGeneration._generate_transformer_data(transformer_id, df_transformer, output_filepath, count)

            # read heat sink file
            heat_sink_filepath = os.path.join(heat_sink_configuration.optimization_directory, f"{heat_sink_configuration.study_name}.csv")
            df_heat_sink = pd.read_csv(heat_sink_filepath)
            DataGeneration._generate_heat_sink_data(heat_sink_id, df_heat_sink, output_filepath)
