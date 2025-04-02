"""Inductor optimization class."""
# python libraries
import os

# 3rd party libraries

# own libraries
import hct
import dct
from dct.heat_sink_dtos import *

# configure root logger
# logging.basicConfig(format='%(levelname)s,%(asctime)s:%(message)s', encoding='utf-8')
# logging.getLogger().setLevel(logging.ERROR)

class HeatSinkOptimization:
    """Optimation support class for heat sink optimation."""

    # Simulation configuration list
    sim_config_list = []

    @staticmethod
    def init_configuration(toml_heat_sink: dct.TomlHeatSink, toml_prog_flow: dct.FlowControl) -> bool:
        """
        Initialize the configuration.

        :param toml_heat_sink: toml heat sink class
        :type toml_heat_sink: dct.TomlHeatSink
        :param toml_prog_flow: toml program flow class
        :type toml_prog_flow: dct.FlowControl
        :return: True, if the configuration was successful initialized
        :rtype: bool
        """
        # Variable declaration
        # Return variable initialized to False
        heat_sink_optimization_successful = False

        # Check if path exists
        if not os.path.exists(toml_heat_sink.fan_data.heat_sink_fan_path):
            print(f"Fan data path {toml_heat_sink.fan_data.heat_sink_fan_path} does not exists!")
            # Return with false if path does not exist
            return heat_sink_optimization_successful

        # Generate the fan-list
        for (_, _, file_name_list) in os.walk(toml_heat_sink.fan_data.heat_sink_fan_path):
            fan_list = file_name_list

        if len(fan_list) == 0:
            print(f"No fan design data found in path {toml_heat_sink.fan_data.heat_sink_fan_path}!")
            # Return with false
            return heat_sink_optimization_successful

        heat_sink_study_name = toml_prog_flow.configuration_data_files.heat_sink_configuration_file.replace(".toml", "")

        hct_config = hct.OptimizationParameters(

            # general parameters
            heat_sink_study_name=heat_sink_study_name,
            heat_sink_optimization_directory=os.path.join(toml_prog_flow.general.project_directory, toml_prog_flow.heat_sink.subdirectory,
                                                          toml_prog_flow.configuration_data_files.heat_sink_configuration_file.replace(".toml", "")),

            # geometry parameters
            height_c_list=toml_heat_sink.design_space.height_c_list,
            width_b_list=toml_heat_sink.design_space.width_b_list,
            length_l_list=toml_heat_sink.design_space.length_l_list,
            height_d_list=toml_heat_sink.design_space.height_d_list,
            number_fins_n_list=toml_heat_sink.design_space.number_fins_n_list,
            thickness_fin_t_list=toml_heat_sink.design_space.thickness_fin_t_list,
            fan_list=fan_list,

            # boundary conditions
            t_ambient=toml_heat_sink.boundary_conditions.t_ambient,
            area_min=toml_heat_sink.boundary_conditions.area_min,

            # constraints
            number_directions=toml_heat_sink.settings.number_directions
        )

        # Empty the list
        HeatSinkOptimization.sim_config_list = []
        # Add configuration to list
        HeatSinkOptimization.sim_config_list.append(hct_config)
        print(f"{HeatSinkOptimization.sim_config_list=}")
        # Set return value to true and return
        heat_sink_optimization_successful = True

        return heat_sink_optimization_successful

    @staticmethod
    def calculate_r_th_copper_coin(cooling_area: float, height_pcb: float = 1.55e-3,
                                   height_pcb_heat_sink: float = 3.0e-3) -> tuple[float, float]:
        """
        Calculate the thermal resistance of the copper coin.

        Assumptions are made with some geometry factors from a real copper coin for TO263 housing.
        :param cooling_area: cooling area in m²
        :type cooling_area: float
        :param height_pcb: PCB thickness, e.g. 1.55 mm
        :type height_pcb: float
        :param height_pcb_heat_sink: Distance from PCB to heat sink in m
        :type height_pcb_heat_sink: float
        :return: r_th_copper_coin, effective_copper_coin_cooling_area
        :rtype: tuple[float, float]
        """
        factor_pcb_area_copper_coin = 1.42
        factor_bottom_area_copper_coin = 0.39
        thermal_conductivity_copper = 136  # W/(m*K)

        effective_pcb_cooling_area = cooling_area / factor_pcb_area_copper_coin
        effective_bottom_cooling_area = effective_pcb_cooling_area / factor_bottom_area_copper_coin

        r_pcb = 1 / thermal_conductivity_copper * height_pcb / effective_pcb_cooling_area
        r_bottom = 1 / thermal_conductivity_copper * height_pcb_heat_sink / effective_bottom_cooling_area

        r_copper_coin = r_pcb + r_bottom

        return r_copper_coin, effective_bottom_cooling_area

    @staticmethod
    def calculate_r_th_tim(copper_coin_bot_area: float, transistor_cooling: TransistorCooling) -> float:
        """
        Calculate the thermal resistance of the thermal interface material (TIM).

        :param copper_coin_bot_area: bottom copper coin area in m²
        :type copper_coin_bot_area: float
        :param transistor_cooling: Transistor cooling DTO
        :type transistor_cooling: TransistorCooling
        :return: r_th of TIM material
        :rtype: float
        """
        r_th_tim = 1 / transistor_cooling.tim_conductivity * transistor_cooling.tim_thickness / copper_coin_bot_area

        return r_th_tim

    # Simulation handler. Later the simulation handler starts a process per list entry.
    @staticmethod
    def _simulation(act_hct_config: hct.OptimizationParameters, act_ginfo: dct.GeneralInformation,
                    target_number_trials: int, re_simulate: bool, debug: bool):
        """
        Perform the simulation.

        :param circuit_id: Name of the filtered optimal electrical circuit
        :type  circuit_id: int
        :param act_io_config: inductor configuration for the optimization
        :type  act_io_config: fmt.InductorOptimizationDTO
        :param act_ginfo: General information about the study
        :type  act_ginfo: dct.GeneralInformation:
        :param target_number_trials: Number of trials for the optimization
        :type  target_number_trials: int
        :param re_simulate: Flag to control, if the point are to re-simulate (ASA: Correct the parameter description)
        :type  re_simulate: bool
        :param debug: Debug mode flag
        :type debug: bool
        """
        # Variable declaration

        # Check number of trials
        if target_number_trials > 0:
            print(f"{HeatSinkOptimization.sim_config_list=}")
            hct.Optimization.start_proceed_study(config=act_hct_config, number_trials=target_number_trials)
        else:
            print(f"Target number of trials = {target_number_trials} which are less equal 0!. No simulation is performed")

        # Plot options ASA: Later to add to server
        # df_heat_sink = hopt.Optimization.study_to_df(act_hct_config)
        # hopt.Optimization.df_plot_pareto_front(df_heat_sink, (50, 60))

    # Simulation handler. Later the simulation handler starts a process per list entry.
    @staticmethod
    def simulation_handler(act_ginfo: dct.GeneralInformation, target_number_trials: int,
                           re_simulate: bool = False, debug: bool = False):
        """
        Control the multi simulation processes.

        :param act_ginfo : General information about the study
        :type  act_ginfo : dct.GeneralInformation:
        :param target_number_trials : Number of trials for the optimization
        :type  target_number_trials : int
        :param re_simulate : Flag to control, if the point are to re-simulate (ASA: Correct the parameter description)
        :type  re_simulate : bool
        :param debug : Debug mode flag
        :type  debug : bool
        """
        # Later this is to parallelize with multiple processes
        print(f"{HeatSinkOptimization.sim_config_list=}")
        for act_sim_config in HeatSinkOptimization.sim_config_list:
            # Debug switch
            if target_number_trials != 0:
                if debug:
                    # overwrite input number of trials with 100 for short simulation times
                    if target_number_trials > 100:
                        target_number_trials = 100

            HeatSinkOptimization._simulation(act_sim_config, act_ginfo, target_number_trials, re_simulate, debug)
            if debug:
                # stop after one circuit run
                break
