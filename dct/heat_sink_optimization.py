"""Inductor optimization class."""
# python libraries
import os
import time
import copy
import threading
import logging

# 3rd party libraries

# own libraries
import hct
import dct
from dct.server_ctl_dtos import ProgressData
from dct.server_ctl_dtos import ProgressStatus
from dct.heat_sink_dtos import *

logger = logging.getLogger(__name__)

class HeatSinkOptimization:
    """Optimization support class for heat sink optimization."""

    # Simulation configuration list
    _hct_config: hct.OptimizationParameters
    _progress_data: ProgressData

    def __init__(self) -> None:
        """Initialize the configuration list for the heat sink optimizations."""
        self._hct_config: hct.OptimizationParameters
        self._progress_data: ProgressData = ProgressData(start_time=0.0, run_time=0.0, number_of_filtered_points=0,
                                                         progress_status=ProgressStatus.Idle)
        self._h_lock_stat: threading.Lock = threading.Lock()

    def generate_optimization_list(self, toml_heat_sink: dct.TomlHeatSink, toml_prog_flow: dct.FlowControl) -> bool:
        """
        Initialize the configuration.

        :param toml_heat_sink: toml heat sink class
        :type toml_heat_sink: dct.TomlHeatSink
        :param toml_prog_flow: toml program flow class
        :type toml_prog_flow: dct.FlowControl
        :return: True, if the configuration was successful initialized
        :rtype: bool
        """
        is_list_generation_successful = False
        heat_sink_fan_datapath = os.path.join(os.path.dirname(hct.__file__), "data")

        # Check if path exists
        if not os.path.exists(heat_sink_fan_datapath):
            logger.info(f"Fan data path {heat_sink_fan_datapath} does not exists!")
        # Generate the fan-list
        for (_, _, file_name_list) in os.walk(heat_sink_fan_datapath):
            fan_list = file_name_list

        if not fan_list:
            logger.info(f"No fan design data found in path {heat_sink_fan_datapath}!")

        heat_sink_study_name = toml_prog_flow.configuration_data_files.heat_sink_configuration_file.replace(".toml", "")

        self._hct_config = hct.OptimizationParameters(

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

        is_list_generation_successful = True

        return is_list_generation_successful

    def get_progress_data(self) -> ProgressData:
        """Provide the progress data of the optimization.

        :return: Progress data: Processing start time, actual processing time and status.
                 number of filtered heat sink Pareto front points are obsolete
        :rtype: ProgressData
        """
        # Lock statistical performance data access
        with self._h_lock_stat:
            # Update statistical data if optimisation is running
            if self._progress_data.progress_status == ProgressStatus.InProgress:
                self._progress_data.run_time = time.perf_counter() - self._progress_data.start_time
                # Check for valid entry
                if self._progress_data.run_time < 0:
                    self._progress_data.run_time = 0.0
                    self._progress_data.start_time = time.perf_counter()

        return copy.deepcopy(self._progress_data)

    # Simulation handler. Later the simulation handler starts a process per list entry.
    @staticmethod
    def _optimize(act_hct_config: hct.OptimizationParameters, target_number_trials: int, debug: bool) -> None:
        """
        Perform the simulation.

        :param target_number_trials: Number of trials for the optimization
        :type  target_number_trials: int
        :param debug: Debug mode flag
        :type debug: bool
        """
        # Check number of trials
        if target_number_trials > 0:
            hct.Optimization.start_proceed_study(config=act_hct_config, number_trials=target_number_trials)
        else:
            logger.info(f"Target number of trials = {target_number_trials} which are less equal 0!. No simulation is performed")

        # Plot options ASA: Later to add to server
        # df_heat_sink = hopt.Optimization.study_to_df(act_hct_config)
        # hopt.Optimization.df_plot_pareto_front(df_heat_sink, (50, 60))

    # Simulation handler. Later the simulation handler starts a process per list entry.

    def optimization_handler(self, target_number_trials: int, debug: bool = False) -> None:
        """
        Control the multi simulation processes.

        :param target_number_trials: Number of trials for the optimization
        :type  target_number_trials: int
        :param debug: Debug mode flag
        :type  debug: bool
        """
        # Update statistical data
        with self._h_lock_stat:
            self._progress_data.start_time = time.perf_counter()
            self._progress_data.run_time = 0.0
            self._progress_data.progress_status = ProgressStatus.InProgress

        # Perform optimization
        # Debug switch
        if target_number_trials != 0:
            if debug:
                # overwrite input number of trials with 100 for short simulation times
                if target_number_trials > 100:
                    target_number_trials = 100

        HeatSinkOptimization._optimize(self._hct_config, target_number_trials, debug)

        # Update statistical data
        with self._h_lock_stat:
            self._progress_data.run_time = time.perf_counter() - self._progress_data.start_time
            # Check for valid entry
            if self._progress_data.run_time < 0:
                self._progress_data.run_time = 0.0
                self._progress_data.start_time = time.perf_counter()
            self._progress_data.progress_status = ProgressStatus.Done

class ThermalCalcSupport:
    """Provides functions to calculate the thermal resistance."""

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
