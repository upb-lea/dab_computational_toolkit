"""Shared memory data transfer objects (DTOs) for the server."""

# python libraries
import dataclasses
from enum import Enum
from typing import Any
import threading
import time

# own libraries

# Structure class of ProgressStatus
class ProgressStatus(Enum):
    """Enum of progress status."""

    Idle = 0            # Calculation not started
    InProgress = 1      # Calculation started
    Done = 2            # Calculation performed, results are available
    Skipped = 3         # Skip calculation and use results of previous calculation

@dataclasses.dataclass
class RunTimeMeasure:
    """Statistic data of heat sink optimization."""

    total_time: float
    circuit_time_list: list[float]
    inductor_time_list: list[float]
    transformer_time_list: list[float]
    heat_sink_time_list: list[float]
    summary_time_list: list[float]

@dataclasses.dataclass
class ProgressData:
    """Statistic data of heat sink optimization."""

    run_time: float
    number_of_filtered_points: int
    progress_status: ProgressStatus

@dataclasses.dataclass
class ConfigurationDataEntryDto:
    """DTO for queue configuration data information transfer."""

    configuration_name: str
    number_of_trials: int
    progress_data: ProgressData

@dataclasses.dataclass
class CircuitConfigurationDataDto:
    """DTO for queue circuit configuration information transfer."""

    configuration_name: str
    number_of_trials: int
    filtered_points_name_list: list[tuple[str, int] | Any]
    progress_data: ProgressData

@dataclasses.dataclass
class SummaryDataEntryDto:
    """DTO for queue summary information transfer."""

    configuration_name: str
    number_of_combinations: int
    progress_data: ProgressData

@dataclasses.dataclass
class MagneticDataEntryDto:
    """DTO for queue summary information transfer."""

    magnetic_configuration_name: str
    number_calculations: int
    number_performed_calculations: int
    progress_data: ProgressData

@dataclasses.dataclass
class QueueMainData:
    """DTO for one shared memory data exchange within a queue."""

    circuit_list: list[ConfigurationDataEntryDto]
    inductor_main_list: list[MagneticDataEntryDto]
    transformer_main_list: list[MagneticDataEntryDto]
    heat_sink_list: list[ConfigurationDataEntryDto]
    summary_list: list[SummaryDataEntryDto]
    # For future usage
    # final_summary:  SummaryDataEntryDto
    total_process_time: float
    break_point_notification: str

@dataclasses.dataclass
class QueueDetailData:
    """DTO for one shared memory data exchange within a queue."""

    circuit_data: CircuitConfigurationDataDto
    inductor_list: list[ConfigurationDataEntryDto]
    transformer_list: list[ConfigurationDataEntryDto]
    heat_sink_list: list[ConfigurationDataEntryDto]
    summary_data: SummaryDataEntryDto
    conf_process_time: float
    break_point_notification: str

@dataclasses.dataclass
class QueueParetoFrontData:
    """DTO for one shared memory data exchange within a queue."""

    pareto_front_optuna: str
    evaluation_info: str
    validity: bool


# Class for runtime measurement
class RunTimeMeasurement:
    """Runtime class for measure processing time."""

    def __init__(self) -> None:
        self._start_time: float = 0.0
        self._run_time: float = 0.0
        self._timer_flag: bool = False
        self._timer_lock: threading.Lock = threading.Lock()

    def reset_start_trigger(self) -> None:
        """Reset and start the timer."""
        with self._timer_lock:
            self._start_time = time.perf_counter()
            self._run_time = 0.0
            self._timer_flag = True

    def continue_trigger(self) -> None:
        """Continue the timer without reset."""
        if not self._timer_flag:
            with self._timer_lock:
                self._start_time = time.perf_counter() - self._run_time
                self._timer_flag = True

    def stop_trigger(self) -> None:
        """Stop the timer."""
        if self._timer_flag:
            with self._timer_lock:
                self._timer_flag = False
                self._run_time = time.perf_counter() - self._start_time

    def is_timer_active(self) -> bool:
        """Provide the timer state.

        :return: True, if the time is active
        :rtype: bool
        """
        return self._timer_flag

    def get_runtime(self) -> float:
        """Provide the current measured time since timer start.

        :return: time in seconds
        :rtype: float
        """
        # Variable declaration
        run_time: float = 0.0
        with self._timer_lock:
            if self._timer_flag:
                run_time = time.perf_counter() - self._start_time
            else:
                run_time = self._run_time
        return run_time
