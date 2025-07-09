"""Shared memory data transfer objects (DTOs) for the server."""

# python libraries
import dataclasses
from enum import Enum
from typing import Any

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

    start_time: float
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

    parto_front_optuna: str
    evaluation_info: str
    validity: bool
