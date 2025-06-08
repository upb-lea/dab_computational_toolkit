"""Shared memory data transfer objects (DTOs) for the server."""

# python libraries
import dataclasses

# own libraries


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
class StatData:
    """Statistic data of heat sink optimization."""

    start_proc_time: float
    proc_run_time: float
    nb_of_filtered_points: int
    status: int

@dataclasses.dataclass
class ConfigurationDataEntryDto:
    """DTO for queue configuration data information transfer."""

    conf_name: str
    nb_of_trials: int
    progress_data: StatData

@dataclasses.dataclass
class CircuitConfigurationDataDto:
    """DTO for queue circuit configuration information transfer."""

    conf_name: str
    nb_of_trials: int
    filtered_points_name_list: list[str]
    progress_data: StatData

@dataclasses.dataclass
class SummaryDataEntryDto:
    """DTO for queue summary information transfer."""

    conf_name: str
    nb_of_combinations: int
    progress_data: StatData

@dataclasses.dataclass
class QueueMainData:
    """DTO for one shared memory data exchange within a queue."""

    circuit_list: list[ConfigurationDataEntryDto]
    heat_sink_list: list[ConfigurationDataEntryDto]
    summary_list: list[SummaryDataEntryDto]
    # For future usage
    # final_summary:  SummaryDataEntryDto
    total_proc_time: float
    inductor_proc_time: float
    transformer_proc_time: float

@dataclasses.dataclass
class QueueDetailData:
    """DTO for one shared memory data exchange within a queue."""

    circuit_data: CircuitConfigurationDataDto
    inductor_list: list[ConfigurationDataEntryDto]
    transformer_list: list[ConfigurationDataEntryDto]
    heat_sink_list: list[ConfigurationDataEntryDto]
    summary_data: SummaryDataEntryDto
    conf_proc_time: float
