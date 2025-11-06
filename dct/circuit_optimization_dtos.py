"""Data transfer objects (DTOs) for the Pareto optimization."""
# python libraries
import dataclasses

# 3rd party libraries

# own libraries
from dct.circuit_enums import SamplingEnum

@dataclasses.dataclass
class ParetoFilePaths:
    """File paths for the sub simulation optimization parts."""

    circuit: str
    inductor: str
    transformer: str
    heat_sink: str

@dataclasses.dataclass
class CircuitParetoDesignSpace:
    """Definition of the hardware design space for electronic components."""

    # DAB optimization parameters
    f_s_min_max_list: list[int]
    l_s_min_max_list: list[float]
    transistor_1_name_list: list[str]
    transistor_2_name_list: list[str]

@dataclasses.dataclass
class CircuitOutputRange:
    """Definition of the DAB operating area."""

    v1_min_max_list: list
    duty_cycle_min_max_list: list
    i_min_max_list: list

@dataclasses.dataclass
class CircuitSampling:
    """Definition of the sampling method."""

    sampling_method: SamplingEnum
    sampling_points: int
    sampling_random_seed: int | None
    v1_additional_user_point_list: list[float]
    duty_cycle_additional_user_point_list: list[float]
    i_additional_user_point_list: list[float]
    additional_user_weighting_point_list: list[float]

@dataclasses.dataclass
class CircuitFilter:
    """Filter the results."""

    number_filtered_designs: int
    difference_percentage: float

@dataclasses.dataclass
class CircuitParetoSbcDesign:
    """Config to optimize the Dual-Active Bridge (DAB) converter."""

    circuit_study_name: str
    project_directory: str

    design_space: CircuitParetoDesignSpace
    output_range: CircuitOutputRange
    sampling: CircuitSampling
    filter: CircuitFilter
