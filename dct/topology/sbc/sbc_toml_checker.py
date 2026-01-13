"""Classes and toml checker for the flow control."""
# python libraries
from typing import Literal

# 3rd party libraries

# own libraries
from dct.topology.circuit_optimization_base import TomlGData, TomlCData
from dct.circuit_enums import SamplingEnum

# ######################################################
# sbc general
# ######################################################
class TomlSbcParameterRange(TomlGData):
    """Definition of the SBC operating area."""

    v1_min_max_list: list[float]
    v2_min_max_list: list[float]
    i2_min_max_list: list[float]

class TomlSbcSampling(TomlGData):
    """Definition of the sampling method."""

    sampling_method: SamplingEnum
    sampling_points: int
    sampling_random_seed: int | Literal["random"]
    v1_additional_user_point_list: list[float]
    v2_additional_user_point_list: list[float]
    i2_additional_user_point_list: list[float]
    additional_user_weighting_point_list: list[float]

class TomlSbcGeneral(TomlGData):
    """Definition of the general parameters affecting mostly all kind of calculations."""

    parameter_range: TomlSbcParameterRange
    sampling: TomlSbcSampling

# ######################################################
# sbc circuit
# ######################################################

class TomlSbcCircuitParetoDesignSpace(TomlCData):
    """Definition of the hardware design space for electronic components."""

    # SBC optimization parameters
    f_s_min_max_list: list[int]
    l_s_min_max_list: list[float]
    transistor_1_name_list: list[str]
    transistor_2_name_list: list[str]
    c_par_1: float
    c_par_2: float

class TomlSbcCircuitFilterDistance(TomlCData):
    """Toml checker class for CircuitFilterDistance."""

    number_filtered_designs: int
    difference_percentage: float

class TomlSbcCircuitParetoDesign(TomlCData):
    """Config to optimize the Dual-Active Bridge (SBC) converter."""

    design_space: TomlSbcCircuitParetoDesignSpace
    filter_distance: TomlSbcCircuitFilterDistance
