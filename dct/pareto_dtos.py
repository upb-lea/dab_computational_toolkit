"""Data transfer objects (DTOs) for the Pareto optimization."""
# python libraries
import dataclasses

# 3rd party libraries
import numpy as np

@dataclasses.dataclass
class FilePaths:
    """File paths for the sub simulation optimization parts."""

    circuit: str
    transformer: str
    inductor: str
    heat_sink: str

@dataclasses.dataclass
class DesignSpace:
    """Definition of the hardware design space for electronic components."""

    # DAB optimization parameters
    f_s_min_max_list: np.array
    l_s_min_max_list: np.array
    l_1_min_max_list: np.array
    l_2_min_max_list: np.array
    n_min_max_list: np.array
    transistor_1_list: list[str]
    transistor_2_list: list[str]

@dataclasses.dataclass
class OutputRange:
    """Definition of the DAB operating area."""

    v_1_min_nom_max_list: list
    v_2_min_nom_max_list: list
    p_min_nom_max_list: list
    steps_per_direction: int

@dataclasses.dataclass
class DabDesign:
    """Config to optimize the Dual-Active Bridge (DAB) converter."""

    dab_study_name: str
    project_directory: str

    design_space: DesignSpace
    output_range: OutputRange
