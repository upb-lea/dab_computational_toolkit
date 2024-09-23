"""Data transfer objects (DTOs) for the Pareto optimization."""
# python libraries
import dataclasses

# 3rd party libraries
import numpy as np

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
    f_s_min_max_list: np.array
    l_s_min_max_list: np.array
    l_1_min_max_list: np.array
    l_2__min_max_list: np.array
    n_min_max_list: np.array
    transistor_1_list: list[str]
    transistor_2_list: list[str]

@dataclasses.dataclass
class CircuitOutputRange:
    """Definition of the DAB operating area."""

    v_1_min_nom_max_list: list
    v_2_min_nom_max_list: list
    p_min_nom_max_list: list
    steps_per_direction: int

@dataclasses.dataclass
class CircuitParetoDabDesign:
    """Config to optimize the Dual-Active Bridge (DAB) converter."""

    circuit_study_name: str
    project_directory: str

    design_space: CircuitParetoDesignSpace
    output_range: CircuitOutputRange
