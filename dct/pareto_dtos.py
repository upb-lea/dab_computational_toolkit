"""Data transfer objects (DTOs) for the Pareto optimization."""
# python libraries
import dataclasses

import numpy as np


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

    # misc
    working_directory: str

@dataclasses.dataclass
class WorkArea:
    """Definition of the DAB operating area."""

    v_1_min_nom_max_list: list
    v_2_min_nom_max_list: list
    p_min_nom_max_list: list
    steps_per_direction: int
