"""Data transfer objects (DTOs) for the capacitor optimization."""

# python libraries
import dataclasses

# 3rd party libraries
import pecst
import numpy as np

# own libraries
from dct.server_ctl_dtos import ProgressData

@dataclasses.dataclass
class CapacitorOptimizationDto:
    """DTO for the inductor optimization."""

    circuit_id: str
    progress_data: ProgressData
    capacitor_optimization_dto: pecst.CapacitorRequirements
    capacitor_number_in_circuit: int
    time_array: np.ndarray
    current_array: np.ndarray

@dataclasses.dataclass(init=False)
class CapacitorResults:
    """DTO contains the inductor losses."""

    # identification
    circuit_id: str
    capacitor_id: str
    capacitor_number_in_circuit: int

    # pareto
    loss_total_array: np.ndarray
    volume_total: float
    area_total: float
    n_parallel: int
    n_series: int

    def __init__(self, **kwargs):
        names = set([f.name for f in dataclasses.fields(self)])
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)
