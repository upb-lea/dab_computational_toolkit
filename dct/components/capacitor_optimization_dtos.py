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

    circuit_filtered_point_filename: str
    progress_data: ProgressData
    capacitor_optimization_dto: pecst.CapacitorRequirements

@dataclasses.dataclass(init=False)
class CapacitorResults:
    """DTO contains the inductor losses."""

    loss_total_array: np.ndarray
    volume_total: float
    area_total: float
    circuit_trial_file: str
    capacitor_order_number: str
    n_parallel: int
    n_series: int

    def __init__(self, **kwargs):
        names = set([f.name for f in dataclasses.fields(self)])
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)
