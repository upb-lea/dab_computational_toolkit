"""Data transfer objects (DTOs) for the capacitor optimization."""

# python libraries
import dataclasses

# 3rd party libraries
import pecst
# own libraries
from dct.server_ctl_dtos import ProgressData

@dataclasses.dataclass
class CapacitorOptimizationDto:
    """DTO for the inductor optimization."""

    circuit_filtered_point_filename: str
    progress_data: ProgressData
    capacitor_optimization_dto: pecst.CapacitorRequirements