"""Data transfer objects (DTOs) for the inductor optimization."""

# python libraries
import dataclasses

# 3rd party libraries
import femmt as fmt
# own libraries
from dct.server_ctl_dtos import ProgressData

@dataclasses.dataclass
class InductorOptimizationDto:
    """DTO for the inductor optimization."""

    circuit_filtered_point_filename: str
    progress_data: ProgressData
    inductor_optimization_dto: fmt.InductorOptimizationDTO
