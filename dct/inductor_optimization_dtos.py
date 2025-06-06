"""Data transfer objects (DTOs) for the inductor optimization."""

# python libraries
import dataclasses

# 3rd party libraries
import femmt as fmt
# own libraries
from server_ctl_dtos import StatData as StData

@dataclasses.dataclass
class InductorOptimizationDto:
    """DTO for the inductor optimization."""

    circuit_id: int
    stat_data: StData
    inductor_optimization_dto: fmt.InductorOptimizationDTO
