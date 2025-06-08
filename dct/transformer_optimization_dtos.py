"""Data transfer objects (DTOs) for the transformer optimization."""

# python libraries
import dataclasses

# 3rd party libraries
import femmt as fmt

# own libraries
from dct.server_ctl_dtos import StatData as StData


@dataclasses.dataclass
class TransformerOptimizationDto:
    """DTO for the transformer optimization."""

    circuit_id: int
    stat_data: StData
    transformer_optimization_dto: fmt.fmt.StoSingleInputConfig
