"""Data transfer objects (DTOs) for the transformer optimization."""

# python libraries
import dataclasses

# 3rd party libraries
import femmt as fmt

# own libraries
from dct.server_ctl_dtos import ProgressData


@dataclasses.dataclass
class TransformerOptimizationDto:
    """DTO for the transformer optimization."""

    circuit_filtered_point_filename: str
    progress_data: ProgressData
    transformer_optimization_dto: fmt.fmt.StoSingleInputConfig
