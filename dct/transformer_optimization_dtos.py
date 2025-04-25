"""Data transfer objects (DTOs) for the transformer optimization."""

# python libraries
import dataclasses

# 3rd party libraries
import femmt as fmt

# own libraries


@dataclasses.dataclass
class TransformerOptimizationDto:
    """DTO for the transformer optimization."""

    circuit_id: int
    transformer_optimization_dto: fmt.fmt.StoSingleInputConfig
