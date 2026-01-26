"""Data transfer objects (DTOs) for the transformer optimization."""

# python libraries
import dataclasses

# 3rd party libraries
import femmt as fmt

# own libraries
from dct.server_ctl_dtos import ProgressData
from dct.components.component_dtos import TransformerRequirements, ComponentCooling


@dataclasses.dataclass
class TransformerOptimizationDto:
    """DTO for the transformer optimization."""

    trial_directory: str
    circuit_id: str
    transformer_number_in_circuit: int
    progress_data: ProgressData
    fmt_transformer_optimization_dto: fmt.fmt.StoSingleInputConfig
    number_of_trails: int
    thermal_data: ComponentCooling
    factor_dc_losses_min_max_list: list[float]
    transformer_requirements: TransformerRequirements
