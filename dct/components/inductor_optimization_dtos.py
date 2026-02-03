"""Data transfer objects (DTOs) for the inductor optimization."""

# python libraries
import dataclasses

# 3rd party libraries
import femmt as fmt

# own libraries
from dct.server_ctl_dtos import ProgressData
from dct.components.component_dtos import InductorRequirements, ComponentCooling

@dataclasses.dataclass
class InductorOptimizationDto:
    """DTO for the inductor optimization."""

    trial_directory: str
    circuit_id: str
    inductor_number_in_circuit: int
    progress_data: ProgressData
    fmt_inductor_optimization_dto: fmt.InductorOptimizationDTO
    number_of_trails: int
    thermal_data: ComponentCooling
    factor_dc_losses_min_max_list: list[float]
    inductor_requirements: InductorRequirements
