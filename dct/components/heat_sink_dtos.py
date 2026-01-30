"""Data transfer objects (DTOs) for the heat sink optimization."""

# python libraries
import dataclasses

@dataclasses.dataclass
class HeatSinkBoundaryConditions:
    """Fix parameters for the heat sink cooling."""

    t_ambient: float
    t_hs_max: float
