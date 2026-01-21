"""Data transfer objects (DTOs) for the heat sink optimization."""

# python libraries
import dataclasses


@dataclasses.dataclass
class HeatSinkBoundaryConditions:
    """Fix parameters for the heat sink cooling."""

    t_ambient: float
    t_hs_max: float

@dataclasses.dataclass
class ComponentCooling:
    """Fix parameters for the transistor, inductor and transformer cooling."""

    tim_conductivity: float
    tim_thickness: float
