"""Data transfer objects (DTOs) for the heat sink optimization."""

# python libraries
import dataclasses


@dataclasses.dataclass
class HeatSinkTemp:
    """Fix parameters for the heat sink cooling."""

    t_ambient: float
    t_hs_max: float

@dataclasses.dataclass
class TransistorCooling:
    """Fix parameters for the transistor cooling."""

    tim_conductivity: float
    tim_thickness: float

@dataclasses.dataclass
class InductiveElementCooling:
    """Fix parameters for the inductive element cooling."""

    tim_conductivity: float
    tim_thickness: float
