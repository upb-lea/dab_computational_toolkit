"""Enumerations for the circuit optimization."""
import enum


class SamplingEnum(enum.Enum):
    """Enum for the different sampling classes."""

    meshgrid = "meshgrid"
    latin_hypercube = "latin_hypercube"
    dessca = "dessca"


class CalcModeEnum(enum.Enum):
    """Enum for calculation mode."""

    # Calculate new
    new_mode = "new"
    # Continue calculation
    continue_mode = "continue"
    # skip calculation and use actual data
    skip_mode = "skip"
