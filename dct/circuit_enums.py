"""Enumerations for the circuit optimization."""
import enum


class SamplingEnum(enum.Enum):
    """Enum for the different sampling classes."""

    meshgrid = "meshgrid"
    latin_hypercube = "latin_hypercube"

class TopologyEnum(enum.Enum):
    """Enum for the different topologies."""
    DAB = "DAB"
    SBC = "SBC"
