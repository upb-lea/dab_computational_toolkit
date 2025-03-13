"""Classes and toml checker for the flow control."""
from pydantic import BaseModel
from typing import Literal

# flow control

class General(BaseModel):
    """General flow control information."""

    project_directory: str
    study_name: str
    relative_flag: int

class Breakpoints(BaseModel):
    """Flow control breakpoints."""

    circuit_pareto: Literal['no', 'pause', 'stop']
    circuit_filtered: Literal['no', 'pause', 'stop']
    inductor: Literal['no', 'pause', 'stop']
    transformer: Literal['no', 'pause', 'stop']
    heat_sink: Literal['no', 'pause', 'stop']

class CondBreakpoints(BaseModel):
    """Flow control conditional breakpoints."""

    circuit: int
    inductor: int
    transformer: int
    heat_sink: int

class Circuit(BaseModel):
    """Flow control for the circuit."""

    number_of_trials: int
    re_calculation: Literal['new', 'continue', 'skip']
    subdirectory: str

class Inductor(BaseModel):
    """Flow control for the inductor."""

    number_of_trials: int
    re_calculation: Literal['new', 'continue', 'skip']
    subdirectory: str

class Transformer(BaseModel):
    """Flow control for the transformer."""

    number_of_trials: int
    re_calculation: Literal['new', 'continue', 'skip']
    subdirectory: str

class HeatSink(BaseModel):
    """Flow control for the heat sink."""

    number_of_trials: int
    re_calculation: Literal['new', 'continue', 'skip']
    subdirectory: str
    circuit_study_name_flag: bool

class ConfigurationDataFiles(BaseModel):
    """File paths to the configuration files."""

    circuit_configuration_file: str
    inductor_configuration_file: str
    transformer_configuration_file: str
    heat_sink_configuration_file: str

class FlowControl(BaseModel):
    """General flow control class."""

    general: General
    breakpoints: Breakpoints
    conditional_breakpoints: CondBreakpoints
    circuit: Circuit
    inductor: Inductor
    transformer: Transformer
    heat_sink: HeatSink
    configuration_data_files: ConfigurationDataFiles

# circuit
