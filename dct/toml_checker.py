"""Classes and toml checker for the flow control."""
# python libraries
from typing import Literal

# 3rd party libraries
from pydantic import BaseModel

# own libraries
from materialdatabase.meta.data_enums import Material, DataSource


# ######################################################
# debug
# ######################################################

class DebugGeneral(BaseModel):
    """Debug mode general information."""

    is_debug: bool

class DebugCapacitor(BaseModel):
    """Debug information for the capacitor."""

    number_working_point_max: int

class DebugInductor(BaseModel):
    """Debug information for the inductor."""

    number_reluctance_working_point_max: int
    number_fem_working_point_max: int

class DebugTransformer(BaseModel):
    """Debug information for the transformer."""

    number_reluctance_working_point_max: int
    number_fem_working_point_max: int

class Debug(BaseModel):
    """General information in debug configuration."""

    general: DebugGeneral
    capacitor_1: DebugCapacitor
    capacitor_2: DebugCapacitor
    inductor: DebugInductor
    transformer: DebugTransformer

# ######################################################
# flow control
# ######################################################

class General(BaseModel):
    """General flow control information."""

    project_directory: str

class Breakpoints(BaseModel):
    """Flow control breakpoints."""

    circuit_pareto: Literal['no', 'pause', 'stop']
    circuit_filtered: Literal['no', 'pause', 'stop']
    capacitor_1: Literal['no', 'pause', 'stop']
    capacitor_2: Literal['no', 'pause', 'stop']
    inductor: Literal['no', 'pause', 'stop']
    transformer: Literal['no', 'pause', 'stop']
    heat_sink: Literal['no', 'pause', 'stop']
    pre_summary: Literal['no', 'pause', 'stop']
    summary: Literal['no', 'pause', 'stop']

class CondBreakpoints(BaseModel):
    """Flow control conditional breakpoints."""

    circuit: int
    inductor: int
    transformer: int
    heat_sink: int

class Circuit(BaseModel):
    """Flow control for the circuit."""

    number_of_trials: int
    calculation_mode: Literal['new', 'continue', 'skip']
    subdirectory: str

class Capacitor1(BaseModel):
    """Flow control for the capacitor 1."""

    calculation_mode: Literal['new', 'skip']
    subdirectory: str

class Capacitor2(BaseModel):
    """Flow control for the capacitor 2."""

    calculation_mode: Literal['new', 'skip']
    subdirectory: str

class Inductor(BaseModel):
    """Flow control for the inductor."""

    number_of_trials: int
    calculation_mode: Literal['new', 'continue', 'skip']
    subdirectory: str

class Transformer(BaseModel):
    """Flow control for the transformer."""

    number_of_trials: int
    calculation_mode: Literal['new', 'continue', 'skip']
    subdirectory: str

class HeatSink(BaseModel):
    """Flow control for the heat sink."""

    number_of_trials: int
    calculation_mode: Literal['new', 'continue', 'skip']
    subdirectory: str

class PreSummary(BaseModel):
    """Flow control for the pre-summary."""

    calculation_mode: Literal['new', 'skip']
    subdirectory: str

class Summary(BaseModel):
    """Flow control for the summary."""

    calculation_mode: Literal['new', 'skip']
    subdirectory: str

class ConfigurationDataFiles(BaseModel):
    """File paths to the configuration files."""

    general_configuration_file: str
    circuit_configuration_file: str
    capacitor_1_configuration_file: str
    capacitor_2_configuration_file: str
    inductor_configuration_file: str
    transformer_configuration_file: str
    heat_sink_configuration_file: str

class FlowControl(BaseModel):
    """General flow control class."""

    general: General
    breakpoints: Breakpoints
    conditional_breakpoints: CondBreakpoints
    circuit: Circuit
    capacitor_1: Capacitor1
    capacitor_2: Capacitor2
    inductor: Inductor
    transformer: Transformer
    heat_sink: HeatSink
    pre_summary: PreSummary
    summary: Summary
    configuration_data_files: ConfigurationDataFiles


# ######################################################
# capacitor 1 and capacitor 2
# ######################################################

class TomlCapacitorSelection(BaseModel):
    """Capacitor selection details."""

    maximum_peak_to_peak_voltage_ripple: float
    temperature_ambient: float
    voltage_safety_margin_percentage: float
    maximum_number_series_capacitors: int
    lifetime_h: float

# ######################################################
# inductor
# ######################################################

class TomlInductorDesignSpace(BaseModel):
    """Toml checker class for InductorDesignSpace."""

    core_name_list: list[str]
    material_name_list: list[Material]
    litz_wire_name_list: list[str]
    core_inner_diameter_min_max_list: list[float]
    window_h_min_max_list: list[float]
    window_w_min_max_list: list[float]

class TomlInductorInsulation(BaseModel):
    """Toml checker class for InductorInsulation."""

    primary_to_primary: float
    core_bot: float
    core_top: float
    core_right: float
    core_left: float

class TomlMaterialDataSources(BaseModel):
    """Toml checker class for MaterialDataSources."""

    permeability_datasource: DataSource
    permittivity_datasource: DataSource

class TomlInductorBoundaryConditions(BaseModel):
    """Toml checker class for InductorBoundaryConditions."""

    temperature: float


class TomlFilterDistance(BaseModel):
    """Toml checker class for FilterDistance."""

    factor_dc_losses_min_max_list: list[float]

class TomlInductor(BaseModel):
    """Toml checker class for Inductor."""

    design_space: TomlInductorDesignSpace
    insulations: TomlInductorInsulation
    boundary_conditions: TomlInductorBoundaryConditions
    filter_distance: TomlFilterDistance
    material_data_sources: TomlMaterialDataSources

# ######################################################
# transformer
# ######################################################
class TomlTransformerDesignSpace(BaseModel):
    """Toml checker class for TransformerDesignSpace."""

    material_name_list: list[Material]
    core_name_list: list[str]
    core_inner_diameter_min_max_list: list[float]
    window_w_min_max_list: list[float]
    window_h_bot_min_max_list: list[float]
    primary_litz_wire_list: list[str]
    secondary_litz_wire_list: list[str]
    n_p_top_min_max_list: list[int]
    n_p_bot_min_max_list: list[int]

class TomlTransformerSettings(BaseModel):
    """Toml checker class for TransformerSettings."""

    fft_filter_value_factor: float
    mesh_accuracy: float

class TomlTransformerBoundaryConditions(BaseModel):
    """Toml checker class for TransformerBoundaryConditions."""

    max_transformer_total_height: float
    max_core_volume: float
    temperature: float

class TomlTransformerInsulation(BaseModel):
    """Toml checker class for TransformerInsulation."""

    # insulation for top core window
    iso_window_top_core_top: float
    iso_window_top_core_bot: float
    iso_window_top_core_left: float
    iso_window_top_core_right: float
    # insulation for bottom core window
    iso_window_bot_core_top: float
    iso_window_bot_core_bot: float
    iso_window_bot_core_left: float
    iso_window_bot_core_right: float
    # winding-to-winding insulation
    iso_primary_to_primary: float
    iso_secondary_to_secondary: float
    iso_primary_to_secondary: float

class TomlTransformerFilterDistance(BaseModel):
    """Toml checker class for TransformerFilterDistance."""

    factor_dc_losses_min_max_list: list[float]

class TomlTransformer(BaseModel):
    """Toml checker class for Transformer."""

    design_space: TomlTransformerDesignSpace
    insulation: TomlTransformerInsulation
    boundary_conditions: TomlTransformerBoundaryConditions
    filter_distance: TomlTransformerFilterDistance
    settings: TomlTransformerSettings
    material_data_sources: TomlMaterialDataSources

# ######################################################
# heat sink inclusive data of summary calculation
# ######################################################

class TomlHeatSinkBoundaryConditions(BaseModel):
    """Toml checker for HeatSinkBoundaryConditions."""

    t_ambient: float
    t_hs_max: float
    area_min: float

class TomlHeatSinkSettings(BaseModel):
    """Toml checker for HeatSinkSettings."""

    number_directions: int
    factor_pcb_area_copper_coin: float
    factor_bottom_area_copper_coin: float
    thermal_conductivity_copper: float

class TomlHeatSinkDesignSpace(BaseModel):
    """Toml checker for HeatSinkDesignSpace."""

    height_c_min_max_list: list[float]
    width_b_min_max_list: list[float]
    length_l_min_max_list: list[float]
    height_d_min_max_list: list[float]
    number_fins_n_min_max_list: list[int]
    thickness_fin_t_min_max_list: list[float]

class TomlHeatSinkThermalResistanceData(BaseModel):
    """Toml checker for HeatSinkThermalResistanceData."""

    # [tim_thickness, tim_conductivity]
    transistor_b1_cooling: list[float]
    transistor_b2_cooling: list[float]
    inductor_cooling: list[float]
    transformer_cooling: list[float]

class TomlHeatSink(BaseModel):
    """Toml checker for HeatSink."""

    design_space: TomlHeatSinkDesignSpace
    settings: TomlHeatSinkSettings
    boundary_conditions: TomlHeatSinkBoundaryConditions
    thermal_resistance_data: TomlHeatSinkThermalResistanceData
