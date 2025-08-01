"""Classes and toml checker for the flow control."""
# python libraries
from typing import Literal

# 3rd party libraries
from pydantic import BaseModel

# own libraries
from dct.circuit_enums import SamplingEnum

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
    inductor: Literal['no', 'pause', 'stop']
    transformer: Literal['no', 'pause', 'stop']
    heat_sink: Literal['no', 'pause', 'stop']
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

class Summary(BaseModel):
    """Flow control for the summary."""

    calculation_mode: Literal['new', 'skip']
    subdirectory: str

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
    summary: Summary
    configuration_data_files: ConfigurationDataFiles


# ######################################################
# circuit
# ######################################################

class TomlCircuitParetoDesignSpace(BaseModel):
    """Definition of the hardware design space for electronic components."""

    # DAB optimization parameters
    f_s_min_max_list: list[int]
    l_s_min_max_list: list[float]
    l_1_min_max_list: list[float]
    l_2__min_max_list: list[float]
    n_min_max_list: list[float]
    transistor_1_name_list: list[str]
    transistor_2_name_list: list[str]
    c_par_1: float
    c_par_2: float

class TomlCircuitOutputRange(BaseModel):
    """Definition of the DAB operating area."""

    v1_min_max_list: list[float]
    v2_min_max_list: list[float]
    p_min_max_list: list[float]

class TomlSampling(BaseModel):
    """Definition of the sampling method."""

    sampling_method: SamplingEnum
    sampling_points: int
    sampling_random_seed: int | Literal["random"]
    v1_additional_user_point_list: list[float]
    v2_additional_user_point_list: list[float]
    p_additional_user_point_list: list[float]
    additional_user_weighting_point_list: list[float]

class TomlCircuitFilterDistance(BaseModel):
    """Toml checker class for CircuitFilterDistance."""

    number_filtered_designs: int
    difference_percentage: float

class TomlCircuitParetoDabDesign(BaseModel):
    """Config to optimize the Dual-Active Bridge (DAB) converter."""

    design_space: TomlCircuitParetoDesignSpace
    output_range: TomlCircuitOutputRange
    sampling: TomlSampling
    filter_distance: TomlCircuitFilterDistance

# ######################################################
# inductor
# ######################################################

class TomlInductorDesignSpace(BaseModel):
    """Toml checker class for InductorDesignSpace."""

    core_name_list: list[str]
    material_name_list: list[str]
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

    permeability_datasource: str
    permeability_datatype: str
    permeability_measurement_setup: str
    permittivity_datasource: str
    permittivity_datatype: str
    permittivity_measurement_setup: str

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

    material_name_list: list[str]
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
    filter_distance: TomlTransformerFilterDistance
    settings: TomlTransformerSettings
    boundary_conditions: TomlTransformerBoundaryConditions

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
