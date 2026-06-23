"""Unit tests for heat sink optimization."""

# python libraries
import logging
from enum import Enum
import os
import tempfile

# 3rd party libraries
import pytest

# own libraries
import dct.components.heat_sink_optimization as test_circuit
import dct.toml_checker as tc
import dct.server_ctl_dtos
from dct.components.heat_sink_optimization import HeatSinkOptimization

# Enable logger
pytestlogger = logging.getLogger(__name__)


class TestCase(Enum):
    """Enum of test types."""

    # Valid test case
    LowerBoundary = 0           # Test value at lower boundary
    UpperBoundary = 1           # Test value at lower boundary
    InBetween = 2               # Test value in between
    # Failure test case
    BoundaryInconsistent = 3    # Test when minimum > maximum ( Only for minimum maximum list)
    TooLessEntries = 4          # Test when the list has got too less entries ( Only for minimum maximum list)
    TooMuchEntries = 5          # Test when the list has got too much entries ( Only for minimum maximum list)
    ExceedLowerLimit = 6        # Test when the lower limit is exceeded
    ExceedUpperLimit = 7        # Test when the lower limit is exceeded
    # Special test of additional point list: Valid test case
    APt_NumberOfEntries = 8  # Test when Number of entries in additional point list is inconsistent
    # Special test of additional point list: Valid test case
    SpecialTestNumberOfEntries = 8  # Test when Number of entries in additional point list is inconsistent
    ParameterInconsistent = 8  # triggers ValueError via invalid toml parameter


#########################################################################################################
# test of verify_optimization_parameter of circuit_optimization
#########################################################################################################

""" Information
## Transformer #####################################################################
Implemented boundary
[design_space]
    height_c_min_max_list: list[float]        -> 0<value<5
    width_b_min_max_list: list[float]         -> 0<value<5
    length_l_min_max_list: list[float]        -> 0<value<5
    height_d_min_max_list: list[float]        -> 0<value<5
    number_cooling_channels_n_min_max_list: list[int]     -> 3<=value<=100
    thickness_fin_t_min_max_list: list[float] -> 0<value<0.1
[settings]
    number_directions: int -> 2<=value<=3
    factor_pcb_area_copper_coin: float     -> 0<value<10
    factor_bottom_area_copper_coin: float  -> 0<value<10
    thermal_conductivity_copper: float     -> 80<=value<=200
[boundary_conditions]
    t_ambient: float -> -40<value<125
    t_hs_max: float  -> -40<value<125
    area_min: float  -> 0<value<25
[thermal_resistance_data]
    transistor_b1_cooling: list[float] -> [0<value<0.01 , 1<value<100]
    transistor_b2_cooling: list[float] -> [0<value<0.01 , 1<value<100]
    inductor_cooling: list[float]      -> [0<value<0.01 , 1<value<100]
    transformer_cooling: list[float]   -> [0<value<0.01 , 1<value<100]

"""

# test parameter list (counter)
@pytest.mark.parametrize("test_index, test_type", [
    # Valid test case
    # Test value at lower boundary
    (0, TestCase.LowerBoundary),
    # Test value at lower boundary
    (1, TestCase.UpperBoundary),
    # Test value in between
    (2, TestCase.InBetween),
    # Failure test case
    # Test when minimum > maximum ( Only for minimum maximum list)
    (3, TestCase.BoundaryInconsistent),
    # Test when the list has got too few entries ( Only for minimum maximum list)
    (4, TestCase.TooLessEntries),
    # Test when the list has got too many entries ( Only for minimum maximum list)
    (5, TestCase.TooMuchEntries),
    # Test when the lower limit is exceeded
    (6, TestCase.ExceedLowerLimit),
    # Test when the lower limit is exceeded
    (7, TestCase.ExceedUpperLimit)
])
# Unit test function
def test_verify_optimization_parameter(test_index: int, test_type: TestCase) -> None:
    """Test the method load_toml_file.

    :param test_index: Test index of performed test is used as start index for test lists
    :type  test_index: int
    :param test_type: Type of performed test
    :type  test_type: TestCase
    """
    # Variable declaration

    # Check if core list is not empty
    # List entries for values and list (exception *in between for values):
    # at lower boundary | at upper boundary | in between | minimum > maximum* | too few entries*
    # too many entries* | exceed the lower limit | exceed the upper limit
    int_min_max_list_configuration_ge3_lt100: list[list[int]] = (
        [[3, 3], [99, 99], [4, 8], [70, 30], [5], [6, 30, 83], [2, 20], [8, 100]])
    float_min_max_list_configuration_gt0_lt5: list[list[float]] = (
        [[1e-18, 1e-18], [4.9999, 4.9999], [0.1, 2], [4, 3.5], [1], [0.1, 2, 2.3], [0, 1.1], [1, 5]])
    float_min_max_list_configuration_gt0_lt1em1: list[list[float]] = (
        [[1e-18, 1e-18], [0.09999, 0.09999], [0.01, 0.055], [0.04, 0.035], [0.06], [0.01, 0.08, 0.083], [0, 0.03], [0.06, 0.1]])
    float_value_list_configuration_gt0_lt25: list[float] = [1e-17, 24.991, 3.4, 7.4, 3.343, 6.6, 0, 25]
    float_value_list_configuration_ge80_le200: list[float] = [80, 200, 130.78, 182.3, 133.1, 99.76, 79.7, 200.001]
    float_value_list_configuration_gt0_lt10: list[float] = [1.1e-20, 9.99, 5.5, 3.2, 0.38, 7.6, -0, 10]
    float_value_list_configuration_gem40_le125: list[float] = [-40, 125, 80.1, 105.55, -21.3, 105.5, -40.21, 275]
    int_value_list_configuration_ge2_le3 = [2, 3, 2, 3, 2, 3, 1, 4]

    # Initialize the transformer parameters
    test_heat_sink_parameter: tc.TomlHeatSink = tc.TomlHeatSink(
        design_space=dct.TomlHeatSinkDesignSpace(
            height_c_min_max_list=float_min_max_list_configuration_gt0_lt5[test_index],
            width_b_min_max_list=float_min_max_list_configuration_gt0_lt5[test_index],
            length_l_min_max_list=float_min_max_list_configuration_gt0_lt5[test_index],
            height_d_min_max_list=float_min_max_list_configuration_gt0_lt5[test_index],
            number_cooling_channels_n_min_max_list=int_min_max_list_configuration_ge3_lt100[test_index],
            thickness_fin_t_min_max_list=float_min_max_list_configuration_gt0_lt1em1[test_index]),
        settings=dct.TomlHeatSinkSettings(
            number_directions=int_value_list_configuration_ge2_le3[test_index],
            factor_pcb_area_copper_coin=float_value_list_configuration_gt0_lt10[test_index],
            factor_bottom_area_copper_coin=float_value_list_configuration_gt0_lt10[test_index],
            thermal_conductivity_copper=float_value_list_configuration_ge80_le200[test_index]),
        boundary_conditions=dct.TomlHeatSinkBoundaryConditions(
            t_ambient=float_value_list_configuration_gem40_le125[test_index],
            t_hs_max=float_value_list_configuration_gem40_le125[test_index],
            area_min=float_value_list_configuration_gt0_lt25[test_index]),
    )

    # Create boundary list from minimum-maximum list with assigned parameters
    min_max_list_name_list: list[str] = ["height_c_min_max_list", "width_b_min_max_list", "length_l_min_max_list",
                                         "height_d_min_max_list", "number_cooling_channels_n_min_max_list", "thickness_fin_t_min_max_list"]
    value_name_list: list[str] = (["number_directions", "factor_pcb_area_copper_coin", "factor_bottom_area_copper_coin",
                                   "thermal_conductivity_copper", "t_ambient", "t_hs_max", "area_min"])

    # Perform the test
    is_consistent, error_report = test_circuit.HeatSinkOptimization.verify_optimization_parameter(test_heat_sink_parameter)

    if test_type == TestCase.LowerBoundary or test_type == TestCase.UpperBoundary or test_type == TestCase.InBetween:
        # No error and empty report string
        assert error_report == ""
        assert is_consistent

    elif test_type == TestCase.ExceedUpperLimit:
        # Check if all minimum-maximum list parameters are identified
        for parameter_name in min_max_list_name_list:
            assert parameter_name in error_report

        # Check if all value_name_list parameters are identified
        for parameter_name in value_name_list:
            assert parameter_name in error_report

        # Error is indicated
        assert not is_consistent

    elif test_type == TestCase.ExceedLowerLimit:
        # Check if all minimum-maximum list parameters are identified
        for parameter_name in min_max_list_name_list:
            assert parameter_name in error_report

        # Check if all value_name_list list parameters are identified
        for parameter_name in value_name_list:
            assert parameter_name in error_report

        # Error is indicated
        assert not is_consistent

    elif test_type == TestCase.TooLessEntries:
        # Check if all minimum-maximum list parameters are identified
        for parameter_name in min_max_list_name_list:
            assert parameter_name in error_report

        # Check if not any value_name_list parameter is identified
        for parameter_name in value_name_list:
            assert parameter_name not in error_report

        # Error is indicated
        assert not is_consistent

    elif test_type == TestCase.TooMuchEntries:
        # Check if all minimum-maximum list parameters are identified
        for parameter_name in min_max_list_name_list:
            assert parameter_name in error_report

        # Check if not any value_name_list parameter is identified
        for parameter_name in value_name_list:
            assert parameter_name not in error_report

        # Error is indicated
        assert not is_consistent

    elif test_type == TestCase.BoundaryInconsistent:
        # Check if all minimum-maximum list parameters are identified
        for parameter_name in min_max_list_name_list:
            assert parameter_name in error_report

        # Check if not any value_name_list parameter is identified
        for parameter_name in value_name_list:
            assert parameter_name not in error_report

        # Error is indicated
        assert not is_consistent
################

#########################################################################################################
# test of initialize_heat_sink_optimization
#########################################################################################################

# def initialize_heat_sink_optimization(self, toml_heat_sink: dct.TomlHeatSink, toml_prog_flow: dct.FlowControl) -> bool:
# test parameter list (counter)
@pytest.mark.parametrize("test_index, test_type, is_error", [
    # Valid test case
    # Test value at lower boundary
    (0, TestCase.LowerBoundary, False),
    # Test value at upper boundary
    (1, TestCase.UpperBoundary, False),
    # Test value in between
    (2, TestCase.InBetween, False),
    # Failure test case
    # Test when the invalid parameters loaded
    (3, TestCase.ParameterInconsistent, True),
])
# Unit test function
def test_initialize_heat_sink_optimization(test_index: int, test_type: TestCase, is_error: bool) -> None:
    """Test the method initialize_circuit_optimization.

    :param test_index: Test index of the used list element
    :type  test_index: int
    :param test_type: Type of performed test
    :type  test_type: TestCase
    :param is_error: Indicates, if the function exits with error
    :type  is_error: bool
    """
    # Test value tables 
    # float min/max list: geometry dimensions (0 < val)
    float_min_max_geometry: list[list[float]] = (
        [[1e-4, 2e-4], [0.04, 0.0999], [0.05, 0.05], [-0.1, 0.05]])
 
    # int min/max list: number of cooling channels
    int_min_max_cooling_channels: list[list[int]] = (
        [[3, 8], [8, 10], [3, 6], [-1, 5]])
 
    # float value: ambient temperature (°C)
    float_t_ambient: list[float] = [-40.0, 125.0, 25.0, 25.0]
 
    # float value: t_hs_max (°C)
    float_t_hs_max: list[float] = [-40.0, 125.0, 80.0, 80.0]
 
    # float value: area_min (m²)
    float_area_min: list[float] = [1e-6, 1e-2, 1e-4, 1e-4]
 
    # int value: number_directions
    int_number_directions: list[int] = [2, 3, 2, 2]
 
    # float value: factor_pcb_area_copper_coin
    float_factor_pcb: list[float] = [0.01, 0.99, 0.5, 0.5]
 
    # float value: factor_bottom_area_copper_coin
    float_factor_bottom: list[float] = [0.01, 0.99, 0.5, 0.5]
 
    # float value: thermal_conductivity_copper
    float_thermal_conductivity: list[float] = [80.0, 200.0, 120.0, 120.0]
 
    # study and directory name suffixes — one per test_index
    hs_study_name_list: list[str] = ["hs_study_A.toml", "hs_study_B.toml", "hs_study_C.toml", "hs_study_D.toml"]
    subdirectory_list: list[str] = ["hs_sub_dir_A", "hs_sub_dir_B", "hs_sub_dir_C", "hs_sub_dir_D"]
 
    # Build Toml Heat Sink
    toml_heat_sink = tc.TomlHeatSink(
        design_space=tc.TomlHeatSinkDesignSpace(
            height_c_min_max_list=float_min_max_geometry[test_index],
            width_b_min_max_list=float_min_max_geometry[test_index],
            length_l_min_max_list=float_min_max_geometry[test_index],
            height_d_min_max_list=float_min_max_geometry[test_index],
            number_cooling_channels_n_min_max_list=int_min_max_cooling_channels[test_index],
            thickness_fin_t_min_max_list=float_min_max_geometry[test_index],
        ),
        settings=tc.TomlHeatSinkSettings(
            number_directions=int_number_directions[test_index],
            factor_pcb_area_copper_coin=float_factor_pcb[test_index],
            factor_bottom_area_copper_coin=float_factor_bottom[test_index],
            thermal_conductivity_copper=float_thermal_conductivity[test_index],
        ),
        boundary_conditions=tc.TomlHeatSinkBoundaryConditions(
            t_ambient=float_t_ambient[test_index],
            t_hs_max=float_t_hs_max[test_index],
            area_min=float_area_min[test_index],
        ),

    )

    # Build FlowControl (toml_prog_flow)

    with tempfile.TemporaryDirectory() as tmpdir:

        toml_prog_flow = tc.FlowControl(
            general=tc.General(
                project_directory=tmpdir,
                topology="dab",
            ),
            breakpoints=tc.Breakpoints(
                circuit_pareto="no",
                circuit_filtered="no",
                capacitor="no",
                inductor="no",
                transformer="no",
                heat_sink="no",
                pre_summary="no",
                summary="no",
            ),
            conditional_breakpoints=tc.CondBreakpoints(
                circuit=0,
                inductor=0,
                transformer=0,
                heat_sink=0,
            ),
            circuit=tc.Circuit(
                number_of_trials=100,
                calculation_mode="new",
                subdirectory="circuit",
            ),
            capacitor=tc.Capacitor(
                calculation_modes=["new"],
                subdirectory="capacitor",
            ),
            inductor=tc.Inductor(
                numbers_of_trials=[100],
                calculation_modes=["new"],
                subdirectory="inductor",
            ),
            transformer=tc.Transformer(
                numbers_of_trials=[100],
                calculation_modes=["new"],
                subdirectory="transformer",
            ),
            heat_sink=tc.HeatSink(
                number_of_trials=100,
                calculation_mode="new",
                subdirectory=subdirectory_list[test_index],
            ),
            pre_summary=tc.PreSummary(
                calculation_mode="new",
                subdirectory="pre_summary",
            ),
            summary=tc.Summary(
                subdirectory="summary",
            ),
            configuration_data_files=tc.ConfigurationDataFiles(
                topology_files=["topology.toml"],
                capacitor_configuration_files=["capacitor.toml"],
                inductor_configuration_files=["inductor.toml"],
                transformer_configuration_files=["transformer.toml"],
                heat_sink_configuration_file=hs_study_name_list[test_index],
            ),
        )

        # Create the object under test
        test_object: HeatSinkOptimization = HeatSinkOptimization()

        if not is_error:
            is_initialized = test_object.initialize_heat_sink_optimization(toml_heat_sink, toml_prog_flow)
            assert is_initialized 
            assert test_object._hct_config is not None

            expected_study_name = hs_study_name_list[test_index].replace(".toml", "")
            expected_optimization_directory = os.path.join(
                tmpdir,
                subdirectory_list[test_index],
                hs_study_name_list[test_index].replace(".toml", "")
            )

            # Verify general fields
            assert test_object._hct_config.heat_sink_study_name == expected_study_name
            assert test_object._hct_config.heat_sink_optimization_directory == expected_optimization_directory

            # Verify design space fields
            assert test_object._hct_config.height_c_min_max_list == toml_heat_sink.design_space.height_c_min_max_list
            assert test_object._hct_config.width_b_min_max_list == toml_heat_sink.design_space.width_b_min_max_list
            assert test_object._hct_config.length_l_min_max_list == toml_heat_sink.design_space.length_l_min_max_list
            assert test_object._hct_config.height_d_min_max_list == toml_heat_sink.design_space.height_d_min_max_list
            assert test_object._hct_config.number_cooling_channels_n_min_max_list == toml_heat_sink.design_space.number_cooling_channels_n_min_max_list
            assert test_object._hct_config.thickness_fin_t_min_max_list == toml_heat_sink.design_space.thickness_fin_t_min_max_list

            # Verify boundary conditions
            assert test_object._hct_config.t_ambient == toml_heat_sink.boundary_conditions.t_ambient
            assert test_object._hct_config.area_min == toml_heat_sink.boundary_conditions.area_min

            # Verify settings
            assert test_object._hct_config.number_directions == toml_heat_sink.settings.number_directions
 
            # Verify fan list
            assert isinstance(test_object._hct_config.fan_list, list)
        else:
            with pytest.raises(ValueError) as error_message:
                test_object.initialize_heat_sink_optimization(toml_heat_sink, toml_prog_flow)
 
            assert "Heat sink optimization parameter are inconsistent" in str(error_message.value)
