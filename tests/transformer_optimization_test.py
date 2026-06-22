"""Unit tests for transformer optimization."""

# python libraries
import logging
import copy
from enum import Enum
import os
import tempfile

# 3rd party libraries
import pytest

# own libraries
import dct.components.transformer_optimization as test_circuit
import dct.toml_checker as tc
import dct.server_ctl_dtos
import femmt as fmt
from dct.circuit_enums import CalcModeEnum
from dct.components.transformer_optimization_dtos import TransformerOptimizationDto
from dct.server_ctl_dtos import ProgressStatus
from dct.datasets_dtos import TransformerConfiguration, StudyData
from dct.components.component_dtos import TransformerRequirements
from dct.components.transformer_optimization import TransformerOptimization


# Enable logger
pytestlogger = logging.getLogger(__name__)

# Number of transformers used in all test cases (minimum required: 3)
NUMBER_OF_TRANSFORMERS = 3

class TestCase(Enum):
    """Enum of test types."""

    # Valid test case
    LowerBoundary = 0           # Test value at lower boundary
    UpperBoundary = 1           # Test value at lower boundary
    InBetween = 2               # Test value in between
    # Failure test case
    BoundaryInconsistent = 3    # Test when minimum > maximum ( Only for minimum maximum list)
    TooLessEntries = 4          # Test when the list has got too few entries ( Only for minimum maximum list)
    TooMuchEntries = 5          # Test when the list has got too many entries ( Only for minimum maximum list)
    ExceedLowerLimit = 6        # Test when the lower limit is exceeded
    ExceedUpperLimit = 7        # Test when the lower limit is exceeded
    # Special test of additional point list: Valid test case
    APt_NumberOfEntries = 8  # Test when Number of entries in additional point list is inconsistent
    # Special test of additional point list: Valid test case
    SpecialTestNumberOfEntries = 8  # Test when Number of entries in additional point list is inconsistent
    # Data are not initialized
    DataNotInitialized = 8  # Test when Number of entries in additional point list is inconsistent


#########################################################################################################
# test of verify_optimization_parameter of circuit_optimization
#########################################################################################################

""" Information
## Transformer #####################################################################
Implemented boundary
[design_space]
core_name_list: list[str] -> All entries are key names of the core database
material_name_list: list[str] -> All entries are key names of the material database (actual ignored)
litz_wire_name_list: list[str] -> All entries are key names of the litz database
core_inner_diameter_min_max_list: list[float] 0<val<5
window_h_min_max_list: list[float] -> 0<val<5
window_w_min_max_list: list[float] -> 0<val<5
[insulations]
primary_to_primary: float -> 0<val<0.1
core_bot: float           -> 0<val<0.1
core_top: float           -> 0<val<0.1
core_right: float         -> 0<val<0.1
core_left: float          -> 0<val<0.1
[boundary_conditions]
temperature: float        -> -40<=val<=175 °C
[filter_distance]
factor_dc_losses_min_max_list: list[float] -> 0<=val<=100
[material_data_sources] -(Future implementation, actually not implemented) 
permeability_datasource: str
permeability_datatype: str
permeability_measurement_setup: str
permittivity_datasource: str
permittivity_datatype: str
permittivity_measurement_setup: str
"""

@pytest.fixture(scope="module")
def get_name_lists() -> tuple[list[str], list[str]]:
    """Get core names from core database and litz names from litz database.

    This function reads the key (names of components) from the core and litz databases.

    :return: List of transistor names
    :rtype: tuple[list[str], list[str]]
    """
    # Read the core and litz database
    core_database: dict = fmt.core_database()
    litz_database: dict = fmt.litz_database()
    core_keyword_list: list[str] = list(core_database.keys())
    litz_keyword_list: list[str] = list(litz_database.keys())

    return core_keyword_list, litz_keyword_list


# test parameter list (counter)
@pytest.mark.parametrize("test_index, test_type, is_core_list_available", [
    # Valid test case
    # Test value at lower boundary
    (0, TestCase.LowerBoundary, True),
    # Test value at lower boundary
    (1, TestCase.UpperBoundary, True),
    # Test value in between
    (2, TestCase.InBetween, True),
    # Failure test case
    # Test when minimum > maximum ( Only for minimum maximum list)
    (3, TestCase.BoundaryInconsistent, True),
    # Test when the list has got too few entries ( Only for minimum maximum list)
    (4, TestCase.TooLessEntries, True),
    # Test when the list has got too many entries ( Only for minimum maximum list)
    (5, TestCase.TooMuchEntries, True),
    # Test when the lower limit is exceeded
    (6, TestCase.ExceedLowerLimit, True),
    # Test when the lower limit is exceeded
    (7, TestCase.ExceedUpperLimit, True),
    # Test value at lower boundary
    (0, TestCase.LowerBoundary, False),
    # Test value at lower boundary
    (1, TestCase.UpperBoundary, False),
    # Test value in between
    (2, TestCase.InBetween, False),
    # Failure test case
    # Test when minimum > maximum ( Only for minimum maximum list)
    (3, TestCase.BoundaryInconsistent, False),
    # Test when the list has got too few entries ( Only for minimum maximum list)
    (4, TestCase.TooLessEntries, False),
    # Test when the list has got too many entries ( Only for minimum maximum list)
    (5, TestCase.TooMuchEntries, False),
    # Test when the lower limit is exceeded
    (6, TestCase.ExceedLowerLimit, False),
    # Test when the lower limit is exceeded
    (7, TestCase.ExceedUpperLimit, False)
])
# Unit test function
def test_verify_optimization_parameter(get_name_lists: tuple[list[str], list[str]], test_index: int, test_type: TestCase,
                                       is_core_list_available: bool) -> None:
    """Test the method verify_optimization_parameter.

    :param get_name_lists: List of core names list and litz names list
    :type  get_name_lists: tuple[list[str]]
    :param test_index: Test index of performed test is used as start index for test lists
    :type  test_index: int
    :param test_type: Type of performed test
    :type  test_type: TestCase
    :param is_core_list_available: Indicates if the core list is available (true) or if dimension data are to use (false)
    :type  is_core_list_available: bool
    """
    # Variable declaration
    # Called only on time while parametric test
    core_list, litz_list = get_name_lists

    # Check if core list is not empty
    if is_core_list_available:
        core_list_len = len(core_list)
        assert core_list_len > 0
        # Generate the core list entries
        one_core_name = core_list[core_list_len // 2]
        core_sub_list = copy.deepcopy(core_list)
        core_sub_list = core_sub_list[:len(core_sub_list) // 2]
        core_first_name_wrong_list = copy.deepcopy(core_list)
        core_first_name_wrong_list[0] = one_core_name + "first_name_wrong"
        core_last_name_wrong_list = copy.deepcopy(core_list)
        core_last_name_wrong_list[core_list_len - 1] = one_core_name + "last_name_wrong"
        # List entries for the list:
        # All names | one name | half of the list | empty list
        # one wrong name | first name wrong | last name wrong
        # Check if core list is not empty
        core_name_list_configuration: list[list[str]] = (
            [core_list, [one_core_name], core_sub_list, ["1", "2", "3"], [], [one_core_name + "wrong"],
             core_first_name_wrong_list, core_last_name_wrong_list])
    else:
        # Initialize an empty list
        core_name_list_configuration = [[], [], [], [], [], [], [], []]
    # Assemble litz test parameter list
    litz_list_len = len(litz_list)
    assert litz_list_len > 0
    # Generate the litz list entries
    one_litz_name = litz_list[litz_list_len//2]
    litz_sub_list = copy.deepcopy(litz_list)
    litz_sub_list = litz_sub_list[:len(litz_sub_list)//2]
    litz_first_name_wrong_list = copy.deepcopy(litz_list)
    litz_first_name_wrong_list[0] = one_litz_name+"first_name_wrong"
    litz_last_name_wrong_list = copy.deepcopy(litz_list)
    litz_last_name_wrong_list[litz_list_len-1] = one_litz_name+"last_name_wrong"
    # List entries for the list:
    # All names | one name | half of the list | empty list
    # one wrong name | first name wrong | last name wrong
    litz_name_list_configuration: list[list[str]] = (
        [litz_list, [one_litz_name], litz_sub_list, ["1", "2", "3"], [], [one_litz_name+"wrong"],
         litz_first_name_wrong_list, litz_last_name_wrong_list])

    # List entries for values and list (exception *in between for values):
    # at lower boundary | at upper boundary | in between | minimum > maximum* | too few entries*
    # too many entries* | exceed the lower limit | exceed the upper limit
    int_min_max_list_configuration_gt0_le10000: list[list[int]] = (
        [[1, 1], [10000, 10000], [1000, 6300], [7000, 300], [500], [600, 3000, 8300], [-1, 7000], [2, 10005]])
    float_min_max_list_configuration_gt0_lt5: list[list[float]] = (
        [[1e-18, 1e-18], [4.9999, 4.9999], [0.1, 2], [4, 3.5], [1], [0.1, 2, 2.3], [0, 1.1], [1, 5]])
    float_min_max_list_configuration_ge0_le100: list[list[float]] = (
        [[0, 0], [100, 100], [34, 77], [90, 67], [33], [33, 66, 99], [-1.0e-18, 88], [55, 100.1]])
    float_value_list_configuration_gt0_lt1: list[float] = [1e-17, 0.9991, 0.34, 0.74, 0.33, 0.66, -0.1, 1.0]
    float_value_list_configuration_ge0_le1: list[float] = [0, 1, 0.44, 0.84, 0.38, 0.76, -0.1, 1.01]
    float_value_gem40_le175: list[float] = [-40, 175, 80.1, 105.55, -21.3, 125.5, -40.21, 275]
    float_value_gt0_lt1em1: list[float] = [1e-17, 0.0991, 0.034, 0.074, 0.033, 0.066, 0, 0.1]
    float_value_list_configuration_gt0_le1em2xgt1_le100: list[list[float]] = (
        [[1e-17, 1.1], [0.01, 100], [0.0034, 73.78], [-1, -282.3], [0.0033], [0.0066, 22.76, 33], [0, 10], [0.011, 100.01]])
    float_value_gt0_lt5: list[float] = [1e-18, 4.9999, 2, 3.5, 1, 0.1, -0.11, 5]

    # Check if core list is not empty
    if is_core_list_available:
        # Create an empty list for the core parameter values
        design_space_min_max_entry = []
    else:
        # Create an empty list for the core parameter values
        design_space_min_max_entry = float_min_max_list_configuration_gt0_lt5[test_index]

    # Initialize the transformer parameters
    test_transformer_parameter: tc.TomlTransformer = tc.TomlTransformer(
        design_space=tc.TomlTransformerDesignSpace(
            material_name_list=["3C95"],
            core_name_list=core_name_list_configuration[test_index],
            core_inner_diameter_min_max_list=design_space_min_max_entry,
            window_w_min_max_list=design_space_min_max_entry,
            window_h_bot_min_max_list=design_space_min_max_entry,
            primary_litz_wire_list=litz_name_list_configuration[test_index],
            secondary_litz_wire_list=litz_name_list_configuration[test_index],
            n_p_top_min_max_list=int_min_max_list_configuration_gt0_le10000[test_index],
            n_p_bot_min_max_list=int_min_max_list_configuration_gt0_le10000[test_index]),
        insulation=tc.TomlTransformerInsulation(
            # insulation for top core window
            iso_window_top_core_top=float_value_gt0_lt1em1[test_index],
            iso_window_top_core_bot=float_value_gt0_lt1em1[test_index],
            iso_window_top_core_left=float_value_gt0_lt1em1[test_index],
            iso_window_top_core_right=float_value_gt0_lt1em1[test_index],
            # insulation for bottom core window
            iso_window_bot_core_top=float_value_gt0_lt1em1[test_index],
            iso_window_bot_core_bot=float_value_gt0_lt1em1[test_index],
            iso_window_bot_core_left=float_value_gt0_lt1em1[test_index],
            iso_window_bot_core_right=float_value_gt0_lt1em1[test_index],
            # winding-to-winding insulation
            iso_primary_to_primary=float_value_gt0_lt1em1[test_index],
            iso_secondary_to_secondary=float_value_gt0_lt1em1[test_index],
            iso_primary_to_secondary=float_value_gt0_lt1em1[test_index]),
        thermal_data=tc.TomlThermalData(thermal_cooling=float_value_list_configuration_gt0_le1em2xgt1_le100[test_index]),
        boundary_conditions=tc.TomlTransformerBoundaryConditions(
            max_transformer_total_height=float_value_gt0_lt5[test_index],
            max_core_volume=float_value_gt0_lt5[test_index],
            temperature=float_value_gem40_le175[test_index]),
        filter_distance=dct.TomlTransformerFilterDistance(
            factor_dc_losses_min_max_list=float_min_max_list_configuration_ge0_le100[test_index]),
        settings=dct.TomlTransformerSettings(
            fft_filter_value_factor=float_value_list_configuration_ge0_le1[test_index],
            mesh_accuracy=float_value_list_configuration_gt0_lt1[test_index]),
        material_data_sources=tc.TomlMaterialDataSources(
            permeability_datasource="LEA_MTB",
            permittivity_datasource="LEA_MTB")
    )

    # Create boundary list from minimum-maximum list with assigned parameters
    min_max_list_name_list_w_o_core_list: list[str] = (
        ["core_inner_diameter_min_max_list", "window_w_min_max_list", "window_h_bot_min_max_list",
         "factor_dc_losses_min_max_list", "n_p_top_min_max_list", "n_p_bot_min_max_list"])
    min_max_list_name_list_w_core_list: list[str] = (
        ["factor_dc_losses_min_max_list", "n_p_top_min_max_list", "n_p_bot_min_max_list", "thermal_cooling"])
    value_name_list: list[str] = (
        ["iso_window_top_core_top", "iso_window_top_core_bot", "iso_window_top_core_left", "iso_window_top_core_right",
         "iso_window_bot_core_top", "iso_window_bot_core_bot", "iso_window_bot_core_left", "iso_window_bot_core_right",
         "iso_primary_to_primary", "iso_secondary_to_secondary", "iso_primary_to_secondary", "temperature",
         "max_transformer_total_height", "max_core_volume", "fft_filter_value_factor", "mesh_accuracy"])

    # Perform the test
    is_consistent, error_report = test_circuit.TransformerOptimization.verify_optimization_parameter(test_transformer_parameter)
    # Prepare verification list depending on available core list
    if is_core_list_available:
        min_max_list_name_list = min_max_list_name_list_w_core_list
    else:
        min_max_list_name_list = min_max_list_name_list_w_o_core_list

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
# test of initialize_transformer_optimization_list
#########################################################################################################

# initialize_transformer_optimization_list(self, configuration_data_list: list[TransformerConfiguration],
#                                           transformer_requirements_list: list[TransformerRequirements]) -> None:

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
    # Test when the transformer_toml_data is None -> ValueError raised
    (3, TestCase.DataNotInitialized, True),
])
# Unit test function
def test_initialize_transformer_optimization_list(test_index: int, test_type: TestCase, is_error: bool) -> None:
    """Test the method initialize_transformer_optimization_list.
    :param test_index: Test index of the used list element
    :type  test_index: int
    :param test_type: Type of performed test
    :type  test_type: TestCase
    :param is_error: Indicates, if the function exits with error
    :type  is_error: bool
    """
    # Create dummy Test value lists - 4 entries, one per test_index
    # Pattern: lower boundary | upper boundary | in between | error case

    # float min/max list: 0 < val < 5 (core dimensions)
    float_min_max_gt0_lt5: list[list[float]] = [
        [1e-4, 2e-4],     # lower boundary
        [4.0, 4.9],       # upper boundary
        [0.5, 2.5],       # in between
        [0.5, 2.5],       # error case (toml_data=None; these values are never reached)
    ]

    # int min/max list: turns ratio
    int_min_max_n_p: list[list[int]] = [
        [1, 2],         # lower boundary
        [8, 10],        # upper boundary
        [3, 6],         # in between
        [3, 6],         # error case
    ]

    # float min/max list: 0 < val <= 100  (factor_dc_losses)
    float_min_max_gt0_le100: list[list[float]] = [
        [1e-4, 1e-4],     # lower boundary
        [99.0, 100.0],    # upper boundary
        [34.0, 77.0],     # in between
        [34.0, 77.0],     # error case
    ]

    # float value: 0 < val < 0.1  (insulation values)
    float_value_gt0_lt0_1: list[float] = [
        1e-5,   # lower boundary
        0.09,   # upper boundary
        0.03,   # in between
        0.03,   # error case
    ]
 
    # float value: -40 <= val <= 175  (temperature)
    float_value_gem40_le175: list[float] = [
        -40.0,   # lower boundary
        175.0,   # upper boundary
        80.0,    # in between
        80.0,    # error case
    ]
 
    # float value: max transformer total height (> 0)
    float_max_height: list[float] = [
        0.01,   # lower boundary
        0.5,    # upper boundary
        0.15,   # in between
        0.15,   # error case
    ]
 
    # float value: max core volume (> 0)
    float_max_core_volume: list[float] = [
        1e-6,   # lower boundary
        1e-3,   # upper boundary
        5e-5,   # in between
        5e-5,   # error case
    ]

    # thermal_cooling: [tim_thickness (0 < val <= 0.01), tim_conductivity (0 < val <= 100)]
    float_thermal_cooling: list[list[float]] = [
        [1e-5, 1.0],      # lower boundary
        [0.01, 100.0],    # upper boundary
        [0.005, 50.0],    # in between
        [0.005, 50.0],    # error case
    ]
    
    # fft_filter_value_factor and mesh_accuracy (float > 0)
    float_fft_filter: list[float] = [1e-4, 0.9, 0.5, 0.5]
    float_mesh_accuracy: list[float] = [0.1, 0.9, 0.5, 0.5]

    # number_of_trials (int > 0)
    int_number_of_trials: list[int] = [1, 10000, 500, 500]

    # transformer target parameters
    float_l_s12_target: list[float] = [1e-9,  1e-3,  5e-6,  5e-6]
    float_l_h_target: list[float]   = [1e-6,  1e-2,  5e-4,  5e-4]
    float_n_target: list[float]     = [0.5,   10.0,  3.0,   3.0]
 
    # Fixed time / current waveforms (same for all test indices)
    time_vec: list[float]   = [0.0, 5e-6, 1e-5]
    current_1_vec: list[float] = [0.0, 5.0,  0.0]
    current_2_vec: list[float] = [0.0, -2.5, 0.0]
 
    # Study and circuit name suffixes — one per test_index
    study_name_prefix: list[str]  = ["trf_study_A", "trf_study_B", "trf_study_C", "trf_study_D"]
    circuit_id_prefix: list[str]  = ["trf_circuit_0", "trf_circuit_1", "trf_circuit_2", "trf_circuit_3"]


    # Build TomlTransformer
    if not is_error:
        toml_data: tc.TomlTransformer | None = tc.TomlTransformer(
            design_space=tc.TomlTransformerDesignSpace(
                material_name_list=["3C95"],
                core_name_list=[],
                core_inner_diameter_min_max_list=float_min_max_gt0_lt5[test_index],
                window_w_min_max_list=float_min_max_gt0_lt5[test_index],
                window_h_bot_min_max_list=float_min_max_gt0_lt5[test_index],
                primary_litz_wire_list=[],
                secondary_litz_wire_list=[],
                n_p_top_min_max_list=int_min_max_n_p[test_index],
                n_p_bot_min_max_list=int_min_max_n_p[test_index],
            ),
            insulation=tc.TomlTransformerInsulation(
                iso_window_top_core_top=float_value_gt0_lt0_1[test_index],
                iso_window_top_core_bot=float_value_gt0_lt0_1[test_index],
                iso_window_top_core_left=float_value_gt0_lt0_1[test_index],
                iso_window_top_core_right=float_value_gt0_lt0_1[test_index],
                iso_window_bot_core_top=float_value_gt0_lt0_1[test_index],
                iso_window_bot_core_bot=float_value_gt0_lt0_1[test_index],
                iso_window_bot_core_left=float_value_gt0_lt0_1[test_index],
                iso_window_bot_core_right=float_value_gt0_lt0_1[test_index],
                iso_primary_to_primary=float_value_gt0_lt0_1[test_index],
                iso_secondary_to_secondary=float_value_gt0_lt0_1[test_index],
                iso_primary_to_secondary=float_value_gt0_lt0_1[test_index],
            ),
            thermal_data=tc.TomlThermalData(
                thermal_cooling=float_thermal_cooling[test_index],
            ),
            boundary_conditions=tc.TomlTransformerBoundaryConditions(
                max_transformer_total_height=float_max_height[test_index],
                max_core_volume=float_max_core_volume[test_index],
                temperature=float_value_gem40_le175[test_index],
            ),
            filter_distance=dct.TomlTransformerFilterDistance(
                factor_dc_losses_min_max_list=float_min_max_gt0_le100[test_index],
            ),
            material_data_sources=tc.TomlMaterialDataSources(
                permeability_datasource="LEA_MTB",
                permittivity_datasource="LEA_MTB",
            ),
            settings=tc.TomlTransformerSettings(
                fft_filter_value_factor=float_fft_filter[test_index],
                mesh_accuracy=float_mesh_accuracy[test_index],
            ),
        )
    else:
        toml_data = None
    
    # Build configuration_data_list (minimum 3 elements)
    with tempfile.TemporaryDirectory() as tmpdir:

        config_list: list[TransformerConfiguration] = [
            TransformerConfiguration(
                study_data=dct.StudyData(
                    study_name=f"{study_name_prefix[test_index]}_{i}",
                    optimization_directory=tmpdir,
                    number_of_trials=int_number_of_trials[test_index],
                    calculation_mode=CalcModeEnum.new_mode,
                ),
                simulation_calculation_mode=CalcModeEnum.new_mode,
                transformer_toml_data=toml_data,
            )
            for i in range(NUMBER_OF_TRANSFORMERS)
        ]
    
        # Build transformer_requirements_list (minimum 3 elements)
        req_list: list[TransformerRequirements] = [
            TransformerRequirements(
                transformer_number_in_circuit=i,
                circuit_id=f"{circuit_id_prefix[test_index]}_{i}",
                l_s12_target=float_l_s12_target[test_index],
                l_h_target=float_l_h_target[test_index],
                n_target=float_n_target[test_index],
                time_vec=time_vec,
                current_1_vec=current_1_vec,
                current_2_vec=current_2_vec,
            )
            for i in range(NUMBER_OF_TRANSFORMERS)
        ]

        # Create the object under test
        test_object: TransformerOptimization = TransformerOptimization()

        # Call method and verify every DTO field
        if not is_error:
            test_object.initialize_transformer_optimization_list(config_list, req_list)

            # Verify the length of the final list
            assert len(test_object._optimization_config_list) == NUMBER_OF_TRANSFORMERS

            for i, req in enumerate(req_list):
                config = config_list[i]
                trf_toml = config.transformer_toml_data

                # Exactly one DTO must have been appended per requirement
                assert len(test_object._optimization_config_list[i]) == 1

                dto: TransformerOptimizationDto = test_object._optimization_config_list[i][0]

                expected_trial_directory = os.path.join(
                    config.study_data.optimization_directory,
                    req.circuit_id,
                    config.study_data.study_name
                )

                sto: fmt.StoSingleInputConfig = dto.fmt_transformer_optimization_dto

                # Verify Insulation DTO
                assert sto.insulations.iso_window_top_core_top    == trf_toml.insulation.iso_window_top_core_top
                assert sto.insulations.iso_window_top_core_bot    == trf_toml.insulation.iso_window_top_core_bot
                assert sto.insulations.iso_window_top_core_left   == trf_toml.insulation.iso_window_top_core_left
                assert sto.insulations.iso_window_top_core_right  == trf_toml.insulation.iso_window_top_core_right
                assert sto.insulations.iso_window_bot_core_top    == trf_toml.insulation.iso_window_bot_core_top
                assert sto.insulations.iso_window_bot_core_bot    == trf_toml.insulation.iso_window_bot_core_bot
                assert sto.insulations.iso_window_bot_core_left   == trf_toml.insulation.iso_window_bot_core_left
                assert sto.insulations.iso_window_bot_core_right  == trf_toml.insulation.iso_window_bot_core_right
                assert sto.insulations.iso_primary_to_primary     == trf_toml.insulation.iso_primary_to_primary
                assert sto.insulations.iso_secondary_to_secondary == trf_toml.insulation.iso_secondary_to_secondary
                assert sto.insulations.iso_primary_to_secondary   == trf_toml.insulation.iso_primary_to_secondary

                # Verify Material Data Sources
                assert sto.material_data_sources.permeability_datasource == trf_toml.material_data_sources.permeability_datasource
                assert sto.material_data_sources.permittivity_datasource == trf_toml.material_data_sources.permittivity_datasource

                # Verify FMT Transformer Optimization DTO (Sto Single Input Config) 
                assert sto.stacked_transformer_study_name       == config.study_data.study_name
                assert sto.temperature                          == trf_toml.boundary_conditions.temperature
                assert sto.max_transformer_total_height         == trf_toml.boundary_conditions.max_transformer_total_height
                assert sto.max_core_volume                      == trf_toml.boundary_conditions.max_core_volume
                assert sto.n_p_top_min_max_list                 == trf_toml.design_space.n_p_top_min_max_list
                assert sto.n_p_bot_min_max_list                 == trf_toml.design_space.n_p_bot_min_max_list
                assert sto.material_list                        == trf_toml.design_space.material_name_list
                assert sto.core_name_list                       == trf_toml.design_space.core_name_list
                assert sto.core_inner_diameter_min_max_list     == trf_toml.design_space.core_inner_diameter_min_max_list
                assert sto.window_w_min_max_list                == trf_toml.design_space.window_w_min_max_list
                assert sto.window_h_bot_min_max_list            == trf_toml.design_space.window_h_bot_min_max_list
                assert sto.primary_litz_wire_list               == trf_toml.design_space.primary_litz_wire_list
                assert sto.secondary_litz_wire_list             == trf_toml.design_space.secondary_litz_wire_list
                assert sto.fft_filter_value_factor              == trf_toml.settings.fft_filter_value_factor
                assert sto.mesh_accuracy                        == trf_toml.settings.mesh_accuracy
                assert sto.stacked_transformer_optimization_directory == expected_trial_directory

                # Verify Thermal DTO
                assert dto.thermal_data.tim_thickness    == trf_toml.thermal_data.thermal_cooling[0]
                assert dto.thermal_data.tim_conductivity == trf_toml.thermal_data.thermal_cooling[1]

                # Verify Filter distance
                assert dto.factor_dc_losses_min_max_list == trf_toml.filter_distance.factor_dc_losses_min_max_list

                # Verify transfromer_requirements fields
                assert dto.transformer_requirements.transformer_number_in_circuit == req.transformer_number_in_circuit
                assert dto.transformer_requirements.circuit_id == req.circuit_id
                assert dto.transformer_requirements.l_s12_target == req.l_s12_target
                assert dto.transformer_requirements.l_h_target == req.l_h_target
                assert dto.transformer_requirements.n_target == req.n_target
                assert dto.transformer_requirements.time_vec == req.time_vec
                assert dto.transformer_requirements.current_1_vec == req.current_1_vec
                assert dto.transformer_requirements.current_2_vec == req.current_2_vec

                # Verify Progress Data
                assert dto.progress_data.progress_status == ProgressStatus.Idle
                assert dto.progress_data.run_time == 0
                assert dto.progress_data.number_of_filtered_points == 0

                # Verify Transformer Optimization DTO (Top level)
                assert dto.trial_directory == expected_trial_directory
                assert dto.circuit_id == req.circuit_id
                assert dto.transformer_number_in_circuit == req.transformer_number_in_circuit
                assert dto.number_of_trails == config.study_data.number_of_trials
        else:
            with pytest.raises(ValueError) as error_message:
                test_object.initialize_transformer_optimization_list(config_list, req_list)
            
            assert "Serious programming error in transformer optimization" in str(error_message.value)
