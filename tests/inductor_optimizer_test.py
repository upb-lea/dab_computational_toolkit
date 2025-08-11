"""Unit tests for inductor optimization."""

# python libraries
import logging
import copy
from enum import Enum

# 3rd party libraries
import pytest

# own libraries
import dct.inductor_optimization as test_circuit
import dct.toml_checker as tc
import dct.server_ctl_dtos
import femmt as fmt

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


#########################################################################################################
# test of verify_optimization_parameter of circuit_optimization
#########################################################################################################

""" Information
## Inductor #####################################################################
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
factor_dc_losses_min_max_list: list[float] -> 0<val<=100
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
    float_min_max_list_configuration_gt0_lt5: list[list[float]] = (
        [[1e-18, 1e-18], [4.9999, 4.9999], [0.1, 2], [4, 3.5], [1], [0.1, 2, 2.3], [0, 1.1], [1, 5]])
    float_min_max_list_configuration_gt0_le100: list[list[float]] = (
        [[1e-18, 1e-18], [100, 100], [34, 77], [90, 67], [33], [33, 66, 99], [0, 88], [55, 100.1]])
    float_value_gem40_le175: list[float] = [-40, 175, 80.1, 105.55, -21.3, 125.5, -40.21, 275]
    float_value_gt0_lt1em1: list[float] = [1e-17, 0.0991, 0.034, 0.074, 0.033, 0.066, 0, 0.1]

    # Check if core list is not empty
    if is_core_list_available:
        # Create an empty list for the core parameter values
        design_space_min_max_entry = []
    else:
        # Create an empty list for the core parameter values
        design_space_min_max_entry = float_min_max_list_configuration_gt0_lt5[test_index]

    # Initialize the inductor parameters
    test_inductor_parameter: tc.TomlInductor = tc.TomlInductor(
        design_space=tc.TomlInductorDesignSpace(
            core_name_list=core_name_list_configuration[test_index],
            material_name_list=["3C95"],
            litz_wire_name_list=litz_name_list_configuration[test_index],
            core_inner_diameter_min_max_list=design_space_min_max_entry,
            window_h_min_max_list=design_space_min_max_entry,
            window_w_min_max_list=design_space_min_max_entry),
        insulations=tc.TomlInductorInsulation(
            primary_to_primary=float_value_gt0_lt1em1[test_index],
            core_bot=float_value_gt0_lt1em1[test_index],
            core_top=float_value_gt0_lt1em1[test_index],
            core_right=float_value_gt0_lt1em1[test_index],
            core_left=float_value_gt0_lt1em1[test_index]),
        boundary_conditions=tc.TomlInductorBoundaryConditions(
            temperature=float_value_gem40_le175[test_index]),
        filter_distance=dct.TomlFilterDistance(
            factor_dc_losses_min_max_list=float_min_max_list_configuration_gt0_le100[test_index]),
        material_data_sources=tc.TomlMaterialDataSources(
            permeability_datasource="",
            permeability_datatype="",
            permeability_measurement_setup="",
            permittivity_datasource="",
            permittivity_datatype="",
            permittivity_measurement_setup="")
    )

    # Create boundary list from minimum-maximum list with assigned parameters
    min_max_list_name_list_w_o_core_list: list[str] = ["core_inner_diameter_min_max_list", "window_h_min_max_list",
                                                       "window_w_min_max_list", "factor_dc_losses_min_max_list"]
    min_max_list_name_list_w_core_list: list[str] = ["factor_dc_losses_min_max_list"]
    value_name_list: list[str] = ["primary_to_primary", "core_bot", "core_top", "core_right", "core_left", "temperature"]

    # Perform the test
    is_consistent, error_report = test_circuit.InductorOptimization.verify_optimization_parameter(test_inductor_parameter)
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
