"""Unit tests for boundary check."""

# python libraries
import logging
import copy
from enum import Enum
import sys

# 3rd party libraries
import pytest
from _pytest.logging import LogCaptureFixture

# own libraries
import dct.boundary_check as test_module

# Enable logger
pytestlogger = logging.getLogger(__name__)

class TestCase(Enum):
    """Enum of test types."""

    # Valid test case
    ValidValues = 0             # Test value at lower boundary
    # Failure test case
    BoundaryInconsistent = 1    # Test when minimum > maximum ( Only for minimum maximum list)
    EmptyList = 2          # Test when the list has got too less entries ( Only for minimum maximum list)
    ExceedLowerLimit = 3        # Test when the lower limit is exceeded
    ExceedUpperLimit = 4        # Test when the lower limit is exceeded

#########################################################################################################
# test of convert_int_list_to_float_list
#########################################################################################################

# convert_int_list_to_float_list(int_value_list: list[int]) -> list[float]:
# test parameter list (counter)
@pytest.mark.parametrize("test_list, is_valid", [
    # Valid test case
    # All values
    ([0, 1, -1, 42, sys.maxsize, -sys.maxsize - 1], True),
    # One value
    ([-1], True),
    # Invalid list (empty list)
    ([], False)
])
# Unit test function
def test_convert_int_list_to_float_list(caplog: LogCaptureFixture, test_list: list[int], is_valid: bool) -> None:
    """Test the method convert_int_list_to_float_list.

    :param caplog: class instance for logger data
    :type  caplog: LogCaptureFixture
    :param test_list: List of integer to convert
    :type  test_list: list[int]
    :param is_valid: Indicates, if a valid list is provided
    :type  is_valid: bool
    """
    with caplog.at_level(logging.INFO):
        # Perform the test
        result = test_module.BoundaryCheck.convert_int_list_to_float_list(test_list)

        # Check if the test case is valid
        if is_valid:
            # Verity the result
            index = 0
            for entry_value in test_list:
                assert result[index] == float(entry_value)
                index = index + 1
        else:
            if len(caplog.records) > 0:
                assert caplog.records[0].message == "List is empty. No value is converted!"
            else:
                pytest.fail("No warning logged")

#########################################################################################################
# test of check_float_value
#########################################################################################################

# check_float_value(minimum: float, maximum: float, parameter_value: float,
#                      parameter_name: str, check_type_minimum: CheckCondition, check_type_maximum: CheckCondition) -> tuple[bool, str]:

# test parameter list (counter)
@pytest.mark.parametrize("test_index, report_index, is_passed, check_condition_minimum, check_condition_maximum", [
    # Valid test case
    # Perform standard test
    (-1, 0, False, test_module.CheckCondition.check_inclusive, test_module.CheckCondition.check_inclusive),
    # Test valid boundary inclusive, inclusive
    (0, 0, True, test_module.CheckCondition.check_inclusive, test_module.CheckCondition.check_inclusive),
    # Test valid boundary inclusive, exclusive
    (1, 0, True, test_module.CheckCondition.check_inclusive, test_module.CheckCondition.check_exclusive),
    # Test valid boundary  exclusive, inclusive
    (2, 0, True, test_module.CheckCondition.check_exclusive, test_module.CheckCondition.check_inclusive),
    # Test valid boundary  exclusive, exclusive
    (3, 0, True, test_module.CheckCondition.check_inclusive, test_module.CheckCondition.check_exclusive),
    # Test valid boundary inclusive, ignore
    (4, 0, True, test_module.CheckCondition.check_inclusive, test_module.CheckCondition.check_ignore),
    # Test valid boundary ignore, inclusive
    (5, 0, True, test_module.CheckCondition.check_ignore, test_module.CheckCondition.check_inclusive),
    # Test valid boundary  exclusive, ignore
    (6, 0, True, test_module.CheckCondition.check_exclusive, test_module.CheckCondition.check_ignore),
    # Test valid boundary ignore, exclusive,
    (7, 0, True, test_module.CheckCondition.check_ignore, test_module.CheckCondition.check_exclusive),
    # Test valid boundary ignore, ignore
    (8, 0, True, test_module.CheckCondition.check_ignore, test_module.CheckCondition.check_ignore),
    # Failure test case
    # Test when the lower limit is exceeded with inclusive boundaries
    (9, 1, False, test_module.CheckCondition.check_inclusive, test_module.CheckCondition.check_inclusive),
    # Test when the lower limit is exceeded with inclusive boundaries
    (10, 2, False, test_module.CheckCondition.check_inclusive, test_module.CheckCondition.check_inclusive),
    # Test when the lower limit is exceeded with exclusive boundaries
    (11, 3, False, test_module.CheckCondition.check_exclusive, test_module.CheckCondition.check_exclusive),
    # Test when the lower limit is exceeded with exclusive boundaries
    (12, 4, False, test_module.CheckCondition.check_exclusive, test_module.CheckCondition.check_exclusive)
])
# Unit test function
def test_check_float_value(test_index: int, report_index: int, is_passed: bool,
                           check_condition_minimum: test_module.CheckCondition,
                           check_condition_maximum: test_module.CheckCondition) -> None:
    """Test the method check_float_value.

    :param test_index: Index to define the input parameter index
    :type  test_index: int
    :param report_index: Index to define the expected error report
    :type  report_index: int
    :param is_passed: Indicates, if the test is expected to passed
    :type  is_passed: bool
    :param check_condition_minimum: Check condition for the minimum boundary
    :type  check_condition_minimum: CheckCondition
    :param check_condition_maximum: Check condition for the maximum boundary
    :type  check_condition_maximum: CheckCondition
    """
    string_test_values: list[str] = ["A", "b9", "Test123", "x_Y_z890", "6Secure//Pass01", "Alpha//Bravo2024", "z_Z9y_Y8_x_X7", "7Long_String_Test999"]
    str_test_len = len(string_test_values)

    parameter_name = string_test_values[test_index % str_test_len]

    small_constant: float = 1.000000000001

    # Variable definition
    # Boundary limit test
    # List entries for values and list (exception *in between for values):
    # at lower boundary | at upper boundary | in between | minimum > maximum* | too few entries*
    # exceed the lower limit | exceed the upper limit
    boundary_limit_list: list[list[float]] = [[24, 5e10], [-sys.float_info.max, 20], [0, sys.float_info.max],
                                              [5e10, 24]]
    value_list_boundary_inclusive: list[list[float]] = (
        [[24, 77.987, 5e10], [-sys.float_info.max, 1e-12, 20], [0, 8e12, sys.float_info.max], [26]])
    value_list_boundary_exclusive: list[list[float]] = (
        [[24*small_constant, 77.987, 5e10/small_constant], [-sys.float_info.max/small_constant, 1e-12, 20/small_constant],
         [1.e-2, 8e12, sys.float_info.max/small_constant], [26]])

    # Special test lists
    limit_list: list[float] = [-2e5, 5e10]
    value_list: list[list[float]] = (
        [[5e10, -2e5, 377], [23, -2e5, 5e10 / small_constant], [8.4e9, -2e5 / small_constant, 5e10, 87],
         [23, 5e10 / small_constant, -2e5 / small_constant],
         [455, -2e5, 377, 5e10 * small_constant, 22], [5e10, -2e5 * small_constant, 377, 22],
         [8.4e9, -2e5 / small_constant, 6e10], [8.4e9, -3e5, 5e10 / small_constant, 87],
         [5e11, 8.4e9, -3e5],
         # Invalid test cases
         [5e10, -2e5 * small_constant, 377], [-2e5, 377, 5e10 * small_constant, 22],
         [8.4e9, -3e5, 5e10 / small_constant, 87], [-2e5 / small_constant, 5e10, 87]
         ])

    # Error messages container
    error_limit_message_list = (
        ["",
         f"    Minimum boundary value {5e10} is greater than maximum value {24}!\n",
         "    Length of minimum maximum list  is not 2!\n"])

    # Check test index
    if test_index < 0:
        for index in range(len(boundary_limit_list)):
            for entry_value in value_list_boundary_inclusive[index]:
                # Perform the test with inclusive check type
                is_consistent, report = test_module.BoundaryCheck.check_float_value(
                    boundary_limit_list[index][0], boundary_limit_list[index][1], entry_value, parameter_name,
                    test_module.CheckCondition.check_inclusive, test_module.CheckCondition.check_inclusive)
                if index < 3:
                    assert is_consistent
                    assert report == error_limit_message_list[0]
                else:
                    assert not is_consistent
                    assert report == error_limit_message_list[index - 2]
            for entry_value in value_list_boundary_exclusive[index]:
                # Perform the test with exclusive check type
                is_consistent, report = test_module.BoundaryCheck.check_float_value(
                    boundary_limit_list[index][0], boundary_limit_list[index][1], entry_value, parameter_name,
                    test_module.CheckCondition.check_exclusive, test_module.CheckCondition.check_exclusive)
                if index < 3:
                    assert is_consistent
                    assert report == error_limit_message_list[0]
                else:
                    assert not is_consistent
                    assert report == error_limit_message_list[index - 2]
    else:
        # expected report
        # Error messages container
        error_message_list = (
            ["",
             f"    Parameter {parameter_name}= {-2e5 * small_constant} is less than minimum value {-2e5}!\n",
             f"    Parameter {parameter_name}= {5e10 * small_constant} is greater than maximum value {5e10}!\n",
             f"    Parameter {parameter_name}= {-3e5} is less equal minimum value {-2e5}!\n",
             f"    Parameter {parameter_name}= {5e10} is greater equal maximum value {5e10}!\n"])

        # Reset the report
        report = ""
        result = True
        for entry_value in value_list[test_index]:
            # Perform the tests
            is_consistent, message = test_module.BoundaryCheck.check_float_value(
                limit_list[0], limit_list[1], entry_value, parameter_name,
                check_condition_minimum, check_condition_maximum)
            report = report + message
            result = result and is_consistent

        assert result == is_passed
        assert report == error_message_list[report_index]


#########################################################################################################
# test of check_float_value_list
#########################################################################################################

#  check_float_value_list(minimum: float, maximum: float, value_list: list[tuple[float, str]],
#                               check_type_minimum: CheckCondition, check_type_maximum: CheckCondition) -> tuple[bool, str]:

# test parameter list (counter)
@pytest.mark.parametrize("test_index, report_index, is_passed, check_condition_minimum, check_condition_maximum", [
    # Valid test case
    # Test valid boundary inclusive, inclusive
    (0, 0, True, test_module.CheckCondition.check_inclusive, test_module.CheckCondition.check_inclusive),
    # Test valid boundary inclusive, exclusive
    (1, 0, True, test_module.CheckCondition.check_inclusive, test_module.CheckCondition.check_exclusive),
    # Test valid boundary  exclusive, inclusive
    (2, 0, True, test_module.CheckCondition.check_exclusive, test_module.CheckCondition.check_inclusive),
    # Test valid boundary  exclusive, exclusive
    (3, 0, True, test_module.CheckCondition.check_exclusive, test_module.CheckCondition.check_exclusive),
    # Test valid boundary inclusive, ignore
    (4, 0, True, test_module.CheckCondition.check_inclusive, test_module.CheckCondition.check_ignore),
    # Test valid boundary ignore, inclusive
    (5, 0, True, test_module.CheckCondition.check_ignore, test_module.CheckCondition.check_inclusive),
    # Test valid boundary  exclusive, ignore
    (6, 0, True, test_module.CheckCondition.check_exclusive, test_module.CheckCondition.check_ignore),
    # Test valid boundary ignore, exclusive,
    (7, 0, True, test_module.CheckCondition.check_ignore, test_module.CheckCondition.check_exclusive),
    # Test valid boundary ignore, ignore
    (8, 0, True, test_module.CheckCondition.check_ignore, test_module.CheckCondition.check_ignore),
    # Failure test case
    # Test when the lower limit and upper limit is exceeded with inclusive boundaries
    (9, 1, False, test_module.CheckCondition.check_inclusive, test_module.CheckCondition.check_inclusive),
    # Test when the lower and upper limit is exceeded with exclusive boundaries
    (10, 2, False, test_module.CheckCondition.check_exclusive, test_module.CheckCondition.check_exclusive),
    # Test when the list is empty
    (11, 0, False, test_module.CheckCondition.check_exclusive, test_module.CheckCondition.check_exclusive)
])
# Unit test function
def test_check_float_value_list(caplog: LogCaptureFixture, test_index: int, report_index: int, is_passed: bool,
                                check_condition_minimum: test_module.CheckCondition,
                                check_condition_maximum: test_module.CheckCondition) -> None:
    """Test the method check_float_value_list.

    :param caplog: class instance for logger data
    :type  caplog: LogCaptureFixture
    :param test_index: Index to define the input parameter index
    :type  test_index: int
    :param report_index: Index to define the expected error report
    :type  report_index: int
    :param is_passed: Indicates, if the test is expected to passed
    :type  is_passed: bool
    :param check_condition_minimum: Check condition for the minimum boundary
    :type  check_condition_minimum: CheckCondition
    :param check_condition_maximum: Check condition for the maximum boundary
    :type  check_condition_maximum: CheckCondition
    """
    string_test_values: list[str] = ["A", "c9", "Test123", "x_Y_z890", "6Secure//Pass01", "Alpha//Bravo2024", "z_Z9y_Y8_x_X7", "7Long_String_Test999"]
    str_test_len = len(string_test_values)
    small_constant: float = 1.000000000001

    # Variable definition
    # Boundary limit test
    # List entries for values and list (exception *in between for values):
    # at lower boundary | at upper boundary | in between | minimum > maximum* | too few entries*
    # exceed the lower limit | exceed the upper limit
    # Test limits and values lists
    limit_list: list[float] = [-3.87e3, 7.987e8]
    parameter_value_list: list[list[tuple[float, str]]] = (
        [[(7.987e8, string_test_values[test_index % str_test_len]),
          (-3.87e3, string_test_values[(test_index + 1) % str_test_len]),
          (377, string_test_values[(test_index + 2) % str_test_len])],
         [(-3.87e3, string_test_values[test_index % str_test_len]),
          (23, string_test_values[(test_index + 1) % str_test_len]),
          (-2e2, string_test_values[(test_index + 2) % str_test_len]),
          (7.987e8 / small_constant, string_test_values[(test_index + 3) % str_test_len])],
         [(8.4e7, string_test_values[test_index % str_test_len]),
          (-3.87e3 / small_constant, string_test_values[(test_index + 1) % str_test_len]),
          (7.987e8, string_test_values[(test_index + 2) % str_test_len]),
          (87, string_test_values[(test_index + 3) % str_test_len])],
         [(23, string_test_values[test_index % str_test_len]),
          (7.987e8 / small_constant, string_test_values[(test_index + 1) % str_test_len]),
          (-3.87e3 / small_constant, string_test_values[(test_index + 2) % str_test_len])],
         [(455, string_test_values[test_index % str_test_len]),
          (-2e3, string_test_values[(test_index + 1) % str_test_len]),
          (377, string_test_values[(test_index + 2) % str_test_len]),
          (7.987e8 * small_constant, string_test_values[(test_index + 3) % str_test_len])],
         [(7.987e8, string_test_values[test_index % str_test_len]),
          (-3.87e3 * small_constant, string_test_values[(test_index + 1) % str_test_len]),
          (377, string_test_values[(test_index + 2) % str_test_len])],
         [(8.4e9, string_test_values[test_index % str_test_len]),
          (-3.87e3 / small_constant, string_test_values[(test_index + 1) % str_test_len]),
          (6e10, string_test_values[(test_index + 2) % str_test_len])],
         [(8.4e2, string_test_values[test_index % str_test_len]),
          (-5.87e43, string_test_values[(test_index + 1) % str_test_len]),
          (7.987e8 / small_constant, string_test_values[(test_index + 2) % str_test_len]),
          (87, string_test_values[(test_index + 3) % str_test_len])],
         [(5e11, string_test_values[test_index % str_test_len]),
          (8.4e3, string_test_values[(test_index + 1) % str_test_len]),
          (-3.76e25, string_test_values[(test_index + 2) % str_test_len])],
         # Invalid test cases
         [(7.987e8, string_test_values[test_index % str_test_len]),
          (-3.87e3 * small_constant, string_test_values[(test_index + 1) % str_test_len]),
          (377, string_test_values[(test_index + 2) % str_test_len]),
          (7.987e8 * small_constant, string_test_values[(test_index + 3) % str_test_len])],
         [(8.4e9, string_test_values[test_index % str_test_len]),
          (-3e2, string_test_values[(test_index + 1) % str_test_len]),
          (7.987e8 / small_constant, string_test_values[(test_index + 2) % str_test_len]),
          (867, string_test_values[(test_index + 3) % str_test_len]),
          (-3.87e3, string_test_values[(test_index + 4) % str_test_len]),
          (87, string_test_values[(test_index + 5) % str_test_len])],
         []])

    # Error messages container
    error_limit_message_list = (
        ["",
         f"    Minimum boundary value {5e10} is greater than maximum value {24}!\n",
         "    Length of minimum maximum list  is not 2!\n"])

    list_name: list[str] = ([string_test_values[(test_index + 1) % str_test_len],
                             string_test_values[(test_index + 3) % str_test_len],
                             string_test_values[test_index % str_test_len],
                             string_test_values[(test_index + 4) % str_test_len]])

    # Check test index
    # expected report
    # Error messages container
    error_message_list = (
        ["",
         (f"    Parameter {list_name[0]}= {-3.87e3 * small_constant} is less than minimum value {-3.87e3}!\n"
          f"    Parameter {list_name[1]}= {7.987e8 * small_constant} is greater than maximum value {7.987e8}!\n"),
         (f"    Parameter {list_name[2]}= {8.4e9} is greater equal maximum value {7.987e8}!\n"
          f"    Parameter {list_name[3]}= {-3.87e3} is less equal minimum value {-3.87e3}!\n")])

    with caplog.at_level(logging.INFO):
        # Perform the test
        result, report = test_module.BoundaryCheck.check_float_value_list(
            limit_list[0], limit_list[1], parameter_value_list[test_index], check_condition_minimum, check_condition_maximum)

        assert report == error_message_list[report_index]

        # Check logger output
        if not is_passed and error_message_list[report_index] == "":
            if len(caplog.records) > 0:
                assert caplog.records[0].message == "List is empty. There is not performed any check!"
            else:
                pytest.fail("No warning logged")
        else:
            assert result == is_passed


#########################################################################################################
# test of check_float_min_max_values_list
#########################################################################################################

# check_float_min_max_values_list(minimum: float, maximum: float, min_max_value_list: list[tuple[list[float], str]],
#                                     check_type_minimum: CheckCondition, check_type_maximum: CheckCondition) -> tuple[bool, str]:

# test parameter list (counter)
@pytest.mark.parametrize("test_index, report_index, is_passed,  is_limit_error, check_condition_minimum, check_condition_maximum", [
    # Valid test case
    # Test valid boundary inclusive, inclusive
    (0, 0, True, False, test_module.CheckCondition.check_inclusive, test_module.CheckCondition.check_inclusive),
    # Test valid boundary  exclusive, exclusive
    (1, 0, True, False, test_module.CheckCondition.check_exclusive, test_module.CheckCondition.check_exclusive),
    # Test valid boundary ignore, inclusive
    (2, 0, True, False, test_module.CheckCondition.check_ignore, test_module.CheckCondition.check_ignore),
    # Failure test case
    # Test when the lower limit and upper limit is exceeded with inclusive boundaries
    (3, 1, False, False, test_module.CheckCondition.check_inclusive, test_module.CheckCondition.check_inclusive),
    # Test when the lower and upper limit is exceeded with exclusive boundaries
    (4, 2, False, False, test_module.CheckCondition.check_exclusive, test_module.CheckCondition.check_exclusive),
    # Test when number of entries are too high
    (5, 3, False, False, test_module.CheckCondition.check_exclusive, test_module.CheckCondition.check_exclusive),
    # Test when the list is empty
    (6, 4, False, True, test_module.CheckCondition.check_exclusive, test_module.CheckCondition.check_exclusive)
])
# Unit test function
def test_check_float_min_max_values_list(caplog: LogCaptureFixture, test_index: int, report_index: int, is_passed: bool,
                                         is_limit_error: bool, check_condition_minimum: test_module.CheckCondition,
                                         check_condition_maximum: test_module.CheckCondition) -> None:
    """Test the method check_float_min_max_values_list.

    :param caplog: class instance for logger data
    :type  caplog: LogCaptureFixture
    :param test_index: Index to define the input parameter index
    :type  test_index: int
    :param report_index: Index to define the expected error report
    :type  report_index: int
    :param is_passed: Indicates, if the test is expected to passed
    :type  is_passed: bool
    :param is_limit_error: Indicates, if a limit error shall be set
    :type  is_limit_error: bool
    :param check_condition_minimum: Check condition for the minimum boundary
    :type  check_condition_minimum: CheckCondition
    :param check_condition_maximum: Check condition for the maximum boundary
    :type  check_condition_maximum: CheckCondition
    """
    string_test_values: list[str] = ["A", "c9", "Test123", "x_Y_z890", "6Secure//Pass01", "Alpha//Bravo2024", "z_Z9y_Y8_x_X7", "7Long_String_Test999"]
    str_test_len = len(string_test_values)
    parameter_name = string_test_values[test_index % str_test_len]
    small_constant: float = 1.000000000001

    # Variable definition
    # Boundary limit test
    # List entries for values and list (exception *in between for values):
    # at lower boundary | at upper boundary | in between | minimum > maximum* | too few entries*
    # exceed the lower limit | exceed the upper limit
    # Test limits and values lists

    # Test limits
    if not is_limit_error:
        limit_list: list[float] = [-7873.4, 2987.1]
    else:
        limit_list = [7873.4, -2987.1]

    pre_tuple_list: list[tuple[list[float], str]] = [([-388, 858.4], "dummy"), ([-188.4, 1258.9], "dummy")]
    post_tuple_1: tuple[list[float], str] = ([-478.3, 658.4], "dummy")
    post_tuple_2: tuple[list[float], str] = ([0, 1], "dummy")
    # Minimum maximum value list
    # First tuple entry: Boundary values, second tuple entry: In between
    min_max_value_list: list[tuple[list[float], str]] = (
        [([-7873.4, 2987.1], parameter_name),
         ([-7873.4 / small_constant, 2987.1 / small_constant], parameter_name),
         ([-8.87e8, 32987.1], parameter_name),
         # Invalid test cases
         ([-7873.4 * small_constant, 2987.1 * small_constant], parameter_name),
         ([-7873.4, 2987.1], parameter_name),
         ([73.4, 298.1, 312.1], parameter_name),
         ([], parameter_name)])

    # Copy tuple reference
    min_max_tuple = min_max_value_list[test_index]

    if len(min_max_tuple[0]) > 1:
        e_min_max_values = min_max_tuple[0]
    else:
        e_min_max_values = [0, 1]

    # Error messages container
    error_message_list = (
        ["",
         (f"    In list {parameter_name} the minimum entry value {e_min_max_values[0]} " 
          f"is less than boundary value {limit_list[0]}!\n" 
          f"    In list {parameter_name} the maximum entry value {e_min_max_values[1]} " 
          f"is greater than boundary value {limit_list[1]}!\n"),
         (f"    In list {parameter_name} the minimum entry value {e_min_max_values[0]} " 
          f"is less equal boundary value {limit_list[0]}!\n" 
          f"    In list {parameter_name} the maximum entry value {e_min_max_values[1]} " 
          f"is greater equal boundary value {limit_list[1]}!\n"),
         f"    Length of minimum maximum list {parameter_name} is not 2!\n",
         f"    Minimum boundary value {limit_list[0]} is greater equal maximum value {limit_list[1]}!\n"
         ])

    with caplog.at_level(logging.INFO):
        # Create list of tuples, if the list in tuple is not empty
        if len(min_max_tuple[0]) > 0:
            min_max_tuple_list: list[tuple[list[float], str]] = copy.deepcopy(pre_tuple_list)
            min_max_tuple_list.append(min_max_value_list[test_index])
            min_max_tuple_list.append(post_tuple_1)
            min_max_tuple_list.append(post_tuple_2)
        else:
            min_max_tuple_list = [min_max_tuple]

        # Perform the test
        result, report = test_module.BoundaryCheck.check_float_min_max_values_list(
            limit_list[0], limit_list[1], min_max_tuple_list,
            check_condition_minimum, check_condition_maximum)

        assert report == error_message_list[report_index]

        # Check logger output
        if not is_passed and error_message_list[report_index] == "":
            if len(caplog.records) > 0:
                assert caplog.records[0].message == "List is empty. There is not performed any check!"
            else:
                pytest.fail("No warning logged")
        else:
            assert result == is_passed


#########################################################################################################
# test of check_float_min_max_values
#########################################################################################################

# check_float_min_max_values(minimum: float, maximum: float, min_max_value: list[float], parameter_name: str,
#                                   check_type_minimum: CheckCondition, check_type_maximum: CheckCondition) -> tuple[bool, str]:

# test parameter list (counter)
@pytest.mark.parametrize(
    "test_index, report_index, is_passed, is_limit_error, check_condition_minimum, check_condition_maximum", [
        # Valid test case
        # Test valid boundary inclusive, inclusive
        (0, 0, True, False, test_module.CheckCondition.check_inclusive, test_module.CheckCondition.check_inclusive),
        # Test valid boundary inclusive, exclusive
        (1, 0, True, False, test_module.CheckCondition.check_inclusive, test_module.CheckCondition.check_exclusive),
        # Test valid boundary  exclusive, inclusive
        (2, 0, True, False, test_module.CheckCondition.check_exclusive, test_module.CheckCondition.check_inclusive),
        # Test valid boundary  exclusive, exclusive
        (3, 0, True, False, test_module.CheckCondition.check_exclusive, test_module.CheckCondition.check_exclusive),
        # Test valid boundary inclusive, ignore
        (4, 0, True, False, test_module.CheckCondition.check_inclusive, test_module.CheckCondition.check_ignore),
        # Test valid boundary ignore, inclusive
        (5, 0, True, False, test_module.CheckCondition.check_ignore, test_module.CheckCondition.check_inclusive),
        # Test valid boundary  exclusive, ignore
        (6, 0, True, False, test_module.CheckCondition.check_exclusive, test_module.CheckCondition.check_ignore),
        # Test valid boundary ignore, exclusive,
        (7, 0, True, False, test_module.CheckCondition.check_ignore, test_module.CheckCondition.check_exclusive),
        # Test valid boundary ignore, ignore
        (8, 0, True, False, test_module.CheckCondition.check_ignore, test_module.CheckCondition.check_ignore),
        # Failure test case
        # Test when the lower limit and upper limit is exceeded with inclusive boundaries
        (9, 1, False, False, test_module.CheckCondition.check_inclusive, test_module.CheckCondition.check_inclusive),
        # Test when the lower and upper limit is exceeded with exclusive boundaries
        (10, 2, False, False, test_module.CheckCondition.check_exclusive, test_module.CheckCondition.check_exclusive),
        # Test when minimum is higher than maximum
        (11, 3, False, False, test_module.CheckCondition.check_exclusive, test_module.CheckCondition.check_exclusive),
        # Test when limit is inconsistent?
        (12, 4, False, True, test_module.CheckCondition.check_exclusive, test_module.CheckCondition.check_exclusive),
        # Test when number of entries are too high
        (13, 5, False, False, test_module.CheckCondition.check_exclusive, test_module.CheckCondition.check_exclusive),
        # Test when number of entries are too low
        (14, 5, False, False, test_module.CheckCondition.check_exclusive, test_module.CheckCondition.check_exclusive)])
# Unit test function
def test_check_float_min_max_values(test_index: int, report_index: int, is_passed: bool,
                                    is_limit_error: bool, check_condition_minimum: test_module.CheckCondition,
                                    check_condition_maximum: test_module.CheckCondition) -> None:
    """Test the method load_toml_file.

    :param test_index: Index to define the input parameter index
    :type  test_index: int
    :param report_index: Index to define the expected error report
    :type  report_index: int
    :param is_passed: Indicates, if the test is expected to passed
    :type  is_passed: bool
    :param is_limit_error: Indicates, if a limit error shall be set
    :type  is_limit_error: bool
    :param check_condition_minimum: Check condition for the minimum boundary
    :type  check_condition_minimum: CheckCondition
    :param check_condition_maximum: Check condition for the maximum boundary
    :type  check_condition_maximum: CheckCondition
    """
    string_test_values: list[str] = ["A", "c9", "Test123", "x_Y_z890", "6Secure//Pass01", "Alpha//Bravo2024", "z_Z9y_Y8_x_X7", "7Long_String_Test999"]
    str_test_len = len(string_test_values)
    small_constant: float = 1.000000000001
    parameter_name = string_test_values[test_index % str_test_len]

    # Variable definition
    # Test limits
    if not is_limit_error:
        limit_list: list[float] = [-6.87e3, 9.987e8]
    else:
        limit_list = [9.987e8, -6.87e3]
    # Minimum maximum value list
    # First tuple entry: Boundary values, second tuple entry: In between
    min_max_value_list: list[tuple[list[float], list[float]]] = (
        [([-6.87e3, 9.987e8], [-2.88e2, 8.25e8]),
         ([-6.87e3, 9.987e8 / small_constant], [-4.88e2, 8.258]),
         ([-6.87e3 / small_constant, 9.987e8], [-7.88e2, 8.2e7]),
         ([-6.87e3 / small_constant, 9.987e8 / small_constant], [11, 34]),
         ([-6.87e3, 3.987e10], [-2.88e2, 8.25e8]),
         ([-3.87e4, 9.987e8], [-4.88, 8.258]),
         ([-6.87e3 / small_constant, 8.987e18], [-3.88e2, 5.2e7]),
         ([-9.87e4, 9.987e8 / small_constant], [22, 377]),
         ([-2.87e4, 9.987e18], [0, 0]),
         # Invalid test cases
         ([-6.87e3 * small_constant, 9.987e8 * small_constant], [-7.87e3, 19.987e8]),
         ([-6.87e3, 9.987e8], [-4.87e4, 3.987e10]),
         ([2.88, 1.25], [-2.88e2, -18.25e2]),
         ([-2.88e2, 8.25e8], [-2.88e2, 8.25e8]),
         ([-2.88e2, 33, 8.25e8], [-1.88e2, 43, 8.25e2]),
         ([-2.88e2], [1])
         ])

    # Check test index (limit and in between)
    for min_max_entry in min_max_value_list[test_index]:
        if len(min_max_entry) == 2:
            max_index = 1
        else:
            max_index = 0

        # Error messages container
        error_message_list = (
            ["",
             (f"    In list {parameter_name} the minimum entry value {min_max_entry[0]} " 
              f"is less than boundary value {limit_list[0]}!\n" 
              f"    In list {parameter_name} the maximum entry value {min_max_entry[max_index]} " 
              f"is greater than boundary value {limit_list[1]}!\n"),
             (f"    In list {parameter_name} the minimum entry value {min_max_entry[0]} " 
              f"is less equal boundary value {limit_list[0]}!\n" 
              f"    In list {parameter_name} the maximum entry value {min_max_entry[max_index]} "
              f"is greater equal boundary value {limit_list[1]}!\n"),
             (f"    In list {parameter_name}: Minimum value {min_max_entry[0]} "
              f"is greater than {min_max_entry[max_index]}!\n"),
             f"    Minimum boundary value {limit_list[0]} is greater equal maximum value {limit_list[1]}!\n",
             f"    Length of minimum maximum list {parameter_name} is not 2!\n"
             ])

        # Perform the test
        result, report = test_module.BoundaryCheck.check_float_min_max_values(
            limit_list[0], limit_list[1], min_max_entry, parameter_name, check_condition_minimum, check_condition_maximum)

        assert report == error_message_list[report_index]
        assert result == is_passed

#########################################################################################################
# test of check_dictionary
#########################################################################################################

# check_dictionary(keyword_dictionary: dict, keyword: str, keyword_list_name: str) -> tuple[bool, str]:
# test parameter list (counter)
@pytest.mark.parametrize("test_index, test_keyword, report_index, is_passed, is_keyword_list_empty", [
    # Valid test case
    # Test keyword has only one letter
    (0, "A", 0, True, False),
    # Test keyword starts with a number
    (1, "6Secure//Pass01", 0, True, False),
    # Test keyword is at the end (and starts with a number)
    (2, "7Long_String_Test999", 0, True, False),
    # Test keyword is at the beginning
    (3, "c9", 0, True, False),
    # Test keyword is in between
    (4, "Alpha//Bravo2024", 0, True, False),
    # Failure test case
    # Test when the keyword is empty
    (5, "", 1, False, False),
    # Test when the keyword list is empty
    (6, "Hallo", 2, False, True),
    # Test when the keyword is not found
    (7, "Hallo", 3, False, False)])
# Unit test function
def test_check_dictionary(test_index: int, test_keyword: str, report_index: int, is_passed: bool, is_keyword_list_empty: bool) -> None:
    """Test the method load_toml_file.

    :param test_index: Index to define the input parameter index
    :type  test_index: int
    :param test_keyword: Test keyword to search in keyword dictionary
    :type  test_keyword: str
    :param report_index: Index to define the expected error report
    :type  report_index: int
    :param is_passed: Indicates, if the test is expected to passed
    :type  is_passed: bool
    :param is_keyword_list_empty: Indicates, if the keyword dictionary shall be initialized as empty
    :type  is_keyword_list_empty: bool

    """
    string_test_values: list[str] = ["c9", "A", "Test123", "x_Y_z890", "6Secure//Pass01", "Alpha//Bravo2024", "z_Z9y_Y8_x_X7", "7Long_String_Test999"]
    str_test_len = len(string_test_values)
    keyword_list_name = string_test_values[test_index % str_test_len]

    if not is_keyword_list_empty:
        keyword_dictionary: dict = {
            "c9": 5.22, "A": 3.34, "Test123": 7, "x_Y_z890": 2.233, "6Secure//Pass01": [1, 2, 3], "Alpha//Bravo2024": [4, 5, 6],
            "z_Z9y_Y8_x_X7": "Alphabet", "7Long_String_Test999": ["Rather long string", "And information"]}
    else:
        keyword_dictionary = {}

    # Error messages container
    error_message_list = (
        ["",
         "    Keyword is empty!\n",
         "    Dictionary is empty!\n",
         (f"    Keyword '{test_keyword}' in {keyword_list_name} does not match any keyword within dictionary:\n" 
          "    dict_keys(['c9', 'A', 'Test123', 'x_Y_z890', '6Secure//Pass01', 'Alpha//Bravo2024', " 
          "'z_Z9y_Y8_x_X7', '7Long_String_Test999'])!\n")])

    # Perform the test
    result, report = test_module.BoundaryCheck.check_dictionary(keyword_dictionary, test_keyword, keyword_list_name)

    assert report == error_message_list[report_index]
    assert result == is_passed
