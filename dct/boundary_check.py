"""Inductor optimization class."""
# python libraries
import enum
import logging

# 3rd party libraries

# own libraries

# configure root logger
logger = logging.getLogger(__name__)

class CheckCondition(enum.Enum):
    """Enum for type of check."""

    check_ignore = 0
    check_inclusive = 1
    check_exclusive = 2

class BoundaryCheck:
    """Boundary check for parameter."""

    @staticmethod
    def convert_int_list_to_float_list(int_value_list: list[int]) -> list[float]:
        """
        Convert a list of integer values in a list of float values.

        :param int_value_list: List of integer values
        :type int_value_list: list[int]
        :return: List of float values
        :rtype: list[float]
        """
        # Variable declaration
        float_value_list: list[float] = []

        # Check if list is empty
        if len(int_value_list) == 0:
            logger.info("List is empty. No value is converted!")

        # Perform insulation value check
        # Perform the boundary check
        for entry_value in int_value_list:
            float_value_list.append(float(entry_value))

        return float_value_list

    @staticmethod
    def check_float_value_list(minimum: float, maximum: float, value_list: list[tuple[float, str]],
                               check_type_minimum: CheckCondition, check_type_maximum: CheckCondition) -> tuple[bool, str]:
        """
        Verify the listed values according minimum and maximum.

        :param minimum: Minimum value of the range
        :type  minimum: float
        :param maximum: Maximum value of the range
        :type  maximum: float
        :param value_list: List of float values to check and the parameter name
        :type  value_list: list[tuple[float, str]]
        :param check_type_minimum: Type of check to perform according the minimum value
        :type  check_type_minimum: CheckCondition
        :param check_type_maximum: Type of check to perform according the maximum value
        :type  check_type_maximum: CheckCondition
        :return: tuple: Indication if the verification failed | Error text with description about the deviation
        :rtype: tuple[bool, str]
        """
        # Variable declaration
        is_check_list_passed: bool = True
        inconsistency_list_report: str = ""

        # Check if list is empty
        if len(value_list) == 0:
            logger.info("List is empty. There is not performed any check!")

        for check_parameter in value_list:
            is_check_passed, issue_report = BoundaryCheck.check_float_value(
                minimum, maximum, check_parameter[0], check_parameter[1], check_type_minimum, check_type_maximum)
            if not is_check_passed:
                inconsistency_list_report = inconsistency_list_report + issue_report
                is_check_list_passed = False

        return is_check_list_passed, inconsistency_list_report

    @staticmethod
    def check_float_min_max_values_list(minimum: float, maximum: float, min_max_value_list: list[tuple[list[float], str]],
                                        check_type_minimum: CheckCondition, check_type_maximum: CheckCondition) -> tuple[bool, str]:
        """
        Perform a boundary check of the minimum/maximum-pair.

        :param minimum: Minimum value of the range
        :type  minimum: float
        :param maximum: Maximum value of the range
        :type  maximum: float
        :param min_max_value_list: List of 2 float values according provided minimum and maximum to check and the parameter name
        :type  min_max_value_list: list[tuple[list[float], str]]
        :param check_type_minimum: Type of check to perform according the minimum value
        :type  check_type_minimum: CheckCondition
        :param check_type_maximum: Type of check to perform according the maximum value
        :type  check_type_maximum: CheckCondition
        :return: tuple: Indication if the verification failed | Error text with description about the deviation
        :rtype: tuple[bool, str]
        """
        # Variable declaration
        is_check_list_passed: bool = True
        inconsistency_list_report: str = ""

        # Check if list is empty
        if len(min_max_value_list) == 0:
            logger.info("List is empty. There is not performed any check!")

        for check_parameter in min_max_value_list:
            is_check_passed, issue_report = BoundaryCheck.check_float_min_max_values(
                minimum, maximum, check_parameter[0], check_parameter[1], check_type_minimum, check_type_maximum)
            if not is_check_passed:
                inconsistency_list_report = inconsistency_list_report + issue_report
                is_check_list_passed = False

        return is_check_list_passed, inconsistency_list_report

    @staticmethod
    def check_float_value(minimum: float, maximum: float, parameter_value: float,
                          parameter_name: str, check_type_minimum: CheckCondition, check_type_maximum: CheckCondition) -> tuple[bool, str]:
        """
        Verify the value according minimum and maximum.

        :param minimum: Minimum value of the range
        :type  minimum: float
        :param maximum: Maximum value of the range
        :type  maximum: float
        :param parameter_value: Float values to check and the value name
        :type  parameter_value: float
        :param parameter_name: Name of parameter to mention in inconsistency report, if check fails
        :type  parameter_name: str
        :param check_type_minimum: Type of check to perform according the minimum value
        :type  check_type_minimum: CheckCondition
        :param check_type_maximum: Type of check to perform according the maximum value
        :type  check_type_maximum: CheckCondition
        :return: tuple: Indication if the verification failed | Error text with description about the deviation
        :rtype: tuple[bool, str]
        """
        # Variable declaration
        is_check_passed: bool = True
        inconsistency_report: str = ""

        # Check the consistency of the input parameter itself
        if minimum > maximum:
            inconsistency_report = f"    Minimum boundary value {minimum} is greater than maximum value {maximum}!\n"
            is_check_passed = False
        else:
            # Perform the boundary check
            # Check minimum boundary
            if check_type_minimum == CheckCondition.check_exclusive:
                if parameter_value <= minimum:
                    inconsistency_report = f"    Parameter {parameter_name}= {parameter_value} is less equal minimum value {minimum}!\n"
                    is_check_passed = False

            elif check_type_minimum == CheckCondition.check_inclusive:
                if parameter_value < minimum:
                    inconsistency_report = f"    Parameter {parameter_name}= {parameter_value} is less than minimum value {minimum}!\n"
                    is_check_passed = False

            # Check maximum boundary
            if check_type_maximum == CheckCondition.check_exclusive:
                if parameter_value >= maximum:
                    inconsistency_report = f"    Parameter {parameter_name}= {parameter_value} is greater equal maximum value {maximum}!\n"
                    is_check_passed = False

            elif check_type_maximum == CheckCondition.check_inclusive:
                if parameter_value > maximum:
                    inconsistency_report = f"    Parameter {parameter_name}= {parameter_value} is greater than maximum value {maximum}!\n"
                    is_check_passed = False

        return is_check_passed, inconsistency_report

    @staticmethod
    def check_float_min_max_values(minimum: float, maximum: float, min_max_value: list[float], parameter_name: str,
                                   check_type_minimum: CheckCondition, check_type_maximum: CheckCondition) -> tuple[bool, str]:
        """
        Perform a boundary check for the minimum/maximum-pair.

        :param minimum: Minimum value of the range
        :type  minimum: float
        :param maximum: Maximum value of the range
        :type  maximum: float
        :param min_max_value: 2 float values for the minimum and maximum boundary check
        :type  min_max_value: list[tuple[list[float], str]]
        :param parameter_name: Name of parameter to mention in inconsistency report, if check fails
        :type  parameter_name: str
        :param check_type_minimum: Type of check to perform according the minimum value
        :type  check_type_minimum: CheckCondition
        :param check_type_maximum: Type of check to perform according the maximum value
        :type  check_type_maximum: CheckCondition
        :return: tuple: Indication if the verification failed | Error text with description about the deviation
        :rtype: tuple[bool, str]
        """
        # Variable declaration
        is_check_passed: bool = True
        inconsistency_report: str = ""

        # Check the consistency of the input parameter itself
        if minimum >= maximum:
            inconsistency_report = f"    Minimum boundary value {minimum} is greater equal maximum value {maximum}!\n"
            is_check_passed = False
        # Check length of list
        elif len(min_max_value) != 2:
            inconsistency_report = inconsistency_report + f"    Length of minimum maximum list {parameter_name} is not 2!\n"
            is_check_passed = False
        else:
            # Perform consistency check
            if min_max_value[0] > min_max_value[1]:
                inconsistency_report = (
                    inconsistency_report + f"    In list {parameter_name}: Minimum value {min_max_value[0]} " + f"is greater than {min_max_value[1]}!\n")
                is_check_passed = False

            # Perform the boundary check

            #  Check minimum boundary
            if check_type_minimum == CheckCondition.check_exclusive:
                if min_max_value[0] <= minimum:
                    inconsistency_report = inconsistency_report + f"    In list {parameter_name} the minimum entry value {min_max_value[0]} "
                    inconsistency_report = inconsistency_report + f"is less equal boundary value {minimum}!\n"
                    is_check_passed = False

            elif check_type_minimum == CheckCondition.check_inclusive:
                if min_max_value[0] < minimum:
                    inconsistency_report = inconsistency_report + f"    In list {parameter_name} the minimum entry value {min_max_value[0]} "
                    inconsistency_report = inconsistency_report + f"is less than boundary value {minimum}!\n"
                    is_check_passed = False

            # Check maximum boundary
            if check_type_maximum == CheckCondition.check_exclusive:
                if min_max_value[1] >= maximum:
                    inconsistency_report = inconsistency_report + f"    In list {parameter_name} the maximum entry value {min_max_value[1]} "
                    inconsistency_report = inconsistency_report + f"is greater equal boundary value {maximum}!\n"
                    is_check_passed = False

            elif check_type_maximum == CheckCondition.check_inclusive:
                if min_max_value[1] > maximum:
                    inconsistency_report = inconsistency_report + f"    In list {parameter_name} the maximum entry value {min_max_value[1]} "
                    inconsistency_report = inconsistency_report + f"is greater than boundary value {maximum}!\n"
                    is_check_passed = False

        return is_check_passed, inconsistency_report

    @staticmethod
    def check_dictionary(keyword_dictionary: dict, keyword: str, keyword_list_name: str) -> tuple[bool, str]:
        """
        Check the keyword according match in keyword dictionary.

        :param keyword_dictionary: Dictionary with keywords
        :type  keyword_dictionary: dict
        :param keyword: Keyword to check
        :type  keyword: str
        :param keyword_list_name: Name of keyword to mention in inconsistency report, if check fails
        :type  keyword_list_name: str
        :return: tuple: Indication if the verification failed | Error text with description about the deviation
        :rtype: tuple[bool, str]
        """
        # Variable declaration
        is_check_passed: bool = True
        inconsistency_report: str = ""

        # Check the consistency of the input parameter itself
        if len(keyword_dictionary) == 0:
            inconsistency_report = "    Dictionary is empty!\n"
            is_check_passed = False

        if keyword == "":
            inconsistency_report = "    Keyword is empty!\n"
            is_check_passed = False

        if is_check_passed:
            # Check if keyword matches keyword list
            is_matched = keyword in keyword_dictionary
            if not is_matched:
                inconsistency_report = f"    Keyword '{keyword}' in {keyword_list_name} does not match any keyword within dictionary:\n"
                inconsistency_report = inconsistency_report + f"    {keyword_dictionary.keys()}!\n"
                is_check_passed = False

        return is_check_passed, inconsistency_report
