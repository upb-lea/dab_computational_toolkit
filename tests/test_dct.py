"""Unit tests for class dctmainctl."""

# python libraries
import pytest
from _pytest.logging import LogCaptureFixture
import os
from typing import Optional, Any
import tempfile
import logging

# 3rd party libraries
import numpy as np
import numpy.testing

# own libraries
import dct

# Enable logger
pytestlogger = logging.getLogger(__name__)

#########################################################################################################
# test of load_toml_file
#########################################################################################################

# Testresults
exp_config = {'integervalue': {'tst_value': 10}, 'string': {'tst_string': 'Hello'}}

# Invalid toml file
tst_invalid_toml = """
string
tst_string = "Hello"

[integervalue]
tst_value = 10
"""
# Valid toml file
tst_valid_toml = """
[string]
tst_string = "Hello"

[integervalue]
tst_value = 10
"""

# parameterset for values
@pytest.mark.parametrize("tst_toml_data, is_path_existing, exp_is_result_valid, exp_toml_object", [
    # --invalid inputs----
    # Path does not exist
    (tst_valid_toml, False, False, None),
    # Input file does not exist
    ("", True, False, None),
    # Input file, with wrong format
    (tst_invalid_toml, True, False, None),
    # --valid inputs----
    # input file, with valid format
    (tst_valid_toml, True, True, exp_config)
])
# definition of the testfunction
def test_load_toml_file(tst_toml_data: str, is_path_existing: bool, exp_is_result_valid: bool, exp_toml_object: Any):
    """Test method load_toml_file(self, toml_file: str) -> tuple[bool, dict]: according values.

    :param tst_toml_data: Toml-file input
    :type  tst_toml_data: str
    :param is_path_existing: Flag to generate existing file path name
    :type  is_path_existing: bool
    :param exp_is_result_valid: Expected return value: Is result valid
    :type  exp_is_result_valid: bool
    :param exp_toml_object: Expected return data
    :type  exp_toml_object: Any
    """

    # Create the instance
    tst_dct: dct.DctMainCtl = dct.DctMainCtl()
    with tempfile.TemporaryDirectory() as tmpdir:
        valid_filepath = os.path.join(tmpdir, "config.toml")
        # Check for invalid path flag
        with open(valid_filepath, "w") as f:
            f.write(tst_toml_data)
        invalid_filepath =  os.path.join(tmpdir, "not_exist.toml")
        # Check for empty content
        if not tst_toml_data == "":
            if not is_path_existing:
                filepath = os.path.join(tmpdir,"invalid_path","config.toml")
            else:
                filepath = valid_filepath
        else:
            filepath = invalid_filepath
        # Perform the test
        is_result_valid, toml_object = tst_dct.load_toml_file(filepath)

    # Evaluate the result
    assert is_result_valid == exp_is_result_valid
    assert toml_object == exp_toml_object



#########################################################################################################
# test of load_generate_logging_config
#########################################################################################################

# Valid toml file
tst_valid_logger_toml = """
[loggers]
keys=root,dct,transistordatabase,femmt,hct

[handlers]
keys=console

[formatters]
keys=simple

[logger_root]
level=INFO
handlers=console
qualname=

[logger_dct]
level=INFO
handlers=console
qualname=dct
propagate=0

[logger_transistordatabase]
level=INFO
handlers=console
qualname=transistordatabase
propagate=0

[logger_femmt]
level=INFO
handlers=console
qualname=femmt
propagate=0

[logger_hct]
level=INFO
handlers=console
qualname=hct
propagate=0

[handler_console]
class=StreamHandler
level=NOTSET
formatter=simple
args=(sys.stdout,)

[formatter_simple]
format=%(asctime)s - %(name)s - %(levelname)s - %(message)s
datefmt=%Y-%m-%d %H:%M:%S
"""
# Expected log messages
'Path /tmp/tmp28rwxnbb/invalid_path does not exists!'

@pytest.mark.parametrize("tst_toml_data, is_path_existing, exp_error_type, error_flag", [
    # --invalid inputs----
    # Path does not exist
    (tst_valid_logger_toml, False, None, False),
    # Input file, with wrong format
    (tst_valid_toml, True, KeyError, True),
    # --valid inputs----
    # Input file does not exist
    ("", True, None, False),
    # Input file exists
    (tst_valid_logger_toml, True, None, False)
])

# definition of the testfunction
def test_load_generate_logging_config(caplog: LogCaptureFixture, tst_toml_data: str, is_path_existing: bool, exp_error_type: Any, error_flag: bool):
    """Test method load_generate_logging_config(logging_config_file: str) -> None: according values.

    :param tst_toml_data: Toml-file input
    :type  tst_toml_data: str
    :param is_path_existing: Flag to generate existing file path name
    :type  is_path_existing: bool
    :param exp_error_type: Kind of error in case of an error
    :type  exp_error_type: Any
    :param error_flag: flag to indicate, if an error is raised by the method under test
    :type  error_flag: bool
    """

    # Create the instance
    tst_dct: dct.DctMainCtl = dct.DctMainCtl()
    with tempfile.TemporaryDirectory() as tmpdir:
        valid_filepath = os.path.join(tmpdir, "config.toml")
        # Check for invalid path flag
        with open(valid_filepath, "w") as f:
            f.write(tst_toml_data)
        invalid_filepath =  os.path.join(tmpdir, "not_exist.toml")
        # Check for empty content
        if not tst_toml_data == "":
            if not is_path_existing:
                filepath = os.path.join(tmpdir,"invalid_path","config.toml")
            else:
                filepath = valid_filepath
        else:
            filepath = invalid_filepath

        # Check if expected test result is no error
        if error_flag is False:
            # Perform the test
            with caplog.at_level(logging.INFO):
                tst_dct.load_generate_logging_config(filepath)
                # Expected messages
                expected_message = [f"Found existing logging configuration {filepath}.",
                                    f"Path {os.path.join(tmpdir,"invalid_path")} does not exists!",
                                    "Generate a new logging.conf file."]
                assert expected_message[1] == caplog.records[0].message
        else:  # generate_channel raises an error
            with pytest.raises(exp_error_type):
                tst_dct.load_generate_logging_config(filepath)

