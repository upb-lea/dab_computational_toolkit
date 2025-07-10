"""Unit tests for class dctmainctl."""

# python libraries
import pytest
from _pytest.logging import LogCaptureFixture
import os
import shutil
from typing import Any
import tempfile
import logging

# 3rd party libraries

# own libraries
import dct

# Enable logger
# pytestlogger = logging.getLogger(__name__)

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

# @pytest.mark.skip(reason="Test1 is skipped")
# test parameter list
@pytest.mark.parametrize("tst_toml_data, is_path_existing, exp_is_result_valid, exp_toml_object", [
    # --invalid inputs----
    # Path does not exist
    (tst_valid_toml, False, False, {}),
    # Input file does not exist
    ("", True, False, {}),
    # Input file, with wrong format
    (tst_invalid_toml, True, False, {}),
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
propagate=1

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

# Invalid toml file
tst_invalid_logger_toml = """
[nix]
keys=root,dct,transistordatabase,femmt,hct

[nor]
keys=console

[formatters]
keys=simple

[logger_badkey]
level=INFO
handlers=console
"""

# Expected log messages
#"Path /tmp/tmp28rwxnbb/invalid_path does not exists!"
#"Path /tmp/tmpf6wmd2ku/NotExistingFolder does not exists!"
#"Generate a new logging.conf file."
# @pytest.mark.skip(reason="Test2 is skipped")
# test parameter list
@pytest.mark.parametrize("tst_toml_data, is_path_existing, is_file_existing, restore_logger_flag, exp_message_id", [
    # --invalid inputs----
    # Path does not exist
    (tst_valid_logger_toml, False, False, False, 1),
    # Input file, with wrong format
    (tst_invalid_logger_toml, True, True, False, 3),
    # Input file does not exist
    (tst_valid_logger_toml, True, False, False, 2),
    # --valid inputs----
    # Valid input file exists
    (tst_valid_logger_toml, True, True, True, 0)
])

# definition of the testfunction
def test_load_generate_logging_config(caplog: LogCaptureFixture, tst_toml_data: str, is_path_existing: bool, is_file_existing: bool, restore_logger_flag: bool, exp_message_id: int):
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
    # Prepare the setup
    # Create path
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create the file-path name
        valid_filepath = os.path.join(tmpdir, "loggerconfig.toml")
        # Check for valid file flag
        if is_file_existing:
            # Store the logger configuration data
            with open(valid_filepath, "w") as f:
                f.write(tst_toml_data)
            # Check for valid path flag
        if is_path_existing:
            filepath = valid_filepath
        else:
            filepath = os.path.join(tmpdir,"NotExistingFolder","NotExistigFile.toml")

        # Perform the test
        # Check if expected test result is no error
        # Perform the test
        with caplog.at_level(logging.INFO):
            if restore_logger_flag:
                tst_dct.load_generate_logging_config(filepath)
                caplog.set_level(logging.INFO)
            else:
                tst_dct.load_generate_logging_config(filepath)

            # Expected messages
            expected_message = ["",
                                f"Path {os.path.join(tmpdir,"NotExistingFolder")} does not exists!",
                                "Generate a new logging.conf file.",
                                f"Logging configuration file {valid_filepath} is inconsistent."]
            if len(caplog.records) > 0:
                assert expected_message[exp_message_id] == caplog.records[0].message
            else:
                assert expected_message[exp_message_id] == ""

#########################################################################################################
# test of delete_study_content
#########################################################################################################

# Valid toml file
tst_ascii_file_data = "This is a test file"

# @pytest.mark.skip(reason="Test2 is skipped")
# test parameter list
@pytest.mark.parametrize("is_path_existing, is_file_existing, exp_message_id", [
    # --invalid inputs----
    # Path does not exist
    (False, False, 1),
    # --valid inputs----
    # Input file does not exist
    (True, False, 2),
    # Valid input file exists
    (True, True, 0)
])

# definition of the testfunction
def test_delete_study_content(caplog: LogCaptureFixture, is_path_existing: bool, is_file_existing: bool, exp_message_id: int):
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
    # Prepare the setup
    # Create path
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create filenames
        study_asc_name = "study_data.txt"
        study_bin_name = "study_data.sqlite3"
        # Create the pathname
        valid_asc_source = os.path.join(tmpdir, study_asc_name)
        valid_bin_source = os.path.join(tmpdir, study_bin_name)
        # Create 2 files
        # Ascii file
        with open(valid_asc_source, "w") as f:
            f.write(tst_ascii_file_data)
        # Binary file (1kb random data)
        with open(valid_bin_source, "wb") as f:
            f.write(os.urandom(1024))
        # Create subfolders (3 levels) and store files within each level
        actual_dir = tmpdir
        for level in ["level1", "level2", "level3"]:
            actual_dir = os.path.join(actual_dir, level)
            os.makedirs(actual_dir, exist_ok=True)
            # Copy ascii file
            target_file = os.path.join(actual_dir, study_asc_name)
            shutil.copy(valid_asc_source, target_file)
            # Copy binary file
            target_file = os.path.join(actual_dir, study_bin_name)
            shutil.copy(valid_bin_source, target_file)

        # Assign pathname
        if is_path_existing:
            path_name = tmpdir
        else:
            path_name = os.path.join(tmpdir,"NotExistingFolder")

        # Check, if files not exists
        if not is_file_existing:
            # Delete the files at tmpdir
            os.remove(valid_asc_source)
            os.remove(valid_bin_source)

        # Perform the test
        # Check if expected test result is no error
        # Perform the test
        with caplog.at_level(logging.INFO):
            tst_dct.delete_study_content(path_name,os.path.splitext(study_bin_name)[0])
            # Expected messages
            expected_message = ["",
                                f"Path {path_name} does not exists!",
                                f"File of study {os.path.splitext(study_bin_name)[0]} does not exists in {path_name}!"]

            if len(caplog.records) > 0:
                assert expected_message[exp_message_id] == caplog.records[0].message
            else:
                assert expected_message[exp_message_id] == ""

            # Check, if the folders are deleted, if the file exists
            if is_path_existing:
                assert len(os.listdir(path_name)) == 0

#########################################################################################################
# test of check_study_data
#########################################################################################################


#########################################################################################################
# test of get_nb_of_pkl_files
#########################################################################################################


#########################################################################################################
# test of check_breakpoint
#########################################################################################################


#########################################################################################################
# test of circuit_toml_2_dto
#########################################################################################################



#########################################################################################################
# test of generate_zip_archive
#########################################################################################################


#########################################################################################################
# test of get_init_queue_data
#########################################################################################################






