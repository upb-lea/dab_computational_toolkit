"""Unit tests for class dctmainctl."""

# python libraries
import pytest
from _pytest.logging import LogCaptureFixture
import os
import shutil
import tempfile
import logging

# 3rd party libraries

# own libraries
import dct

# Enable logger
pytestlogger = logging.getLogger(__name__)

#########################################################################################################
# test of load_toml_file
#########################################################################################################

# Testresults
exp_config = {'integer_value': {'test_value': 10}, 'string': {'test_string': 'Hello'}}

# Invalid toml file
test_invalid_toml = """
string
test_string = "Hello"

[integer_value]
test_value = 10
"""
# Valid toml file
test_valid_toml = """
[string]
test_string = "Hello"

[integer_value]
test_value = 10
"""

# Skip file command
# @pytest.mark.skip(reason="Test of load_toml_file is skipped")
# test parameter list
@pytest.mark.parametrize("test_toml_data, is_path_existing, is_exp_result_valid, exp_toml_object, exp_message_id", [
    # --invalid inputs----
    # Path does not exist
    (test_valid_toml, False, False, {}, 1),
    # Input file does not exist
    ("", True, False, {}, 2),
    # Input file, with wrong format
    (test_invalid_toml, True, False, {}, 3),
    # --valid inputs----
    # input file, with valid format
    (test_valid_toml, True, True, exp_config, 0)
])
# Unit test function
def test_load_toml_file(caplog: LogCaptureFixture, test_toml_data: str, is_path_existing: bool,
                        is_exp_result_valid: bool, exp_toml_object: dict, exp_message_id: int) -> None:
    """Test the method load_toml_file.

    :param caplog: class instance for logger data
    :type  caplog: LogCaptureFixture
    :param test_toml_data: Test data of the toml file
    :type  test_toml_data: str
    :param is_path_existing: Flag to indicate, if the path exists for the test
    :type  is_path_existing: bool
    :param is_exp_result_valid: Flag to indicate, if the expected result is valid
    :type  is_exp_result_valid: bool
    :param exp_toml_object: Configuration data load from the toml-file
    :type  exp_toml_object: dict
    :param exp_message_id: List index of the expected message
    :type  exp_message_id: int
    """
    # Variable declaration
    # Create the instance
    test_dct: dct.DctMainCtl = dct.DctMainCtl()
    invalid_path_name = ""

    with tempfile.TemporaryDirectory() as tmpdir:
        valid_filepath = os.path.join(tmpdir, "config.toml")
        # Check for invalid path flag
        with open(valid_filepath, "w") as f:
            f.write(test_toml_data)
        invalid_filepath = os.path.join(tmpdir, "not_exist.toml")
        # Check for empty content
        if not test_toml_data == "":
            if not is_path_existing:
                invalid_path_name = os.path.join(tmpdir, "invalid_path")
                filepath = os.path.join(invalid_path_name, "config.toml")
            else:
                filepath = valid_filepath
        else:
            filepath = invalid_filepath

        # Perform the test
        with caplog.at_level(logging.INFO):
            is_result_valid, toml_object = test_dct.load_toml_file(filepath)

            # Expected messages
            expected_message = ["",
                                f"Path {invalid_path_name} does not exists!",
                                f"File {filepath} does not exists!",
                                "toml-file is not conform to toml-format:\nExpected '=' after a key in a key/value pair (at line 2, column 7)"]

            # Evaluate the result
            if len(caplog.records) > 0:
                assert caplog.records[0].message == expected_message[exp_message_id]
            else:
                assert "" == expected_message[exp_message_id]

            assert is_result_valid == is_exp_result_valid
            assert toml_object == exp_toml_object


#########################################################################################################
# test of load_generate_logging_config
#########################################################################################################

# Valid toml file
test_valid_logger_toml = """
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
test_invalid_logger_toml = """
[nix]
keys=root,dct,transistordatabase,femmt,hct

[nor]
keys=console

[formatters]
keys=simple

[logger_bad_key]
level=INFO
handlers=console
"""

# Skip file command
# @pytest.mark.skip(reason="Test of load_generate_logging_config is skipped")
# test parameter list
@pytest.mark.parametrize("test_toml_data, is_path_existing, is_file_existing, is_logger_to_restore, exp_message_id", [
    # --invalid inputs----
    # Path does not exist
    (test_valid_logger_toml, False, False, False, 1),
    # Input file, with wrong format
    (test_invalid_logger_toml, True, True, False, 3),
    # Input file does not exist
    (test_valid_logger_toml, True, False, False, 2),
    # --valid inputs----
    # Valid input file exists
    (test_valid_logger_toml, True, True, True, 0)
])
# Unit test function
def test_load_generate_logging_config(caplog: LogCaptureFixture, test_toml_data: str, is_path_existing: bool, is_file_existing: bool,
                                      is_logger_to_restore: bool, exp_message_id: int) -> None:
    """Test method load_generate_logging_config(logging_config_file: str) -> None: according values.

    :param caplog: class instance for logger data
    :type  caplog: LogCaptureFixture
    :param test_toml_data: Test data to load by the method
    :type  test_toml_data: str
    :param is_path_existing: Flag to indicate, if the path exists for the test
    :type  is_path_existing: bool
    :param is_file_existing: Flag to indicate, if the file exists for the test
    :type  is_file_existing: bool
    :param is_logger_to_restore: Flag to indicate, if the logger needs to be restored after test
    :type  is_logger_to_restore: bool
    :param exp_message_id: List index of the expected message
    :type  exp_message_id: int
    """
    # Variable declaration
    # Create the instance
    test_dct: dct.DctMainCtl = dct.DctMainCtl()

    # Prepare the setup
    # Create path
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create the file-path name
        valid_filepath = os.path.join(tmpdir, "logger_config.toml")
        # Check for valid file flag
        if is_file_existing:
            # Store the logger configuration data
            with open(valid_filepath, "w") as f:
                f.write(test_toml_data)
            # Check for valid path flag
        if is_path_existing:
            filepath = valid_filepath
        else:
            filepath = os.path.join(tmpdir, "not_existing_folder", "not_existing_file.toml")

        # Perform the test
        with caplog.at_level(logging.INFO):
            if is_logger_to_restore:
                test_dct.load_generate_logging_config(filepath)
                caplog.set_level(logging.INFO)
            else:
                test_dct.load_generate_logging_config(filepath)

            # Expected messages
            expected_message = ["",
                                f"Path {os.path.join(tmpdir, "not_existing_folder")} does not exists!",
                                "Generate a new logging.conf file.",
                                f"Logging configuration file {valid_filepath} is inconsistent."]

            if len(caplog.records) > 0:
                assert caplog.records[0].message == expected_message[exp_message_id]
            else:
                assert "" == expected_message[exp_message_id]


#########################################################################################################
# test of delete_study_content
#########################################################################################################

# Valid toml file
test_ascii_file_data = "This is a test file"

# Skip file command
# @pytest.mark.skip(reason="Test of delete_study_content is skipped")
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
# Unit test function
def test_delete_study_content(caplog: LogCaptureFixture, is_path_existing: bool, is_file_existing: bool, exp_message_id: int) -> None:
    """Test method load_generate_logging_config(logging_config_file: str) -> None: according values.

    :param caplog: class instance for logger data
    :type  caplog: LogCaptureFixture
    :param is_path_existing: Flag to indicate, if the path exists for the test
    :type  is_path_existing: bool
    :param is_file_existing: Flag to indicate, if the file exists for the test
    :type  is_file_existing: bool
    :param exp_message_id: List index of the expected message
    :type  exp_message_id: int
    """
    # Variable declaration
    # Create the instance
    test_dct: dct.DctMainCtl = dct.DctMainCtl()

    # Prepare the setup
    # Create path
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create filenames
        study_asc_name = "study_data.text"
        study_bin_name = "study_data.sqlite3"
        # Create the pathname
        valid_asc_source = os.path.join(tmpdir, study_asc_name)
        valid_bin_source = os.path.join(tmpdir, study_bin_name)
        # Create 2 files
        # Ascii file
        with open(valid_asc_source, "w") as f:
            f.write(test_ascii_file_data)
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
            path_name = os.path.join(tmpdir, "not_existing_folder")

        # Check, if files not exists
        if not is_file_existing:
            # Delete the files at tmpdir
            os.remove(valid_asc_source)
            os.remove(valid_bin_source)

        # Perform the test
        with caplog.at_level(logging.INFO):
            test_dct.delete_study_content(path_name, os.path.splitext(study_bin_name)[0])
            # Expected messages
            expected_message = ["",
                                f"Path {path_name} does not exists!",
                                f"File of study {os.path.splitext(study_bin_name)[0]} does not exists in {path_name}!"]

            if len(caplog.records) > 0:
                assert caplog.records[0].message == expected_message[exp_message_id]
            else:
                assert "" == expected_message[exp_message_id]

            # Check, if the folders are deleted, if the file exists
            if is_path_existing:
                assert len(os.listdir(path_name)) == 0

#########################################################################################################
# test of check_study_data
#########################################################################################################

# Skip file command
# @pytest.mark.skip(reason="Test of check_study_data is skipped")
# test parameter list
@pytest.mark.parametrize("is_path_existing, is_file_existing, file_name, exp_message_id, exp_result", [
    # --invalid inputs----
    # Path does not exist
    (False, False, "study_data.sqlite3", 1, False),
    # File does not exists
    (True, False, "study_data.sqlite3", 2, False),
    # Filename is wrong
    (True, True, "wrong_study_data.sqlite3", 2, False),
    # Filename extension is wrong
    (True, True, "study_data.text", 2, False),
    # --valid inputs----
    # Input file does not exist
    (True, True, "study_data.sqlite3", 0, True),
])
# Unit test function
def test_check_study_data(caplog: LogCaptureFixture, is_path_existing: bool, is_file_existing: bool, file_name: str,
                          exp_message_id: int, exp_result: bool) -> None:
    """Test method load_generate_logging_config(logging_config_file: str) -> None: according values.

    :param caplog: class instance for logger data
    :type  caplog: LogCaptureFixture
    :param is_path_existing: Flag to indicate, if the path exists for the test
    :type  is_path_existing: bool
    :param is_file_existing: Flag to indicate, if the file exists for the test
    :type  is_file_existing: bool
    :param file_name: Name of the file, which shall be used
    :type  file_name: str
    :param exp_message_id: List index of the expected message
    :type  exp_message_id: int
    :param exp_result: Expected result of the method
    :type  exp_result: bool
    """
    # Variable declaration
    # Create the instance
    test_dct: dct.DctMainCtl = dct.DctMainCtl()
    target_study_name = "study_data"

    # Prepare the setup
    # Create path
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create the file pathname
        valid_file_source = os.path.join(tmpdir, file_name)
        # Check, if file exists
        if is_file_existing:
            with open(valid_file_source, "wb") as f:
                f.write(os.urandom(1024))
        # Assign pathname
        if is_path_existing:
            path_name = tmpdir
        else:
            path_name = os.path.join(tmpdir, "not_existing_folder")

        path_file_name = os.path.join(path_name, f"{target_study_name}.sqlite3")

        # Perform the test
        with caplog.at_level(logging.INFO):
            test_result = test_dct.check_study_data(path_name, target_study_name)
            # Expected messages
            expected_message = ["",
                                f"Path {path_name} does not exists!",
                                f"File {path_file_name} does not exists!"]

            if len(caplog.records) > 0:
                assert caplog.records[0].message == expected_message[exp_message_id]
            else:
                assert "" == expected_message[exp_message_id]

            # Check the result of the method
            assert test_result == exp_result

#########################################################################################################
# test of get_number_of_pkl_files
#########################################################################################################

# Skip file command
# @pytest.mark.skip(reason="Test of get_number_of_pkl_files is skipped")
# test parameter list
@pytest.mark.parametrize("is_path_existing, number_of_pkl_files, prefix, exp_message_id", [
    # --invalid usecase----
    # Path does not exist
    (False, 0, "", 1),
    # --valid usecases----
    # No pkl-file exists
    (True, 0, "", 0),
    # 10 pkl files without prefix
    (True, 10, "", 0),
    # 20 pkl files with prefix
    (True, 10, "C_", 0),

])
# Unit test function
def test_get_number_of_pkl_files(caplog: LogCaptureFixture, is_path_existing: bool, number_of_pkl_files: int, prefix: str, exp_message_id: int) -> None:
    """Test method load_generate_logging_config.

    :param caplog: class instance for logger data
    :type  caplog: LogCaptureFixture
    :param is_path_existing: Flag to indicate, if the path exists for the test
    :type  is_path_existing: bool
    :param number_of_pkl_files: Number of pkl-files
    :type  number_of_pkl_files: int
    :param prefix: File prefix
    :type  prefix: str
    :param exp_message_id: List index of the expected message
    :type  exp_message_id: int
    """
    # Variable declaration
    # Create the instance
    test_dct: dct.DctMainCtl = dct.DctMainCtl()
    target_study_name = "study_data"

    # Prepare the setup
    # Create path
    with tempfile.TemporaryDirectory() as tmpdir:
        # Check, if path is valid and minimum one file is requested
        if is_path_existing:
            path_name = tmpdir
            if number_of_pkl_files > 0:
                # Create requested number of files
                for file_number in range(number_of_pkl_files):
                    # Create the file pathname
                    file_path_name = os.path.join(path_name, f"{prefix}{file_number}.pkl")
                    # Create and store the file
                    with open(file_path_name, "wb") as f:
                        f.write(os.urandom(1024))
        else:
            path_name = os.path.join(tmpdir, "not_existing_folder")

        # Perform the test
        with caplog.at_level(logging.INFO):
            test_result = test_dct.get_number_of_pkl_files(path_name)
            # Expected messages
            expected_message = ["",
                                f"Path {path_name} does not exists!"]

            if len(caplog.records) > 0:
                assert caplog.records[0].message == expected_message[exp_message_id]
            else:
                assert "" == expected_message[exp_message_id]

            # Check the result of the method
            assert test_result == number_of_pkl_files

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
