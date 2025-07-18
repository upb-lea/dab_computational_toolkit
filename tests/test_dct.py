"""Unit tests for class dctmainctl."""

# python libraries
import os
import sys
import shutil
import tempfile
import logging
import threading
import time
import copy

# 3rd party libraries
import pytest
from _pytest.logging import LogCaptureFixture
from _pytest.capture import CaptureFixture
import zipfile

# own libraries
import dct
import dct.toml_checker as tc

# Enable logger
pytestlogger = logging.getLogger(__name__)

# Global test parameter

# FlowControl base parameter set
test_FlowControl_base: tc.FlowControl = tc.FlowControl(
    general=tc.General(project_directory="Hallo"),
    breakpoints=tc.Breakpoints(circuit_pareto="no",
                               circuit_filtered="no",
                               inductor="no",
                               transformer="no",
                               heat_sink="no",
                               summary="no"),
    conditional_breakpoints=tc.CondBreakpoints(
        circuit=1,
        inductor=2,
        transformer=3,
        heat_sink=1),
    circuit=tc.Circuit(number_of_trials=1,
                       calculation_mode="continue",
                       subdirectory="dummy"),
    inductor=tc.Inductor(number_of_trials=1,
                         calculation_mode="continue",
                         subdirectory="dummy"),
    transformer=tc.Transformer(number_of_trials=1,
                               calculation_mode="continue",
                               subdirectory="dummy"),
    heat_sink=tc.HeatSink(number_of_trials=1,
                          calculation_mode="continue",
                          subdirectory="dummy"),
    summary=tc.Summary(calculation_mode="new",
                       subdirectory="dummy"),
    configuration_data_files=tc.ConfigurationDataFiles(
        circuit_configuration_file="dummy",
        inductor_configuration_file="dummy",
        transformer_configuration_file="dummy",
        heat_sink_configuration_file="dummy")
)

#########################################################################################################
# test of load_toml_file
#########################################################################################################

# Test results
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

# test parameter list
@pytest.mark.parametrize("test_toml_data, is_path_existing, is_file_existing, is_logger_to_restore, exp_message_id", [
    # --invalid input parameter----
    # Path does not exist
    (test_valid_logger_toml, False, False, False, 1),
    # Input file, with wrong format
    (test_invalid_logger_toml, True, True, False, 3),
    # Input file does not exist
    (test_valid_logger_toml, True, False, False, 2),
    # --valid input parameter----
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

            # Create wrong path string
            wrong_path = os.path.join(tmpdir, "not_existing_folder")
            # Expected messages
            expected_message = ["",
                                f"Path {wrong_path} does not exists!",
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

# test parameter list
@pytest.mark.parametrize("is_path_existing, is_file_existing, exp_message_id", [
    # --invalid input parameter----
    # Path does not exist
    (False, False, 1),
    # --valid input parameter----
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
        # text file
        with open(valid_asc_source, "w") as f:
            f.write(test_ascii_file_data)
        # Binary file (1 kilo byte random data)
        with open(valid_bin_source, "wb") as f:
            f.write(os.urandom(1024))
        # Create sub folders (3 levels) and store files within each level
        actual_dir = tmpdir
        for level in ["level1", "level2", "level3"]:
            actual_dir = os.path.join(actual_dir, level)
            os.makedirs(actual_dir, exist_ok=True)
            # Copy test file
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
            # Delete the files at temporary folder
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

# test parameter list
@pytest.mark.parametrize("is_path_existing, is_file_existing, file_name, exp_message_id, exp_result", [
    # --invalid input parameter----
    # Path does not exist
    (False, False, "study_data.sqlite3", 1, False),
    # File does not exists
    (True, False, "study_data.sqlite3", 2, False),
    # Filename is wrong
    (True, True, "wrong_study_data.sqlite3", 2, False),
    # Filename extension is wrong
    (True, True, "study_data.text", 2, False),
    # --valid input parameter----
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

# test parameter list
@pytest.mark.parametrize("is_path_existing, number_of_pkl_files, prefix, exp_message_id", [
    # --invalid input parameter----
    # Path does not exist
    (False, 0, "", 1),
    # --valid input parameter----
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
# check_breakpoint(self, break_point_key: str, info: str)

# test parameter list
@pytest.mark.parametrize("keyword, message, keyboard_input, expected_message, is_program_exit, is_wait_for_input", [
    # --invalid use case----
    # Invalid keyword
    ("invalid_keyword", "Message not to return", "", "", False, False),
    # --valid use cases----
    # Valid keyword no - Breakpoint will not wait for an input
    ("no", "message string (no)", "", "", False, False),
    # Valid keyword stop - Breakpoint leads to terminate program
    ("stop", "message string (stop)", "", "Program stops cause by breakpoint at: 'message string (stop)'!", True, False),
    # Valid keyword pause - Breakpoint will leads to pause and continue with valid keyboard input
    ("pause", "message string (pause-c)", "c",
     "Active breakpoint at: 'message string (pause-c)'!\n\n'C'=continue, 'S'=stop the program. Please enter your choice", False, True),
    # Valid keyword pause - Breakpoint leads to pause and continue with valid keyboard input
    ("pause", "message string (pause-C)", "C",
     "Active breakpoint at: 'message string (pause-C)'!\n\n'C'=continue, 'S'=stop the program. Please enter your choice", False, True),
    # Valid keyword pause - Breakpoint leads to pause and terminated with valid keyboard input
    ("pause", "message string (pause-s)", "s",
     "Active breakpoint at: 'message string (pause-s)'!\n\n'C'=continue, 'S'=stop the program. Please enter your choice\nUser stops the program!", True, True),
    # Valid keyword pause - Breakpoint leads to pause and terminated with valid keyboard input
    ("pause", "message string (pause-S)", "S",
     "Active breakpoint at: 'message string (pause-S)'!\n\n'C'=continue, 'S'=stop the program. Please enter your choice\nUser stops the program!", True, True)
])
# Unit test function
def test_check_breakpoint(capsys: CaptureFixture[str], keyword: str, message: str, keyboard_input: str,
                          expected_message: str, is_program_exit: bool, is_wait_for_input: bool) -> None:
    """Test method load_generate_logging_config(logging_config_file: str) -> None: according values.

    :param capsys: class instance for print data
    :type  capsys: CaptureFixture
    :param keyword: keyword to control the method under test
    :type  keyword: str
    :param message: Input message, which is used to create the output message
    :type  message: str
    :param keyboard_input: String, which simulates the keyboard input
    :type  keyboard_input: str
    :param expected_message: expected message string
    :type  expected_message: str
    :param is_program_exit: Flag to indicate, if the program terminates
    :type  is_program_exit: bool
    :param is_wait_for_input: Flag to indicate, if the function waits for keyboard input
    :type  is_wait_for_input: bool
    """
    # Variable declaration
    test_thread: threading.Thread
    is_exit: bool = False
    exit_code: int = -1

    # Create the instance
    test_dct: dct.DctMainCtl = dct.DctMainCtl()

    def test_thread_container():
        nonlocal is_exit
        nonlocal exit_code
        try:
            test_dct.check_breakpoint(keyword, message)
        except SystemExit as e:
            is_exit = True
            exit_code = e.code

    # Setup the thread
    test_thread = threading.Thread(target=test_thread_container)
    test_thread.daemon = True

    # Perform the test
    test_thread.start()
    time.sleep(0.2)

    # Check if the method waits for input
    if is_wait_for_input:
        assert test_thread.is_alive()
        test_dct._key_input_string = keyboard_input
        test_thread.join(timeout=2)

    # Read out the print log
    print_data = capsys.readouterr().out
    # Test on finish thread
    assert not test_thread.is_alive()
    # Check if thread is finished
    if not test_thread.is_alive():
        assert print_data.strip() == expected_message

    # Check if program termination is requested
    if is_program_exit:
        assert is_exit
        assert exit_code is None
    else:
        assert not is_exit
        assert exit_code == -1

#########################################################################################################
# test of circuit_toml_2_dto
#########################################################################################################

# test parameter list (counter)
@pytest.mark.parametrize("test_index", list(range(10)))
# Unit test function
def test_circuit_toml_2_dto(test_index: int) -> None:
    """Test the method load_toml_file.

    :param test_index: Test index of performed test is used as start index for test lists
    :type  test_index: int
    """
    # Variable declaration
    # Minimal value, in between value and maximal value
    float_test_values: list[float] = [-sys.float_info.max, -3.2, -1.2, -3.4e-12, -sys.float_info.min, 0]
    float_test_len = len(float_test_values)

    float_test_arrays: list[list[float]] = [[-3.2, -1.2],
                                            [0.0, 4.53337],
                                            [200.0, 24e10],
                                            [sys.float_info.min, 7e19],
                                            [-sys.float_info.max, sys.float_info.max],
                                            [-sys.float_info.max, -sys.float_info.min],
                                            [-sys.float_info.max, 0.0],
                                            [0.0, sys.float_info.max]]
    float_list_len = len(float_test_arrays)

    int_test_values: list[int] = [0, 1, -1, 42, sys.maxsize, -sys.maxsize - 1]
    int_test_len = len(int_test_values)

    int_test_arrays: list[list[int]] = [[-1, 1],
                                        [0, 2],
                                        [-sys.maxsize - 1, sys.maxsize],
                                        [-5, 990]
                                        ]
    int_list_len = len(int_test_arrays)

    string_test_values: list[str] = [
        bytes([65]).decode("ascii"),
        bytes([98, 57]).decode("ascii"),
        bytes([84, 101, 115, 116, 49, 50, 51]).decode("ascii"),
        bytes([120, 89, 122, 56, 57, 48]).decode("ascii"),
        bytes([54, 83, 101, 99, 117, 114, 101, 80, 97, 115, 115, 48, 49]).decode("ascii"),
        bytes([65, 108, 112, 104, 97, 66, 114, 97, 118, 111, 50, 48, 50, 52]).decode("ascii"),
        bytes([122, 90, 57, 121, 89, 56, 120, 88, 55]).decode("ascii"),
        bytes([55, 76, 111, 110, 103, 83, 116, 114, 105, 110, 103, 84, 101, 115, 116, 57, 57, 57]).decode("ascii")
    ]
    str_test_len = len(string_test_values)

    string_test_arrays: list[list[str]] = [
        [bytes([65]).decode("ascii"), bytes([98, 57]).decode("ascii")],
        [bytes([84, 101, 115, 116, 49, 50, 51]).decode("ascii"), bytes([120, 89, 122, 56, 57, 48]).decode("ascii")],
        [bytes([54, 83, 101, 99, 117, 114, 101, 80, 97, 115, 115, 48, 49]).decode("ascii"),
         bytes([65, 108, 112, 104, 97, 66, 114, 97, 118, 111, 50, 48, 50, 52]).decode("ascii")],
        [bytes([122, 90, 57, 121, 89, 56, 120, 88, 55]).decode("ascii"),
         bytes([55, 76, 111, 110, 103, 83, 116, 114, 105, 110, 103, 84, 101, 115, 116, 57, 57, 57]).decode("ascii")],
        [bytes([65]).decode("ascii"),
         bytes([55, 76, 111, 110, 103, 83, 116, 114, 105, 110, 103, 84, 101, 115, 116, 57, 57, 57]).decode("ascii")],
        [bytes([76, 111, 110, 103, 83, 116, 114, 105, 110, 103, 84, 101, 115, 116, 57, 57, 57]).decode("ascii"),
         bytes([66, 57]).decode("ascii")],
        [bytes([65, 108, 112, 104, 97, 66, 114, 97, 118, 111, 50, 48, 50, 52]).decode("ascii"),
         bytes([54, 83, 101, 99, 117, 114, 101, 80, 97, 115, 115, 48, 49]).decode("ascii")],
        [bytes([84, 101, 115, 116, 49, 50, 51]).decode("ascii"),
         bytes([55, 76, 111, 110, 103, 83, 116, 114, 105, 110, 103, 84, 101, 115, 116, 57, 57, 57]).decode("ascii")]
    ]
    str_list_len = len(string_test_arrays)

    str_sampling_methode_values: list[str] = ["latin_hypercube", "meshgrid"]
    str_sampling_len = len(str_sampling_methode_values)

    # Constant values of FlowControl
    test_parameter_2: tc.FlowControl = copy.deepcopy(test_FlowControl_base)

    # Result value
    result: dct.circuit_optimization_dtos.CircuitParetoDabDesign

    # Create the instance
    test_dct: dct.DctMainCtl = dct.DctMainCtl()

    # Initialize the test parameter
    test_parameter_1: tc.TomlCircuitParetoDabDesign = tc.TomlCircuitParetoDabDesign(
        design_space=tc.TomlCircuitParetoDesignSpace(
            f_s_min_max_list=float_test_arrays[test_index % float_list_len],
            l_1_min_max_list=float_test_arrays[(test_index + 1) % float_list_len],
            l_2__min_max_list=float_test_arrays[(test_index + 2) % float_list_len],
            l_s_min_max_list=float_test_arrays[(test_index + 3) % float_list_len],
            n_min_max_list=float_test_arrays[(test_index + 4) % float_list_len],
            transistor_1_name_list=string_test_arrays[test_index % str_list_len],
            transistor_2_name_list=string_test_arrays[(test_index + 1) % str_list_len],
            c_par_1=float_test_values[test_index % float_test_len],
            c_par_2=float_test_values[(test_index+1) % float_test_len]),
        output_range=tc.TomlCircuitOutputRange(
            v1_min_max_list=float_test_arrays[(test_index + 5) % float_list_len],
            v2_min_max_list=float_test_arrays[(test_index + 6) % float_list_len],
            p_min_max_list=float_test_arrays[test_index % float_list_len]),
        sampling=tc.TomlSampling(
            sampling_method=dct.SamplingEnum.latin_hypercube,
            sampling_points=int_test_values[test_index % int_test_len],
            sampling_random_seed=10,
            v1_additional_user_point_list=float_test_arrays[(test_index + 1) % float_list_len],
            v2_additional_user_point_list=float_test_arrays[(test_index + 2) % float_list_len],
            p_additional_user_point_list=float_test_arrays[test_index % float_list_len],
            additional_user_weighting_point_list=[0.5, 0.5]),
        filter_distance=tc.TomlCircuitFilterDistance(
            number_filtered_designs=int_test_values[(test_index + 1) % int_test_len],
            difference_percentage=float_test_values[(test_index + 3) % float_test_len]))

    test_parameter_2.general.project_directory = string_test_values[(test_index + 1) % str_test_len]
    test_string_circuit = string_test_values[(test_index + 2) % str_test_len]
    test_parameter_2.configuration_data_files.circuit_configuration_file = test_string_circuit+".toml"

    # Perform the test
    result = test_dct.circuit_toml_2_dto(test_parameter_1, test_parameter_2)

    # Check result
    assert result.design_space.f_s_min_max_list == test_parameter_1.design_space.f_s_min_max_list
    assert result.design_space.l_1_min_max_list == test_parameter_1.design_space.l_1_min_max_list
    assert result.design_space.l_2__min_max_list == test_parameter_1.design_space.l_2__min_max_list
    assert result.design_space.l_s_min_max_list == test_parameter_1.design_space.l_s_min_max_list
    assert result.design_space.n_min_max_list == test_parameter_1.design_space.n_min_max_list
    assert result.design_space.transistor_1_name_list == test_parameter_1.design_space.transistor_1_name_list
    assert result.design_space.transistor_2_name_list == test_parameter_1.design_space.transistor_2_name_list
    assert result.design_space.c_par_1 == test_parameter_1.design_space.c_par_1
    assert result.design_space.c_par_2 == test_parameter_1.design_space.c_par_2
    assert result.output_range.v1_min_max_list == test_parameter_1.output_range.v1_min_max_list
    assert result.output_range.v2_min_max_list == test_parameter_1.output_range.v2_min_max_list
    assert result.output_range.p_min_max_list == test_parameter_1.output_range.p_min_max_list
    assert result.sampling.sampling_method == test_parameter_1.sampling.sampling_method
    assert result.sampling.sampling_points == test_parameter_1.sampling.sampling_points
    assert result.sampling.v1_additional_user_point_list == test_parameter_1.sampling.v1_additional_user_point_list
    assert result.sampling.v2_additional_user_point_list == test_parameter_1.sampling.v2_additional_user_point_list
    assert result.sampling.p_additional_user_point_list == test_parameter_1.sampling.p_additional_user_point_list
    assert result.filter.number_filtered_designs == test_parameter_1.filter_distance.number_filtered_designs
    assert result.filter.difference_percentage == test_parameter_1.filter_distance.difference_percentage
    assert result.project_directory == test_parameter_2.general.project_directory
    assert result.circuit_study_name == test_string_circuit

#########################################################################################################
# test of generate_zip_archive
#########################################################################################################

# def generate_zip_archive(self, toml_prog_flow: tc.FlowControl) -> None:

# test parameter list
@pytest.mark.parametrize("is_path_existing, exp_message_id", [
    # --invalid input parameter----
    # Path does not exist
    (False, 1),
    # --valid input parameter----
    # Path exists
    (True, 2)
])
# Unit test function
def test_generate_zip_archive(caplog: LogCaptureFixture, is_path_existing: bool, exp_message_id: int) -> None:
    """Test method load_generate_logging_config(logging_config_file: str) -> None: according values.

    :param caplog: class instance for logger data
    :type  caplog: LogCaptureFixture
    :param is_path_existing: Flag to indicate, if the path exists for the test
    :type  is_path_existing: bool
    :param exp_message_id: List index of the expected message
    :type  exp_message_id: int
    """
    # Variable declaration
    # Folder to exclude from zip
    exclude_level: str = "00_femmt_simulation"
    # Create the instance
    test_dct: dct.DctMainCtl = dct.DctMainCtl()

    # Constant values of FlowControl
    test_parameter_2: tc.FlowControl = copy.deepcopy(test_FlowControl_base)

    # Prepare the setup
    # Create path
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create base path
        base_path = os.path.join(tmpdir, "base_path")
        os.makedirs(base_path, exist_ok=True)
        # Create filenames
        study_asc_name = "study_data.text"
        study_bin_name = "study_data.sqlite3"
        # Create the pathname
        valid_asc_source = os.path.join(base_path, study_asc_name)
        valid_bin_source = os.path.join(base_path, study_bin_name)
        # Create 2 files
        # text file
        with open(valid_asc_source, "w") as f:
            f.write(test_ascii_file_data)
        # Binary file (1 kilo byte random data)
        with open(valid_bin_source, "wb") as f:
            f.write(os.urandom(1024))
        # Create sub folders (3 levels) and store files within each level
        actual_dir = base_path
        for level in ["level1", "level2", exclude_level]:
            actual_dir = os.path.join(actual_dir, level)
            os.makedirs(actual_dir, exist_ok=True)
            # Copy test file
            target_file = os.path.join(actual_dir, study_asc_name)
            shutil.copy(valid_asc_source, target_file)
            # Copy binary file
            target_file = os.path.join(actual_dir, study_bin_name)
            shutil.copy(valid_bin_source, target_file)
            os.makedirs(actual_dir+"a", exist_ok=True)
            # Copy test file
            target_file = os.path.join(actual_dir+"a", study_asc_name)
            shutil.copy(valid_asc_source, target_file)
            # Copy binary file
            target_file = os.path.join(actual_dir+"a", study_bin_name)
            shutil.copy(valid_bin_source, target_file)

        # Assign pathname
        if is_path_existing:
            path_name = base_path
        else:
            path_name = os.path.join(base_path, "not_existing_folder")

        # Overtake constant values of FlowControl and update project directory
        test_flow_control: tc.FlowControl = copy.deepcopy(test_FlowControl_base)
        test_flow_control.general.project_directory = path_name

        # Perform the test
        with caplog.at_level(logging.INFO):
            test_dct.generate_zip_archive(test_flow_control)
            # Expected messages
            expected_message = ["",
                                f"Path {path_name} does not exists!",
                                f"Zip archive created:{path_name}_archived_"]

            # Check, if the folders are deleted, if the file exists
            if is_path_existing:
                # Debug
                mylist = os.listdir(tmpdir)
                is_file_found: bool = False
                # Open the first zip-file (only on exists)
                for zip_file_path in os.listdir(tmpdir):
                    # Check name
                    if zip_file_path.endswith('.zip'):
                        zip_file_path = os.path.join(tmpdir, zip_file_path)
                        is_file_found = True
                        break

                # Check if file is found
                assert is_file_found

                # Read the content of the zip-archive
                with zipfile.ZipFile(zip_file_path, 'r') as zf:
                    zip_contents = zf.namelist()

                # Restore actual path to relative path
                actual_dir = ""

                # Check content on upper level
                assert study_asc_name in zip_contents
                assert study_bin_name in zip_contents
                # Check content (folder and files)
                for level in ["level1", "level2"]:
                    actual_dir = os.path.join(actual_dir, level)
                    # Check text file
                    target_file = os.path.join(actual_dir, study_asc_name)
                    assert target_file in zip_contents
                    # Check binary file
                    target_file = os.path.join(actual_dir, study_bin_name)
                    assert target_file in zip_contents
                    # Check text file in additional folder
                    target_file = os.path.join(actual_dir + "a", study_asc_name)
                    assert target_file in zip_contents
                    # Check binary file in additional folder
                    target_file = os.path.join(actual_dir + "a", study_bin_name)
                    assert target_file in zip_contents

                # Check if the excluded folder is really excluded
                actual_dir = os.path.join(actual_dir, exclude_level)
                target_file = os.path.join(actual_dir, study_asc_name)
                target_file_bin = os.path.join(actual_dir, study_bin_name)
                # Loop over content list
                for name in zip_contents:
                    # Check text file
                    assert target_file not in name
                    # Check binary file
                    assert target_file_bin not in name

                # Check if text file in additional folder still exists
                target_file = os.path.join(actual_dir + "a", study_asc_name)
                assert target_file in zip_contents
                # Check binary file in additional folder
                target_file = os.path.join(actual_dir + "a", study_bin_name)
                assert target_file in zip_contents

            if len(caplog.records) > 0:
                assert caplog.records[0].message.startswith(expected_message[exp_message_id])
            else:
                assert "" == expected_message[exp_message_id]

#########################################################################################################
# test of get_initialization_queue_data
#########################################################################################################
