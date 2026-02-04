"""Visualize the simulation results."""

# python libraries
import os
import sys

# own libraries
from dct.topology.dab import DabCircuitOptimization
from dct.topology.sbc import SbcCircuitOptimization
from dct.circuit_enums import TopologyEnum
import dct.toml_checker as tc
from dct.dctmainctl import DctMainCtl
from dct.constant_path import CIRCUIT_WAVEFORMS_FOLDER
from dct.topology.circuit_optimization_base import CircuitOptimizationBase

def visualize_waveform_verification(working_directory: str) -> None:
    """
    Verify the simulated vs. calculated waveform by visual plots.

    :param working_directory: working directory
    :type working_directory: str
    """
    # Check if workspace path is not provided by argument
    if working_directory == "":
        # Find process workspace
        working_directory = os.path.dirname(os.path.abspath(__file__))
        # Join parent folder of workspace_path and workspace path to absolute path name
        working_directory = os.path.join(os.path.dirname(working_directory), "workspace")

    # Set directory to workspace path
    try:
        # Change to workspace
        os.chdir(working_directory)
    except FileNotFoundError as exc:
        raise ValueError("Error: Workspace folder does not exists!") from exc
    except PermissionError as exc:
        raise ValueError("Error: No permission to change the folder!") from exc

    # Create absolute path
    working_directory = os.path.abspath(working_directory)

    file_path = os.path.join(working_directory, "progFlow.toml")
    flow_control_loaded, dict_prog_flow = DctMainCtl.load_toml_file(file_path)

    if not flow_control_loaded:
        raise ValueError("No data available")

    # Verify toml data and transfer to class
    toml_prog_flow = tc.FlowControl(**dict_prog_flow)

    # Allocate and initialize circuit configuration
    if toml_prog_flow.general.topology == TopologyEnum.dab.value:
        circuit_optimization: CircuitOptimizationBase = DabCircuitOptimization()
    elif toml_prog_flow.general.topology == TopologyEnum.sbc.value:
        circuit_optimization = SbcCircuitOptimization()
    else:
        raise ValueError("Serious programming error in topology selection. Please write an issue!")

    project_directory = toml_prog_flow.general.project_directory
    circuit_configuration_file = toml_prog_flow.configuration_data_files.topology_files[1].replace(".toml", "")

    dto_directory = os.path.join(
        working_directory,
        project_directory,
        toml_prog_flow.circuit.subdirectory,
        circuit_configuration_file,
        CIRCUIT_WAVEFORMS_FOLDER)

    summary_pre_processing_path = os.path.join(project_directory, toml_prog_flow.pre_summary.subdirectory, circuit_configuration_file)

    # circuit_optimization.plot_compare_waveforms(dto_directory)
    # circuit_optimization.visualize_single_lab_data(summary_pre_processing_path, "c961_i38_t1071_cap1_B32714P8105+000_cap2_B32724A9805K000")
    circuit_optimization.visualize_all_lab_data(summary_pre_processing_path)


if __name__ == "__main__":
    # Variable declaration
    arg1 = ""

    # Read the command line
    arguments = sys.argv

    # Check on argument, which corresponds to the workspace file location
    if len(arguments) > 1:
        arg1 = arguments[1]
        # Check if this corresponds to the workspace path
        arg1 = os.path.join(arg1, "workspace")
        print(f"file path={arg1}")
        # Check if the path not exist (absolute or relative path)
        if not os.path.exists(arg1):
            # Consider it as relative path and create the absolute path
            arg1 = os.path.abspath(arg1)
            print(f"new file path={arg1}")
            # Check if the path does not exist
            if not os.path.exists(arg1):
                print(f"Provides argument {arguments[1]} does not corresponds to the path to subfolder 'workspace'.\n")
                print("This is neither the absolute nor the relative path. Program will use the default path!")
                # Reset path variable
                arg1 = ""

        # Convert it to the absolute path
        arg1 = os.path.abspath(arg1)
    # Execute program
    visualize_waveform_verification(arg1)
