"""Constant path and file name storage."""

# python libraries
import os

# 3rd party libraries
# own libraries
import dct

DCT_ROOT = os.path.dirname(os.path.realpath(dct.__file__))
GECKO_PATH = os.path.join(DCT_ROOT, "topology", "dab", "GeckoCIRCUITS")
GECKO_COMPONENT_MODELS_DIRECTORY = "gecko_component_models"
CIRCUIT_CAPACITOR_LOSS_FOLDER = "01_capacitor_results"
CIRCUIT_INDUCTOR_RELUCTANCE_LOSSES_FOLDER = "08_reluctance_inductor_results"
CIRCUIT_INDUCTOR_FEM_LOSSES_FOLDER = "09_fem_inductor_results"
CIRCUIT_TRANSFORMER_RELUCTANCE_LOSSES_FOLDER = "08_reluctance_transformer_results"
CIRCUIT_TRANSFORMER_FEM_LOSSES_FOLDER = "09_fem_transformer_results"
SUMMARY_COMBINATION_FOLDER = "combination_results"
SUMMARY_COMBINATION_PlOTS_FOLDER = "combination_plots"
FILTERED_RESULTS_PATH = "filtered_results"
RELUCTANCE_COMPLETE_FILE = "reluctance_processing_complete.json"
SIMULATION_COMPLETE_FILE = "simulation_processing_complete.json"
PROCESSING_COMPLETE_FILE = "processing_complete.json"

# result data frames filenames
DF_SUMMARY_WITHOUT_HEAT_SINK_WITHOUT_OFFSET = "df_wo_hs_wo_offset.csv"
DF_SUMMARY_WITH_HEAT_SINK_WITHOUT_OFFSET = "df_w_hs_wo_offset.csv"
DF_SUMMARY_FINAL = "df_final.csv"
DF_SUMMARY_FINAL_FILTERED = "df_final_filtered.csv"
