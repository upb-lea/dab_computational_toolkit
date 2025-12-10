"""Constant path storage."""

# python libraries
import os

# 3rd party libraries
# own libraries
import dct

DCT_ROOT = os.path.dirname(os.path.realpath(dct.__file__))
GECKO_PATH = os.path.join(DCT_ROOT, "topology", "dab", "GeckoCIRCUITS")
GECKO_COMPONENT_MODELS_DIRECTORY = "gecko_component_models"
