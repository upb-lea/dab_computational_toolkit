"""DTOs for general datasets."""

# python libraries
import dataclasses
import os
import logging

# 3rd party libraries

# Initialize logger
logger = logging.getLogger(__name__)

class StudyData:
    """Data class containing all general information to perform a study."""

    def __init__(self, study_name: str, optimization_directory: str):
        # Initialize the member variables
        self.study_name = study_name
        self.optimization_directory = optimization_directory

    @staticmethod
    def check_study_data(study_path: str, study_name: str) -> bool:
        """
        Verify if the study path and sqlite3-database file exists.

        Works for all types of studies (circuit, inductor, transformer, heat sink).
        :param study_path: drive location path to the study
        :type  study_path: str
        :param study_name: Name of the study
        :type  study_name: str
        :return: True, if the optimization could be performed successful
        :rtype: bool
        """
        # Variable definition
        # return value initialization to false
        is_study_existing = False

        # check path
        if os.path.exists(study_path) or study_path == "":
            # Assemble file name
            study_name = study_name + ".sqlite3"
            target_file = os.path.join(study_path, study_name)
            # check filename
            if os.path.isfile(target_file):
                is_study_existing = True
            else:
                logger.info(f"File {target_file} does not exists!")
        else:
            logger.info(f"Path {study_path} does not exists!")

        # True = study exists
        return is_study_existing

@dataclasses.dataclass
class FilterData:
    """Information about the filtered circuit designs."""

    filtered_list_files: list[str]
    filtered_list_pathname: str
    circuit_study_name: str
