"""
## UNVEILING SARCASM THROUGH SPEECH: A CNN-BASED APPROACH TO VOCAL FEATURE ANALYSIS ##

This Python file is part of the project undertaken in the course "Deep Learning for Audio Data"
during the winter semester 2023/24 at the Technische Universität Berlin, within the curriculum
of Audio Communication and Technologies M.Sc.

Authors (Group 1):
- Raphaël G. Gillioz
- Blanca Sabater Vilchez
- Pierre S.F. Kolingba-Froidevaux
- Florian Morgner

Description:
This python script is responsible for the download and unzipping of the original database.
    The download parameters are to be specified in the provided yaml configuration file.

Direct usage:
python src/data/download_dataset.py -c path/to/config_file.yaml

Date: 29.03.2024

"""

import os
import gdown
import yaml
import zipfile
import logging
import argparse
from tqdm import tqdm

def download(config_file="project_config.yaml"):
    """
    Downloads and unzips the dataset specified in the provided config file.

    Args:
        config_file (str, optional): Path to the YAML configuration file. Defaults to "project_config.yaml".
    """

    # Get logger with appropriate name and level
    logger = logging.getLogger(__name__)
    if __name__ == "__main__":
        logging.basicConfig(level=logging.DEBUG)  # Set INFO level for direct execution

    logger.info(">> Download the database...")

    # Load config file
    try:
        with open(config_file, "r") as stream:
            config = yaml.safe_load(stream)
    except FileNotFoundError:
        logger.error("> Error: Config file not found!")
        exit(1)  # Exit with error code 1
    except yaml.YAMLError as exc:
        logger.error(f"> Error parsing YAML: {exc}")
        exit(1)

    config_raw = config["MUStARD"]["raw"]
    working_dir = config_raw["base_dir"]

    # Create the working directory if needed
    os.makedirs(working_dir, exist_ok=True)

    # Change working directory (temporarily)
    cwd = os.getcwd()
    os.chdir(working_dir)

    try:
        # Download and unzip dataset of videos (if not already present)
        if not os.path.exists(config_raw["data_dir"]):
            output = "MUStARD_dataset.zip"
            gdown.download(config_raw["data_url"], output, quiet=False)  # Download the zip file
            unzip_specific_folder(output, config_raw["data_dir"])
            os.remove(output)  # Remove the zip file
            logger.info("> Dataset downloaded and unzipped successfully!")
        else:
            logger.warning("> Warning: Dataset seems to be already downloaded. If not, delete the utterances_final folder manually.")

        # Download data descriptions (if not already present)
        if not os.path.exists(config_raw["json_name"]):
            gdown.download(config_raw["json_url"], quiet=False)  # Download the JSON file
            logger.info("> Data description file sarcasm_data.json downloaded successfully!")
        else:
            logger.debug("> Data description file sarcasm_data.json is already present.")

    finally:
        os.chdir(cwd)  # Restore working directory

    logger.info(">> Database downloaded and unzipped!")

# Unzipping function (remains the same)
def unzip_specific_folder(zip_file, target_string):

    # Get logger with appropriate name and level
    logger = logging.getLogger(__name__)
    if __name__ == "__main__":
        logging.basicConfig(level=logging.DEBUG)  # Set INFO level for direct execution

    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        # Get a list of all items (files and directories) in the zip file
        zip_contents = zip_ref.namelist()
        
        # Find the closest folder name containing the target string
        closest_folder = None
        closest_distance = float('inf')
        for item in zip_contents:
            if item.endswith('/') and target_string in item:
                distance = item.find(target_string)
                if distance < closest_distance:
                    closest_folder = item
                    closest_distance = distance
        
        if closest_folder:
            # Extract only the contents of the closest/target folder
            for item in zip_contents:
                if item.startswith(closest_folder) and item != closest_folder:
                    # Extract the item into the target directory
                    zip_ref.extract(item)
            logger.debug(f"> Folder '{closest_folder}' contents extracted to '{target_string}' folder!")
        else:
            logging.error(f"> No folder containing '{target_string}' found in the zip file!")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Download dataset")
    parser.add_argument("--config_file", "-c", default="project_config.yaml", help="Path to the YAML configuration file")
    args = parser.parse_args()

    download(config_file=args.config_file)
