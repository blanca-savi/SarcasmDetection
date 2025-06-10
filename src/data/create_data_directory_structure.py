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
This python script is responsible for the creation of the directory structure for the data.
    The data diretory structure to be build is to be specified in the provided yaml configuration file.

Direct usage:
python src/data/create_data_directory_structure.py -c path/to/config_file.yaml

Date: 29.03.2024

"""

import os
import yaml
import logging
import argparse


def create_structure(config_file="project_config.yaml"):
    """
    Creates the data directory structure specified in the provided config file.

    Args:
        config_file (str, optional): Path to the YAML configuration file. Defaults to "project_config.yaml".
    """

     # Get logger with appropriate name and level
    logger = logging.getLogger(__name__)
    if __name__ == "__main__":
        logging.basicConfig(level=logging.DEBUG)  # Set INFO level for direct execution

    logger.info(">> Creating data directory structure...")

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

    # Create directories and .gitkeep files
    for folder in config["data_structure"]:
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
            logger.info(f"> Created directory: {folder}")
        else:
            logger.debug(f"> Directory already exists: {folder}")

        gitkeep_path = os.path.join(folder, ".gitkeep")
        if not os.path.exists(gitkeep_path):
            with open(gitkeep_path, "w"):
                pass
            logger.debug(f"> Created .gitkeep file in {folder}")

    logger.info(">> The data directory structure is ready!")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Create data directory structure")
    parser.add_argument("--config_file", "-c", default="project_config.yaml", help="Path to the YAML configuration file")
    args = parser.parse_args()
    
    create_structure(config_file=args.config_file)  # Use default config file
