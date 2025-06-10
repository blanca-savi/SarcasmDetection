"""
## UNVEILING SARCASM THROUGH SPEECH: A CNN-BASED APPROACH TO VOCAL FEATURE ANALYSIS ##

This Python file is part of the project undertaken in the course "Deep Learning for Audio Data"
during the winter semester 2023/24 at the Technische Universität Berlin, within the curriculum
of Audio Communication and Technologies M.Sc.

Authors (Group 1):
- Blanca Sabater Vilchez
- Raphaël G. Gillioz
- Pierre S.F. Kolingba-Froidevaux
- Florian Morgner

Description:
This Python script is responsible for setting up the virtual environment
    and installing its dependencies for the project.

Direct usage:
python setup_environment.py --venv-path ../custom_venv

Date: 29.03.2024

"""

import subprocess
import sys
import os
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Setup environement')

# Define required python major required
REQUIRED_PYTHON = "python3"

def main(venv_path):
    """
    Main function to check Python version, create virtual environment, and install dependencies.

    Args:
        venv_path (str): The path to create the virtual environment.
    """

    logger.info(">>> Checking and installing environment and dependencies. ------")
    try:
        check_python_version()
        create_virtual_environment(venv_path)
        install_dependencies(venv_path)
    except Exception as e:
        print("An error occurred:", e)
        return  # Stop execution if an error occurs
    
    logger.info(">>> Development environment ready! ------")


def check_python_version():
    """
    Check if the Python version meets the required version.

    Raises:
        TypeError: If the Python version does not match the required version.
    """

    logger.info(">> Checking python installation...")
    system_major = sys.version_info.major
    if REQUIRED_PYTHON == "python":
        required_major = 2
    elif REQUIRED_PYTHON == "python3":
        required_major = 3
    else:
        raise ValueError("Unrecognized Python interpreter: {}".format(
            REQUIRED_PYTHON))

    if system_major != required_major:
        raise TypeError(
            "This project requires Python {}. Found: Python {}".format(
                required_major, sys.version))
    else:
        logger.info(">> Installed Python passes version test!")


def create_virtual_environment(venv_path):
    """
    Create a virtual environment using venv.

    Args:
        venv_path (str): The path to create the virtual environment.

    Raises:
        FileExistsError: If the virtual environment already exists.
        subprocess.CalledProcessError: If an error occurs while creating the virtual environment.
    """
    logger.info(">> Checking/creating virtual environment...")
    try:
        subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", venv_path], check=True)
        logger.info(">> Virtual environment created successfully!")
    except subprocess.CalledProcessError as e:
        if "FileExistsError" in str(e):
            # Ignore FileExistsError (virtual environment already exists)
            logger.info(">> Virtual environment already exists. Good to go!")
        else:
            # Catch and print other errors
            logger.error("Error creating virtual environment:", e)
            raise


def install_dependencies(venv_path):
    """
    Install dependencies using pip.

    Args:
        venv_path (str): The path to the virtual environment.

    Raises:
        subprocess.CalledProcessError: If an error occurs while installing dependencies.
    """

    logger.info(">> Checking and installing dependencies...")

    # Upgrade pip
    logger.info("> Updating pip...")
    try:
        # Redirect stdout to a log file
        with open("pip_install.log", "w") as log_file:
            subprocess.run([os.path.join(venv_path, "bin", "python"), "-m", "pip", "install", "--upgrade", "pip"], check=True, stdout=log_file, stderr=subprocess.PIPE)
        logger.info("> pip upgraded successfully!")
    except subprocess.CalledProcessError as e:
        logger.error("Error upgrading pip:", e)
        raise e
    
    # Install dependencies
    logger.info("> Installing dependencies...please wait...")
    pip_command = [os.path.join(venv_path, "bin", "pip"), "install", "-r", "requirements.txt"]
    try:
        # Redirect stdout to a log file
        with open("pip_install.log", "a") as log_file:
            subprocess.run(pip_command, check=True, stdout=log_file, stderr=subprocess.PIPE)
        logger.info("> Dependencies installed successfully! (Intsallation details in pip_install.log file)")
    except subprocess.CalledProcessError as e:
        logger.error("Error installing dependencies: %s", e)
        raise e
    
    log_file.close()
    logger.info(">> Environment ready!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--venv_path', type=str, default="../.venvs/depenv", help='Path to the virtual environment')
    args = parser.parse_args()

    main(args.venv_path)