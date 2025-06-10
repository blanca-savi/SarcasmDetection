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
This python script is responsible for the extraction of the audio from the original videos of the database.
    The extraction parameters are to be specified in the provided yaml configuration file.

Direct usage:
python src/data/extract_audio_from_video.py -c path/to/config_file.yaml

Date: 29.03.2024

"""

import subprocess
import os
from datetime import datetime
import yaml
import logging
import argparse
from tqdm import tqdm

def extract_audio(config_file="project_config.yaml"):
    """
    Extracts and prefilters audio from raw dataset videos according to the provided configuration file.

    Args:
        config_file (str, optional): Path to the YAML configuration file. Defaults to "project_config.yaml".
    """

    # Get logger with appropriate name and level
    logger = logging.getLogger(__name__)
    if __name__ == "__main__":
        logging.basicConfig(level=logging.DEBUG)  # Set INFO level for direct execution

    logger.info(">> Extract audio from videos...")

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

    # Check ffmpeg is installed
    try:
        subprocess.run(['ffmpeg', '-version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        logger.error("> ffmpeg is not installed. Please install it to proceed.")
        exit(1)

    # Extract relevant values from config
    config_extr = config["MUStARD"]["audio_extract"]
    input_dir = os.path.join(config_extr["in_base_dir"], config_extr["in_data_dir"])
    output_base_dir = config_extr["out_base_dir"]
    audio_suffix = config_extr["out_data_dir"]
    output_dir = os.path.join(output_base_dir, audio_suffix)
    output_log = os.path.join(output_base_dir, f"{audio_suffix}.log")
    config_ffmpeg = config_extr["ffmpeg"]

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Reset log file
    open(output_log, 'w').close()

    logger.info(f"> Processing videos in '{input_dir}' to '{output_dir}'...")

    # List audio files
    video_files = os.listdir(input_dir)

    # Initialize tqdm progress bar
    progress_bar = tqdm(video_files, desc="Audio data extraction", unit="file")

    for video_file in video_files:
        if video_file.endswith(config_extr["in_data_type"]):
            # Extract file name and set output path
            video_path = os.path.join(input_dir, video_file)
            file_name_no_ext = os.path.splitext(video_file)[0]
            output = os.path.join(output_dir, f"{file_name_no_ext}{config_extr['out_data_type']}")

            # Timestamp log file and write processing start message
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            with open(output_log, 'a') as log_file:
                log_file.write(f"[{timestamp}] Processing video: {file_name_no_ext}\n--------------------\n")

            # Build and execute ffmpeg command
            ffmpeg_command = [
                'ffmpeg', '-i', video_path, '-vn', '-ac', config_ffmpeg["ac"], '-ar', config_ffmpeg["ar"],
                '-acodec', config_ffmpeg["codec"], '-af', config_ffmpeg["af"],
                output, '-y'
            ]

            # Log command and capture output
            with open(output_log, 'a') as log_file:
                log_file.write(f"[{timestamp}] ffmpeg command: {' '.join(ffmpeg_command)}\n")

                try:
                    subprocess.run(ffmpeg_command, check=True, stdout=log_file, stderr=subprocess.STDOUT)
                    logging.debug(f"> Audio extracted from {video_file} successfully.")
                    log_file.write(f"[{timestamp}] Audio extracted successfully!\n--------------------\n")
                except subprocess.CalledProcessError as e:
                    logging.error(f"> Error extracting the audio from {video_file}.")
                    log_file.write(f"[{timestamp}] Error: {e}\n--------------------\n")

        # Progress bar update
        progress_bar.update(1)

    log_file.close()
    logging.info(f">> Audio extraction process completed! (See {output_log} file for more details)")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Extract audio from video")
    parser.add_argument("--config_file", "-c", default="project_config.yaml", help="Path to the YAML configuration file")
    args = parser.parse_args()
    
    extract_audio(config_file=args.config_file)  # Use default config file
