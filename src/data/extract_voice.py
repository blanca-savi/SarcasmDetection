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
This python script is responsible for the voice extraction.
    The extraction parameters are to be specified in the provided yaml configuration file.

Direct usage:
python src/data/extract_voice.py -c path/to/config_file.yaml

Date: 29.03.2024

"""
##   Ps. extracted voice data is also refered as denoised data in our code!

import os
import yaml
import logging
import argparse
from tqdm import tqdm
from df.enhance import enhance, init_df, load_audio, save_audio

def extract_voice(config_file="project_config.yaml"):
    """
    Extracts voice from audio files using the deepfilter package.

    Args:
        config_file (str, optional): Path to the YAML configuration file. Defaults to "project_config.yaml".
    """
    
    # Get loggers for the terminal with appropriate name and level
    terminal_logger = logging.getLogger(__name__)
    if __name__ == "__main__":
        terminal_logger.setLevel(logging.DEBUG)  # Set level for direct execution

    terminal_logger.info(">> Extracting voices...")

    # Load config file
    try:
        with open(config_file, "r") as stream:
            config = yaml.safe_load(stream)
    except FileNotFoundError:
        terminal_logger.error("> Error: Config file not found!")
        exit(1)  # Exit with error code 1
    except yaml.YAMLError as exc:
        terminal_logger.error(f"> Error parsing YAML: {exc}")
        exit(1)

    # Extract relevant values from config
    config_vExtr = config["MUStARD"]["extract_voice"]
    input_dir = os.path.join(config_vExtr["in_base_dir"], config_vExtr["in_data_dir"])
    output_dir = os.path.join(config_vExtr["out_base_dir"], config_vExtr["out_data_dir"])
    logging_dir = os.path.join(config_vExtr["out_base_dir"], "deepFilter.log")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Clear existing log file if it exists
    if os.path.exists(logging_dir):
        open(logging_dir, 'w').close()

    # Get loggers for the log file output with appropriate name and level
    file_logger = logging.getLogger('extract_voice')
    file_logger.setLevel(logging.DEBUG)  # Set the desired level
    file_handler = logging.FileHandler(logging_dir, mode='w')  # Handler for file output
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # Explicitly remove any existing handlers (optional)
    for handler in file_logger.handlers[:]:
        file_logger.removeHandler(handler)

    # Add the file handler to the logger
    file_logger.addHandler(file_handler)

    # List audio files
    audio_files = os.listdir(input_dir)

    # Initialize tqdm progress bar
    progress_bar = tqdm(audio_files, desc="Voice extraction", unit="file")

    # Load default model
    model, df_state, _ = init_df()

    # Loop through files in the input directory
    for audio_file in audio_files:
        if audio_file.endswith('.wav'):
            try:
                # Get full input path
                input_file = os.path.join(input_dir, audio_file)

                # Log input file details
                file_logger.info("> Processing input file: %s", input_file)

                # Load model and denoise audio
                audio, _ = load_audio(input_file, sr = df_state.sr())
                enhanced = enhance(model, df_state, audio)

                # Save enhanced audio to output directory with same audio_file
                output_file = os.path.join(output_dir, audio_file)
                save_audio(output_file, enhanced, df_state.sr())  # Replace with correct function call

                file_logger.info("> Enhanced audio saved to: %s", output_file)

            except Exception as e:
                file_logger.error("> Error processing %s: %s", input_file, str(e))
                terminal_logger.error("> Error processing %s: %s", input_file, str(e))

        # Progress bar update
        progress_bar.update(1)

    terminal_logger.info(f">> Voices extracted! (See {logging_dir} file for more details)")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Extract voice")
    parser.add_argument("--config_file", "-c", default="project_config.yaml", help="Path to the YAML configuration file")
    args = parser.parse_args()

    extract_voice(config_file=args.config_file)  # Use default config file

