"""
## UNVEILING SARCASM THROUGH SPEECH: A CNN-BASED APPROACH TO VOCAL FEATURE ANALYSIS ##

This Python file is part of the project undertaken in the course "Deep Learning for Audio Data"
during the winter semester 2023/24 at the Technische Universität Berlin, within the curriculum
of Audio Communication and Technologies M.Sc.

Authors (Group 1):
- Florian Morgner
- Pierre S.F. Kolingba-Froidevaux
- Raphaël G. Gillioz
- Blanca Sabater Vilchez

Description:
This python script is responsible for the data augmentation of the project.

Direct usage:
python src/data/aug_denoised.py -c path/to/config_file.yaml

Date: 29.03.2024

"""
import os
import librosa
import numpy as np
import soundfile as sf
import yaml
import logging
import argparse
from tqdm import tqdm

def augment_data(config_file="project_config.yaml"):
    """
    Data augmentation for the training

    Args:
        config_file (str, optional): Path to the YAML configuration file. Defaults to "project_config.yaml".
    """

    # Get logger with appropriate name and level
    logger = logging.getLogger(__name__)
    if __name__ == "__main__":
        logging.basicConfig(level=logging.DEBUG)  # Set INFO level for direct execution

    logger.info('>> Data augmentation...')

    # Load config file
    try:
        with open(config_file, "r") as stream:
            config = yaml.safe_load(stream)
    except FileNotFoundError:
        logger.error("Error: Config file not found!")
        exit(1)  # Exit with error code 1
    except yaml.YAMLError as exc:
        logger.error(f"Error parsing YAML: {exc}")
        exit(1)

    # Assign config variables
    config_aug = config["MUStARD"]["data_augmentation"]
    wav_dir = os.path.join(config_aug["in_base_dir"], config_aug["in_data_dir"])
    augmented_dir = os.path.join(config_aug["out_base_dir"], config_aug["out_data_dir"])

    # Create output directory
    os.makedirs(augmented_dir, exist_ok=True)

    # List audio files
    wav_files = os.listdir(wav_dir)

    # Initialize tqdm progress bar
    progress_bar = tqdm(wav_files, desc="Data augmentation", unit="file")

    # Augment data
    for wav_file in wav_files:
        original_wav, sr = librosa.load(os.path.join(wav_dir, wav_file), sr=None, mono=True)

        # Shifting the soundwave
        wav_shift = np.roll(original_wav, int(sr/10))
        filename = os.path.join(augmented_dir, wav_file.replace(".wav", "")+"_shift.wav")
        sf.write(filename, wav_shift, sr)

        # Time-stretching the wave
        wav_time = librosa.effects.time_stretch(original_wav, rate=0.8)
        filename2 = os.path.join(augmented_dir, wav_file.replace(".wav", "")+"_stretch.wav")
        sf.write(filename2, wav_time, sr)

        # Pitch-shifting
        n_steps = np.random.randint(-3, 4, size=1)[0]  # Extraction du nombre réel de la liste
        wav_pitch = librosa.effects.pitch_shift(original_wav, n_steps=n_steps, sr=sr)
        filename3 = os.path.join(augmented_dir, wav_file.replace(".wav", "")+"_pitch.wav")
        sf.write(filename3, wav_pitch, sr)

        # Progress bar update
        progress_bar.update(1)

    logger.info('>> Data augmentation done!')


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Data augmentation")
    parser.add_argument("--config_file", "-c", default="project_config.yaml", help="Path to the YAML configuration file")
    args = parser.parse_args()

    augment_data(config_file=args.config_file)  # Use default config file