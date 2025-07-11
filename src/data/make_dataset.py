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
This Python script calls all scripts needed to download and preprocess the data for training.

Direct usage:
python src/data/make_dataset.py

Date: 29.03.2024

"""

import click
import logging
from pathlib import Path

# Import data preprocessing scripts
import create_data_directory_structure
import download_dataset
import extract_audio_from_video
import extract_voice
import aug_denoised

@click.command()
def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """

    logger = logging.getLogger('Dataset Preparation')
    logging.basicConfig(level=logging.INFO) 
    logger.info('>>> Getting data and preprocess them for training... ------')

    # 1. Create data folder structure
    create_data_directory_structure.create_structure()

    # 2. Download dataset
    download_dataset.download()

    # 3. Extract audio from video
    extract_audio_from_video.extract_audio()

    # 4. Extract voice from audio
    extract_voice.extract_voice()

    # 5. Augment data artificially
    aug_denoised.augment_data()

    logger.info(">>> All data processing steps completed successfully! ------")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    #load_dotenv(find_dotenv())

    main()

