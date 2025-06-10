Unveiling Sarcasm Through Speech: A CNN-Based Approach to Vocal Feature Analysis
==============================

This project aims to detect sarcasm using Convolutional Neural Networks (CNN), focusing on vocal features without relying on text or context. We leverage the [MUStARD dataset](https://github.com/soujanyaporia/MUStARD), perform [deepfilter](https://github.com/Rikorose/DeepFilterNet) denoising on audio data, and utilize a modified [VGGish model](https://github.com/tensorflow/models/tree/master/research/audioset/vggish) for feature extraction. 

The [MUStARD dataset](https://github.com/soujanyaporia/MUStARD) dataset includes sarcastic and non-sarcastic audiovisual excerpts from various TV shows. Audio data is standardized, denoised, balanced and augmented to enhance model training.

We used [VGGish](https://github.com/tensorflow/models/tree/master/research/audioset/vggish) in two key ways: as a feature extractor to transform audio inputs into compact 128-D embeddings for simpler downstream classification, and as a foundational layer in larger models for complex audio data processing, allowing for fine-tuning and improved adaptability. This enabled efficient audio classification and model customization for specific analysis tasks.

## Paper
The accompanying research [paper](docs/SarcasmDetection_Paper.pdf) documenting this project can be found in the docs folder.

## Requirements
Prior to project execution, please ensure you have installed the following software:
- [FFmpeg](https://ffmpeg.org)
- [Make](https://medium.com/@samsorrahman/how-to-run-a-makefile-in-windows-b4d115d7c516)(for Windows users only)

**Note:** The project has been successfully tested on a Mac with an M1 chip and Python versions 3.9 and 3.11.

## Quick start
This project offers a streamlined workflow that automates environment setup, dataset download, data preprocessing, model training, and evaluation. To execute all these steps in a single command from the project's root directory, use:
```
make train
```

Project Organization Structure
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md
    ├── data
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- Final, canonical data sets for modeling.
    │   └── raw            <- Original, immutable data dump.
    │
    ├── docs               <- Paper and documentation
    │
    ├── models             <- Trained and serialized models (VGGish models)
    │
    ├── notebooks          <- Notebooks used for development (Informational only)
    │
    ├── requirements.txt   <- File specifying dependencies for environment recreation
    │
    ├── setup_environment.py        <- Script for installing and checking the virtual environment
    │
    ├── project_config.yaml         <- File containing project parameters
    │
    ├── setup.py
    │
    └── src
        ├── __init__.py
        │
        ├── data
        │   └── make_dataset.py                 <- Main script for downloading and preprocessing the dataset
        │   └── create_data_directory.py        <- Script that creates the folder structure to house the data
        │   └── download_dataset.py             <- Script that downloads and unzips the dataset
        │   └── extract_audio_from_video.py     <- Script that extracts audio files from dataset videos
        │   └── extract_voice.py                <- Voice extraction/denoising script (deepfilter)
        │   └── aug_denoised.py                 <- Script for data augmentation
        │
        │   └── data_sarcasm_check.py           <- Script checking dataset distribution
        │
        ├── features
        │   └── vggish_model-ckpt               <- VGGish model checkpoint
        │   └── vggish_pca-params.npz           <- VGGish embedding PCA parameters
        │
        └── models
            │
            └── sarcasm_training_oneBloc.py     <- Script for processing and training 



--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


## Advanced Usage

For more granular control, you can execute the installation, preprocessing, and training stages independently. This can be achieved through two methods:

1. **Using Makefile commands:**

    List available Makefile commands using:

    ```
    Make help
    ```
2. **Calling individual scripts:**

    Individual scripts reside in the project's root directory. Refer to the script headers for specific usage instructions.

## Parameter Configuration
The project's preprocessing and training parameters are configurable through the [project_config.yaml](project_config.yaml) file. Scripts automatically retrieve these parameters during execution. However, exercise caution when modifying these parameters, as some may impact script execution.

## Other considerations
The model design and default dataset size ensure that the project can run on moderate computing resources within a reasonable timeframe. However, extracting the embedding can be very resource-intensive, and this stage can cause the programme to crash. It is therefore advisable to carry out the training with all other applications on the computer switched off.
While the current implementation combines processing and training into a single script, future improvements will involve separating these steps into distinct scripts for enhanced clarity.

The two VGGish pretained data files provided ([VGGish model checkpoint](https://storage.googleapis.com/audioset/vggish_model.ckpt), in TensorFlow checkpoint format and [Embedding PCA parameters](https://storage.googleapis.com/audioset/vggish_pca_params.npz), in NumPy compressed archive format) where downloaded from the net.
