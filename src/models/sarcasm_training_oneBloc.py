"""
## UNVEILING SARCASM THROUGH SPEECH: A CNN-BASED APPROACH TO VOCAL FEATURE ANALYSIS ##

This Python file is part of the project undertaken in the course "Deep Learning for Audio Data"
during the winter semester 2023/24 at the Technische Universität Berlin, within the curriculum
of Audio Communication and Technologies M.Sc.

Authors (Group 1):
- Pierre S.F. Kolingba-Froidevaux
- Blanca Sabater Vilchez
- Raphaël G. Gillioz
- Florian Morgner

Description:
This python script is responsible for the creationa and the training of the sarcasm detection model.
    The required parameters are to be specified in the provided yaml configuration file.

Direct usage:
python src/models/sarcasm_training_oneBloc.py -c path/to/config_file.yaml

Date: 29.03.2024

"""
##   Ps. extracted voice data is also refered as denoised data in our code!

from __future__ import print_function

import sys
sys.path.append('models/research/audioset/vggish')

import os
import json
import click
import logging
import yaml
from tqdm import tqdm
import numpy as np
import tensorflow.compat.v1 as tf
import librosa
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import vggish_input
import vggish_postprocess
import vggish_params
import vggish_slim

# import resampy  # pylint: disable=import-error
# import tf_slim as slim
# import vggish_params as params

# tf.disable_v2_behavior()


@click.command()
@click.option("--config_file", "-c", default="project_config.yaml", help="Path to the YAML configuration file")
def main(config_file="project_config.yaml"):
    """
    This function prepare and execute the training of the sarcasm model.
    The code is plit in the following main etapes:
        1) Loading parameters
        2) Feature extraction
        3) Rebalance extracted features
        4) Extract embeddings
        5) Split data
        6) Train network

    Args:
        config_file (str, optional): Path to the YAML configuration file. Defaults to "project_config.yaml".
    """

    ## 1) Loading parameters -------------------------------------------------------------------------

    # Check if a virtual environement is active (does not check if it is the right one!)
    if not "VIRTUAL_ENV" in os.environ:
        raise ValueError("Virtual environement not activated! Please activate the appropriate one.")

    # Get logger with appropriate name and level
    logger = logging.getLogger(__name__)
    if __name__ == "__main__":
        logging.basicConfig(level=logging.INFO)  # Set INFO level for direct execution

    logger.info(">>> Start training process...")

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
    
    # Get the parameters from the yaml configuration file
    config_train = config["MUStARD"]["training"]
    json_labels_path = os.path.join(config_train["json_dir"], config_train["json_name"])
    denoised_audio_path = os.path.join(config_train["denois_base_dir"], config_train["denois_data_dir"])
    augmented_audio_path = os.path.join(config_train["aug_base_dir"], config_train["aug_data_dir"])

    ## 2) Feature extraction -------------------------------------------------------------------------

    # Load the labels data from the json file
    logger.debug('> Load JSON...')
    try:
        with open(json_labels_path, 'r') as f:
            try:
                labels_data = json.load(f)
                # Proceed with further processing if JSON parsing is successful
                logger.debug("> JSON file loaded successfully!")
            except json.JSONDecodeError as e:
                # Handle JSON parsing errors
                logger.error("> Error parsing JSON:", e)
                exit(1)
    except FileNotFoundError:
        # Handle file not found errors
        logger.error("> File not found:", json_labels_path)
        exit(1)

    # Extract features and labels
    features, labels = get_all_examples(denoised_audio_path, augmented_audio_path, labels_data)

    ## 3) Rebalance extracted features ---------------------------------------------------------------

    # Print data distribution
    unique, counts = np.unique(labels, return_counts=True)
    logger.info('> Data distribution (0: non-sarcastic, 1: sarcastic):')
    logger.info(dict(zip(unique, counts)))

    # As the length of the audio is not equal for sarcastic and non-sarcastic
    #       we need to take into account this imbalanced distribution of examples
    # - RandomOverSampler initialiation
    ros = RandomOverSampler(random_state=config_train["overSample_rdm_state"])
    # - Flatten features in case of 3D features
    features_flattened = features.reshape(features.shape[0], -1)
    # - Perform oversampling on the labels to match size
    features_oversampled, labels_oversampled = ros.fit_resample(features_flattened, labels)
    # - Reformat feature to be compatible with VGGish
    features_oversampled = features_oversampled.reshape(-1, vggish_params.NUM_FRAMES, vggish_params.NUM_BANDS)
    # - Test new distribution
    unique, counts = np.unique(labels_oversampled, return_counts=True)
    logger.info('> Rebalanced data distribution (0: non-sarcastic, 1: sarcastic):')
    logger.info(dict(zip(unique, counts)))

    ## 4) Extract embeddings -------------------------------------------------------------------------

    # Print the number of available GPUs
    logger.info(f"> Num GPUs Available: {len(tf.config.experimental.list_physical_devices('GPU'))}")

    # Set tensorflow eager mode for immediate evaluation of operations
    tf.config.run_functions_eagerly(config_train["eager_mode"])

    # Set flag parameters for the embeddings extraction
    if not hasattr(tf.flags, 'DEFINE_string'):
        flags = tf.app.flags # Older flag module version
    else:
        flags = tf.flags # Newer flag module version

    if 'num_batches' not in flags.FLAGS:
        flags.DEFINE_integer('num_batches', config_train["nb_batches"], 'Number of batches of examples to feed into the model.')

    if 'train_vggish' not in flags.FLAGS:
        flags.DEFINE_boolean('train_vggish', config_train["train_vggish"], 'If True, allow VGGish parameters to change during training.')

    if 'checkpoint' not in flags.FLAGS:
        flags.DEFINE_string('checkpoint', config_train["checkpoint"], 'Path to the VGGish checkpoint file.')

    if 'pca_params' not in flags.FLAGS:
        flags.DEFINE_string('pca_params', config_train["pca_params"], 'Path to the VGGish pca params file.')
    FLAGS = flags.FLAGS

    # Get the embeddings from VGGisch with post-processing
    embedding_batch_oversampled = extract_embeddings(features_oversampled, FLAGS)

    ## 5) Split data (train and test data) -----------------------------------------------------------

    # Splitting the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(embedding_batch_oversampled, labels_oversampled, test_size=config_train["train_test_size"], stratify=labels_oversampled)

    # Print class distribution
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    logger.info("> Distribution of the classes in the training set:", dict(zip(unique_train, counts_train)))
    logger.info("> Distribution of the classes in the test set:", dict(zip(unique_test, counts_test)))

    ## 6) Train network ------------------------------------------------------------------------------

    # Create and train the model
    logger.info(f"> Batch size: {config_train['batch_size']}")
    train_and_evaluate_model(X_train, y_train, X_test, y_test, config_train["epochs"], config_train["batch_size"], 1, config_train["learning_rate"])

    logger.info(">>> Training process done!")


def get_all_examples(denoised_audio_path, augmented_audio_path, labels_data):
    """
    This function extracts features and their corresponding labels from audio files.

    Args:
        - audio_folder_path (str): The path to the directory containing original audio files.
        - augmented_folder_path (str): The path to the directory containing augmented audio files.
        - labels_data (dict): A dictionary containing labels for the audio files.

    Returns:
        tuple: A tuple containing two NumPy arrays:
            - The first array contains the extracted features.
            - The second array contains the corresponding labels.
    """

    # Get logger with appropriate name and level
    logger = logging.getLogger(__name__)
    logger.info('>> Feature extraction...')

    all_features = []
    all_labels = []

    # Labels and features extraction for denoised/extracted-voice audio
    logger.info('>> Extracting features from denoised audio...')
    for file_name in tqdm(os.listdir(denoised_audio_path)):
        if file_name.endswith('.wav'):
            # Load denoised/extracted-voice audio
            audio_path = os.path.join(denoised_audio_path, file_name)
            audio, sr = librosa.load(audio_path, sr=None)  # Utiliser sr=None pour conserver le taux d'échantillonnage original
            examples = vggish_input.waveform_to_examples(audio, sr)

            # Get the corresponding labels from the json data
            original_name = os.path.splitext(file_name)[0]  # Extraire le nom du fichier d'origine sans extension
            if original_name in labels_data:
                is_sarcastic = labels_data[original_name]["sarcasm"]
                label = 1 if is_sarcastic else 0  # 1 for sarcasme, 0 for non-sarcasme
            else:
                logger.error(f"> Denoised data: the original key {original_name} does not exists int the provided JSON file!")
                continue

            for example in examples:
                all_features.append(example)
                all_labels.append(label)

    # Labels and features extraction for augmented audio
    logger.info('>> Extracting features from augmented audio...')
    for file_name in tqdm(os.listdir(augmented_audio_path)):
        if file_name.endswith('.wav'):
            # Load augmented audio
            audio_path = os.path.join(augmented_audio_path, file_name)
            audio, sr = librosa.load(audio_path, sr=None)
            examples = vggish_input.waveform_to_examples(audio, sr)

            # Get the corresponding labels from the json data
            original_name = "_".join(file_name.split('_')[:-1])  # Extraire le nom du fichier d'origine
            if original_name in labels_data:
                is_sarcastic = labels_data[original_name]["sarcasm"]
                label = 1 if is_sarcastic else 0  # 1 for sarcasme, 0 for non-sarcasme
            else:
                logger.error(f"> Augmented data: the original key {original_name} does not exists int the provided JSON file!")
                continue

            for example in examples:
                all_features.append(example)
                all_labels.append(label)

    logger.info('>> Feature extraction done!')
    return np.array(all_features), np.array(all_labels)


def extract_embeddings(features, FLAGS):
    """
    This function extracts the embeddings of the provided features from VGGish and apply a PCA on them.

    Args:
        - features (NumPy array): previously extracted features
        - FLAGS (tf flags): flag parameters


    Returns:
        embedings (NumPy array): embeddings out of VGGish
        
    """

    # Get logger with appropriate name and level
    logger = logging.getLogger(__name__)
    logger.info('>> Embeddings extraction...')

    pproc = vggish_postprocess.Postprocessor(FLAGS.pca_params)
    with tf.compat.v1.Graph().as_default(), tf.compat.v1.Session() as sess:
        # Initialize VGGish
        vggish_slim.define_vggish_slim(training=False)
        
        # Define the Saver
        saver = tf.compat.v1.train.Saver()
        
        # Restore the checkpoint
        saver.restore(sess, FLAGS.checkpoint)

        # Locate input and output tensors
        features_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
        embedding_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)

        features_reshaped = features.reshape((-1, vggish_params.NUM_FRAMES, vggish_params.NUM_BANDS))
        
        # Run the model to obtain embeddings
        embedding_batch = sess.run(embedding_tensor, feed_dict={features_tensor: features_reshaped})

        # Apply post-processing - PCA (whitens the data) - Loss function way higher with it though
        postprocessed_batch = pproc.postprocess(embedding_batch)

        logger.info('>> Embeddings extraction done!')
        return postprocessed_batch
    

def train_and_evaluate_model(X_train, y_train, X_test, y_test, epochs, batch_size, _NUM_CLASSES, learning_Rate):
    """
    This function generated and trains the two extra layers added to VGGisch with the obtained embeddings.

    Args:
        - X_train, y_train (NumPy arrays): X and Y train sets
        - X_test, y_test (NumPy arrays): X and Y test sets
        - epochs: number of epochs
        - batch_size: yep! You gessed it!
        - _NUM_CLASSES: number of outputed classes
        
    """

    # Get logger with appropriate name and level
    logger = logging.getLogger(__name__)
    logger.info('>> Training...')

    # Generate the model
    model = Sequential([
        Dense(4096, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(_NUM_CLASSES, activation='sigmoid')  # Use of a 'sigmoid' for the binary classification
    ])

    # Compilation of the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_Rate)

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',  # 'binary_crossentropy' for the binary classification
                  metrics=['accuracy'])

    # Callbacks configurations
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
    model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, mode='min', verbose=1)
#    model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1)

    # Training of the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[early_stopping, model_checkpoint])

    # Get the best saved model
    model.load_weights('best_model.keras')

    # Model evaluation
    predictions = model.predict(X_test)
    predictions = np.round(predictions)
    
    logger.info('>> Training done!')
    logger.info(classification_report(y_test, predictions))


if __name__ == "__main__":
    main()