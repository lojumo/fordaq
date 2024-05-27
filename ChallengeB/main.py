"""Module to train the neural network."""

# standard library
import logging

from argparse import ArgumentParser
from pathlib import Path
from typing import List, Tuple

# third-parties library
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix

# project imports
from dataset import get_dataset
from network import build_classifier
from utils import plot_history


def training(
    image_size: Tuple[int, int], epochs: int, learning_rate: float, batch_size: int, training_data: Path, validation_data: Path,
) -> Tuple[tf.keras.Model, List[str]]:
    """Training model

    Args:
        image_size (Tuple[int, int]): Image size used to initialize input.
        epochs (int): Number of epochs.
        learning_rate (float): Learning rate to be used.
        batch_size (int): Batch size.
        training_data (Path): Path to the training data.
        validation_data (Path): Path to the validation data.

    Returns:
        Tuple[tf.keras.Model, List[str]]: Trained network and classes.
    """
    # Get datasets
    AUTOTUNE = tf.data.AUTOTUNE

    training_ds, _ = get_dataset(training_data, image_size, batch_size)
    validation_ds, _ = get_dataset(validation_data, image_size, batch_size)

    classes = training_ds.class_names

    training_ds = training_ds.prefetch(buffer_size=AUTOTUNE)
    validation_ds = validation_ds.prefetch(buffer_size=AUTOTUNE)

    # Build model
    conv_block_params = {
        "convblock_0": dict(
            n_filters=32,
            kernel_size=3,
            padding="same",
            activation="relu",
            use_bn=True,
            dropout_rate=0.5,
        ),
        "convblock_1": dict(
            n_filters=16,
            kernel_size=3,
            padding="same",
            activation="relu",
            use_bn=True,
            dropout_rate=0.5,
        ),
    }

    model = build_classifier(nb_classes=len(classes), img_size=image_size, conv_block_params=conv_block_params)
    model.summary()

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    # Train network
    history = model.fit(training_ds, epochs=epochs, validation_data=validation_ds)
    plot_history(history)

    return model, classes


def evaluate(model: tf.keras.Model, classes: List[str], test_folder: Path, image_size: Tuple[int, int]) -> None:
    """Evaluate trained network

    Args:
        model (tf.keras.Model): Trained network.
        classes (List[str]): List of classes.
        test_folder (str): Location of test images.
        image_size (Tuple[int, int]): Image size.
    """
    # Read images
    if not test_folder.exists():
        raise FileNotFoundError(f"Provided directory do not exist, {test_folder}")

    predicted_labels: List[int] = []
    labels: List[int] = []

    test_ds, nb_images = get_dataset(test_folder, image_size, 1)
    ds = iter(test_ds)

    for _ in range(nb_images):
        image, label = next(ds)
        raw_predictions = model.predict(image)
        scores = tf.nn.softmax(raw_predictions[0])
        predictions = np.argmax(scores)

        labels.append(label)
        predicted_labels.append(predictions)

    matrix_values = confusion_matrix(y_true=labels, y_pred=predicted_labels)

    # Plotting confusion matrix
    matrix = pd.DataFrame(matrix_values, classes, classes)
    print(matrix)

    # Visualising confusion matrix
    plt.figure(figsize = (16,14),facecolor='white')
    heatmap = sns.heatmap(matrix, annot = True, annot_kws = {'size': 20}, fmt = 'd', cmap = 'YlGnBu')
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation = 0, ha = 'right', fontsize = 18, weight='bold')
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation = 0, ha = 'right', fontsize = 18, weight='bold')

    plt.title('Confusion Matrix\n', fontsize = 18, color = 'darkblue')
    plt.ylabel('True label', fontsize = 14)
    plt.xlabel('Predicted label', fontsize = 14)
    # plt.show()
    plt.savefig("./confusion_matrix.jpg")


def main():
    """Main function."""
    # Define configuration
    parser = ArgumentParser(description="Run training and evaluation.")
    parser.add_argument("--train_data", type=Path, help="Path to the training data")
    parser.add_argument("--val_data", type=Path, help="Path to the validation data")
    parser.add_argument("--test_data", type=Path, help="Path to the test data", default=Path(""))
    parser.add_argument("--epochs", type=int, help="Number of epochs", default=10)
    parser.add_argument("--lr", type=float, help="Learning rate to be used", default=0.0001)
    parser.add_argument("--batch_size", type=int, help="Batch size to be used", default=10)
    parser.add_argument("--image_size", type=str, help="Image size (heightxwidth) to be trained on.")

    arguments = parser.parse_args()

    height_width = arguments.image_size.split("x")
    image_size = (int(height_width[0]), int(height_width[1]))

    model, classes = training(image_size, arguments.epochs, arguments.lr, arguments.batch_size, arguments.train_data, arguments.val_data)

    if len(str(arguments.test_data)) > 0:
        evaluate(model, classes, arguments.test_data, image_size)


if __name__ == "__main__":
    # Set-up logger
    logging.basicConfig(level=logging.INFO)

    main()