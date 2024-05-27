"""Module to initialize utility function for the dataset."""

# standard libraries
import logging
from pathlib import Path
from typing import Tuple

# third-parties library
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.utils import image_dataset_from_directory


def get_dataset(
    dataset_directory: Path,
    image_size: Tuple[int, int],
    batch_size: int,
    seed: int = 123,
    verbose: bool = True,
) -> tf.data.Dataset:
    """Initialize the dataset from the provided directory.

    Args:
        dataset_directory (Path): Dataset split.
        image_size (Tuple): Size of the image.
        batch_size (int): Batch size to be used.
        seed (int, optional): Seed for randomness. Defaults to 123.
        verbose (bool, optional): Whether to show additional information. Defaults to True.

    Returns:
        tf.data.Dataset: Initialized dataset.

    Raises:
        FileNotFoundError: if dataset directory do not exist.
    """
    if not dataset_directory.exists():
        raise FileNotFoundError(f"Provided directory do not exist, {dataset_directory}")

    if verbose:
        logging.info(f"Number of images: {len(list(dataset_directory.glob('*/*.jpg')))}")

    ds = image_dataset_from_directory(
        dataset_directory,
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
    )

    if verbose:
        class_names = ds.class_names
        logging.info(f"{len(class_names)} classes available: {class_names}") 

    return ds


def visualize_dataset(ds: tf.data.Dataset) -> None:
    """Visualize 9 samples from the dataset.

    Args:
        ds (tf.data.Dataset): TensorFlow dataset.
    """
    # Get classes
    class_names = ds.class_names

    plt.figure(figsize=(10, 10))
    for images, labels in ds.take(1):
        for idx in range(9):
            _ = plt.subplot(3, 3, idx + 1)
            plt.imshow(images[idx].numpy().astype("uint8"))
            plt.title(class_names[labels[idx]])
            plt.axis("off")
