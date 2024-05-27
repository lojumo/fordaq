"""Module that define the network."""

# Import standard library
from typing import Any, Dict, List, Optional, Tuple

# Import third-parties library
import tensorflow as tf


def get_augmentations() -> tf.keras.layers.Layer:
    """Get augmentations used.

    Returns:
        List[tf.keras.layers.Layer]: List of augmentations.
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomContrast(0.1),
        tf.keras.layers.RandomTranslation(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1)),
    ])


def build_convolution_block(
    n_filters: int,
    kernel_size: int,
    padding: str = "same",
    activation: Optional[str] = "relu",
    use_bn: bool = True,
    pool_size: Optional[Tuple[int, int]] = None,
    dropout_rate: float = 0.25,
    name: str = "convblock",
) -> List[tf.keras.layers.Layer]:
    """Create a simple convolution block

    Args:
        n_filters (int): Number of filters.
        kernel_size (int): Kernel size for Conv2D.
        padding (str, optional): Padding mode. Defaults to "same".
        activation (str, optional): Activation layer. Defaults to "relu".
        use_bn (bool, optional): Whether to use Batch Normalization or not. Defaults to True.
        pool_size (Tuple[int, int], optional): Pool size. Defaults to None.
        dropout_rate (float, optional): Dropout rate. Defaults to 0.25.
        name (str, optional): Base name of the layer. Defaults to `convblock`.

    Returns:
        tf.keras.layers.Layer: Initialized Convolution block.
    """
    # Convolution
    conv_layer = tf.keras.layers.Conv2D(
        filters=n_filters,
        kernel_size=kernel_size,
        padding=padding,
        name=f"{name}_conv",
    )

    layers: List[tf.keras.layers.Layer] = [conv_layer]

    if use_bn:
        layers.append(tf.keras.layers.BatchNormalization(name=f"{name}_batch_norm"))

    if activation is not None:
        layers.append(tf.keras.layers.Activation(activation, name=f"{name}_act"))

    if pool_size is not None:
        layers.append(tf.keras.layers.MaxPooling2D(pool_size, name=f"{name}_maxpool"))

    if dropout_rate > 0.:
        layers.append(tf.keras.layers.Dropout(rate=dropout_rate, name=f"{name}_dropout"))

    return layers


def build_classifier(
    nb_classes: int,
    img_size: Tuple[int, int],
    conv_block_params: Dict[Any, Any],
    use_aug: bool = True,
) -> tf.keras.Model:
    """Build a classifier model.

    Args:
        nb_classes (int): Number of output classes.
        img_size (Tuple[int, int]): Size of the image.
        conv_block_params (Dict[Any, Any]): Convolution block parameters.
        use_aug (bool, optional): Whether to use Augmentations or not.

    Returns:
        tf.keras.Model: Model used to classify the type of woods
    """
    img_shape = img_size + (3,)

    # Create backbone
    preprocess_input = tf.keras.applications.vgg16.preprocess_input

    backbone = tf.keras.applications.VGG16(
        include_top=False,     # Only feature extraction
        weights="imagenet",    # Use pre-trained weights obtained with the ImageNet dataset
        input_shape=img_shape,
    )

    # Freeze backbone as we just want the features without re-training the network
    backbone.trainable = False

    # Build classifier on top
    layers: List[tf.keras.layers.Layer] = []
    for layer_name, kwargs in conv_block_params.items():
        layers += build_convolution_block(name=layer_name, **kwargs)

    layers.append(tf.keras.layers.Flatten(name="flatten"))
    layers.append(tf.keras.layers.Dropout(rate=0.5, name="flatten_dropout"))
    layers.append(tf.keras.layers.Dense(nb_classes, name="targets"))

    classifier = tf.keras.Sequential(layers)

    # Build model
    inputs = tf.keras.layers.Input(shape=img_shape)
    if use_aug:
        data_augmentation = get_augmentations()
        inputs = data_augmentation(inputs)
    inputs_preprocessed = preprocess_input(inputs)
    features = backbone(inputs_preprocessed, training=False)
    outputs = classifier(features)

    return tf.keras.Model(inputs, outputs)
