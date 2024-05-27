"""Module that get all image sizes."""

# standard imports
from pathlib import Path
from tqdm import tqdm

# third-parties library
import cv2
import numpy as np


if __name__ == '__main__':
    """Utility script which gets image shapes."""

    input_sizes = {}
    dataset_path = Path("species_ds")

    for image_path in tqdm(dataset_path.glob("**/*.jpg")):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        image_shape = image.shape[:2]

        if image_shape in input_sizes:
            input_sizes[image_shape] += 1
        else:
            input_sizes[image_shape] = 1

    input_heights = [input_shape[0] for input_shape in input_sizes]
    input_widths = [input_shape[1] for input_shape in input_sizes]

    max_height = np.max(input_heights)
    min_height = np.min(input_heights)

    max_width = np.max(input_widths)
    min_width = np.min(input_widths)

    print(f"Maximum size: {max_height}x{max_width}")
    if (max_height, max_width) in input_sizes:
        print(f"{input_sizes[(max_height, max_width)]} image(s) found.")

    print(f"Minimum size: {min_height}x{min_width}")
    if (min_height, min_width) in input_sizes:
        print(f"{input_sizes[(min_height, min_width)]} image(s) found.")
