"""Module which defines the stitching functions."""

# standard library imports
import os

from typing import List, Tuple

# third-parties library imports
import logging
import cv2
import numpy as np


def get_index(filename: str, is_mask: bool = False) -> int:
    """Get index of the image or mask filename.

    Image format: `xx.jpg`.
    Mask format: `mask_xx.png`.

    Args:
        filename (str): Name of the file.
        is_mask (bool, optional): Whether the file is a mask. Defaults to False.

    Returns:
        int: Index of the image/mask.
    """
    return int(filename[:-4].split("_")[-1]) if is_mask else int(filename[:-4])


def detect_images_masks(folder: str) -> Tuple[List[str], List[str]]:
    """Detect images & masks from the folder provided.

    Images are in format `x.jpg` and the corresponding masks are `mask_x.png`.

    Args:
        folder (str): Input folder.

    Returns:
        Tuple[List[str], List[str]]: Images and corresponding masks in order.

    Raises:
        FileNotFoundError: Error when the folder does not exist.
    """
    # Sanity check
    if not os.path.isdir(folder):
        raise FileNotFoundError("Input folder does not exists.")

    # Detect all files
    filenames: List[str] = os.listdir(folder)
    image_filenames: List[str] = [
        filename for filename in filenames if not filename.startswith("mask_")
    ]
    mask_filenames: List[str] = [
        filename for filename in filenames if filename.startswith("mask_")
    ]

    image_filenames = sorted(image_filenames, key=get_index)
    mask_filenames = sorted(mask_filenames, key=lambda x: get_index(x, True))

    # Assess same length
    assert len(image_filenames) == len(mask_filenames), "Not all images have a corresponding mask or vice-versa."

    return image_filenames, mask_filenames


def stitch_images(folder: str, image_filenames: List[str], mask_filenames: List[str]) -> np.ndarray:
    """Stitches images together using the masks.

    It involves involves several steps:
        1. Feature detection: Identifying and extracting unique features (e.g., corners, edges) from each input image.
        2. Feature matching: Finding correspondences between features in the overlapping regions of the input images.
        3. Homography estimation: Estimating the transformation (e.g., rotation, scaling, translation) that aligns the input images.
        4. Warping: Applying the estimated transformation to the input images.
        5. Blending: Combining the warped images into a single seamless output image.
    
    Args:
        folder (str): Folder containing the images/masks.
        image_filenames (List[str]): List of images to stitch.
        mask_filenames (List[str]): List of masks to extract the interesting zone.

    Returns:
        np.ndarray: The stitched image.
    """
    images_cropped: List = []

    # Obtain cropped image based on mask
    logging.info("Loading images & masks...")
    for image_filename, mask_filename in zip(image_filenames, mask_filenames):              
        # Check image has corresponding mask
        if get_index(image_filename) == get_index(mask_filename, True):
            # Read image
            image = cv2.imread(os.path.join(folder, image_filename), cv2.IMREAD_COLOR)
            
            # Read mask
            mask = cv2.imread(os.path.join(folder, mask_filename), cv2.IMREAD_UNCHANGED)
            rect = cv2.boundingRect(mask)
            
            # Create cropped image based on mask
            image_cropped = image[rect[1]:(rect[1]+rect[3]), rect[0]:(rect[0]+rect[2])]

            # If ROI region is empty, ignore image
            if image_cropped.shape[1] > 0:
                # Rotate image
                image_rotated = cv2.rotate(image_cropped, cv2.ROTATE_90_CLOCKWISE)
                images_cropped.append(image_rotated)
            else:
                # Cropped image is empty
                logging.warn(f"Image do not contain any ROI, see '{image_filename}' and '{mask_filename}'.")

    # Start OpenCV Stitcher class
    logging.info("Start stitching images...")
    stitcher = cv2.Stitcher.create()
    (status, stitched_image) = stitcher.stitch(images_cropped)

    if status == cv2.STITCHER_OK:
        return stitched_image
    else:
        raise ValueError(f"Stitching failed: {status}")
        