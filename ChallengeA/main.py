"""Main file that runs the stitching algorithm."""

# standard library imports
import logging

from argparse import ArgumentParser, Namespace
import os

import cv2

# project imports
from stitching import detect_images_masks, stitch_images


def parse_arguments() -> Namespace:
    """Parse arguments for running the program.

    Returns:
        Namespace: The parsed arguments.
    """
    parser = ArgumentParser(description="Script which stiches together several images into one.")
    parser.add_argument(
        "--input", "-i", type=str, required=True,
        help="Folder containing the images to be stitched.",
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True,
        help="Folder where the output is saved.", default=".",
    )
    parser.add_argument(
        "--cupping", "-c", action="store_true",
        help="Flag indicating whether the stitched image has a 'cup shape'.",
    )

    return parser.parse_args()


def main() -> None:
    """Run program."""

    # Parse arguments
    args: Namespace = parse_arguments()

    # Get all images & masks
    images, masks = detect_images_masks(args.input)
    logging.info(f"Detected {len(images)} images and {len(masks)} masks.")

    image_stitched = stitch_images(args.input, images, masks)
    logging.info("Stitching done!")

    cv2.imshow('Result', image_stitched)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    success = cv2.imwrite(os.path.join(args.output, "stitched_image.jpg"), image_stitched)
    logging.info("Success" if success else "Fail")


if __name__ == "__main__":
    # Set-up logger
    logging.basicConfig(level=logging.INFO)

    main()
