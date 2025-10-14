"""
morphology_ops.py
-----------------
Performs morphological operations such as Erosion, Dilation, Opening, and Closing
for binary images in the Domain-Specific Image Analysis & Enhancement Tool
(Agriculture domain).

Functions:
    - apply_erosion(image, ksize, iterations)
    - apply_dilation(image, ksize, iterations)
    - apply_opening(image, ksize)
    - apply_closing(image, ksize)
"""

import cv2
import numpy as np
from utils.plot_utils import show_images_side_by_side


def apply_erosion(image, ksize=3, iterations=1):
    """
    Apply erosion to shrink bright regions (useful for removing small white noise).

    Args:
        image (numpy.ndarray): Input binary or grayscale image.
        ksize (int): Kernel size for structuring element.
        iterations (int): Number of times erosion is applied.

    Returns:
        eroded (numpy.ndarray): Eroded image.
    """
    kernel = np.ones((ksize, ksize), np.uint8)
    eroded = cv2.erode(image, kernel, iterations=iterations)
    return eroded


def apply_dilation(image, ksize=3, iterations=1):
    """
    Apply dilation to expand bright regions (fills small holes).

    Args:
        image (numpy.ndarray): Input binary or grayscale image.
        ksize (int): Kernel size for structuring element.
        iterations (int): Number of times dilation is applied.

    Returns:
        dilated (numpy.ndarray): Dilated image.
    """
    kernel = np.ones((ksize, ksize), np.uint8)
    dilated = cv2.dilate(image, kernel, iterations=iterations)
    return dilated


def apply_opening(image, ksize=3):
    """
    Apply morphological opening (erosion followed by dilation).
    Removes small white noise from binary images.

    Args:
        image (numpy.ndarray): Input binary image.
        ksize (int): Kernel size.

    Returns:
        opened (numpy.ndarray): Noise-free image.
    """
    kernel = np.ones((ksize, ksize), np.uint8)
    opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return opened


def apply_closing(image, ksize=3):
    """
    Apply morphological closing (dilation followed by erosion).
    Fills small dark holes in white objects.

    Args:
        image (numpy.ndarray): Input binary image.
        ksize (int): Kernel size.

    Returns:
        closed (numpy.ndarray): Filled image.
    """
    kernel = np.ones((ksize, ksize), np.uint8)
    closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return closed


if __name__ == "__main__":
    # Example testing (optional)
    from utils.io_utils import load_image
    import os

    test_path = "data/samples/sample_binary_leaf.png"  # Example test image

    if os.path.exists(test_path):
        img = load_image(test_path, color=False)  # Load in grayscale

        eroded = apply_erosion(img, ksize=5)
        dilated = apply_dilation(img, ksize=5)
        opened = apply_opening(img, ksize=5)
        closed = apply_closing(img, ksize=5)

        show_images_side_by_side(img, eroded, "Original", "Erosion")
        show_images_side_by_side(img, dilated, "Original", "Dilation")
        show_images_side_by_side(img, opened, "Original", "Opening")
        show_images_side_by_side(img, closed, "Original", "Closing")
    else:
        print("Sample binary image not found! Please add one in 'data/samples/'.")
