"""
geometric_ops.py
----------------
Performs geometric transformations (rotation, scaling, translation, cropping)
for the Domain-Specific Image Analysis & Enhancement Tool (Agriculture domain).

Functions:
    - rotate_image(image, angle): Rotate image by a given angle.
    - scale_image(image, scale_x, scale_y): Scale image by given factors.
    - translate_image(image, tx, ty): Translate image by given offsets.
    - crop_image(image, x, y, width, height): Crop a region from the image.
"""

import cv2
import numpy as np
from utils.plot_utils import show_images_side_by_side


def rotate_image(image, angle):
    """
    Rotate an image around its center by a specified angle.

    Args:
        image (numpy.ndarray): Input image.
        angle (float): Rotation angle in degrees.

    Returns:
        rotated (numpy.ndarray): Rotated image.
    """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Get rotation matrix
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h))

    return rotated


def scale_image(image, scale_x=1.0, scale_y=1.0):
    """
    Scale (resize) an image.

    Args:
        image (numpy.ndarray): Input image.
        scale_x (float): Horizontal scaling factor.
        scale_y (float): Vertical scaling factor.

    Returns:
        scaled (numpy.ndarray): Scaled image.
    """
    scaled = cv2.resize(image, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
    return scaled


def translate_image(image, tx, ty):
    """
    Translate (shift) an image along the x and y axes.

    Args:
        image (numpy.ndarray): Input image.
        tx (int): Shift along x-axis (positive = right).
        ty (int): Shift along y-axis (positive = down).

    Returns:
        translated (numpy.ndarray): Translated image.
    """
    rows, cols = image.shape[:2]
    matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    translated = cv2.warpAffine(image, matrix, (cols, rows))
    return translated


def crop_image(image, x, y, width, height):
    """
    Crop a rectangular region from the image.

    Args:
        image (numpy.ndarray): Input image.
        x (int): Top-left x-coordinate.
        y (int): Top-left y-coordinate.
        width (int): Width of the crop region.
        height (int): Height of the crop region.

    Returns:
        cropped (numpy.ndarray): Cropped region.
    """
    cropped = image[y:y+height, x:x+width]
    return cropped


if __name__ == "__main__":
    # Example testing (optional)
    from utils.io_utils import load_image
    import os

    test_path = "data/input/sample_leaf.jpg"  # Change to your sample image
    if os.path.exists(test_path):
        img = load_image(test_path)

        rotated = rotate_image(img, 45)
        scaled = scale_image(img, 0.5, 0.5)
        translated = translate_image(img, 50, 30)
        cropped = crop_image(img, 100, 100, 200, 200)

        show_images_side_by_side(img, rotated, "Original", "Rotated (45°)")
        show_images_side_by_side(img, scaled, "Original", "Scaled (0.5x)")
        show_images_side_by_side(img, translated, "Original", "Translated")
        show_images_side_by_side(img, cropped, "Original", "Cropped")
    else:
        print("Test image not found! Please add one in 'data/input/'.")
