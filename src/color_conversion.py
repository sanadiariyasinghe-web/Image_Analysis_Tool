"""
color_conversion.py
-------------------
Performs color space conversions for the Domain-Specific Image Analysis
& Enhancement Tool (Agriculture domain).

Functions:
    - bgr_to_grayscale(image): Convert BGR image to Grayscale.
    - bgr_to_hsv(image): Convert BGR image to HSV.
    - hsv_to_bgr(image): Convert HSV image back to BGR.
    - bgr_to_binary(image, threshold): Convert BGR/Grayscale image to Binary.
"""

import cv2
from utils.plot_utils import show_images_side_by_side


def bgr_to_grayscale(image):
    """
    Convert a BGR image to Grayscale.

    Args:
        image (numpy.ndarray): Input BGR image.

    Returns:
        gray (numpy.ndarray): Grayscale image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray


def bgr_to_hsv(image):
    """
    Convert a BGR image to HSV color space.

    Args:
        image (numpy.ndarray): Input BGR image.

    Returns:
        hsv (numpy.ndarray): HSV image.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv


def hsv_to_bgr(image):
    """
    Convert an HSV image back to BGR color space.

    Args:
        image (numpy.ndarray): Input HSV image.

    Returns:
        bgr (numpy.ndarray): Converted BGR image.
    """
    bgr = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    return bgr


def bgr_to_binary(image, threshold=127):
    """
    Convert a BGR or Grayscale image to Binary (black & white).

    Args:
        image (numpy.ndarray): Input BGR or Grayscale image.
        threshold (int): Threshold value (0–255).

    Returns:
        binary (numpy.ndarray): Binary image (0 or 255 values).
    """
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray = bgr_to_grayscale(image)
    else:
        gray = image

    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return binary


if __name__ == "__main__":
    # Example testing (optional)
    from utils.io_utils import load_image
    import os

    test_path = "data/input/sample_leaf.jpg"  # Change to your image
    if os.path.exists(test_path):
        img = load_image(test_path)

        gray = bgr_to_grayscale(img)
        hsv = bgr_to_hsv(img)
        binary = bgr_to_binary(img)

        show_images_side_by_side(img, gray, "Original", "Grayscale")
        show_images_side_by_side(img, hsv, "Original", "HSV")
        show_images_side_by_side(gray, binary, "Grayscale", "Binary")
    else:
        print("Test image not found! Please add a sample image in 'data/input/'.")
