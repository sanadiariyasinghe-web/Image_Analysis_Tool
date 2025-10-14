"""
acquisition_display.py
----------------------
Handles image acquisition (loading) and display operations for the
Domain-Specific Image Analysis & Enhancement Tool (Agriculture domain).

Functions:
    - load_and_show_image(path): Load, display, and return an image.
    - display_metadata(image, path): Print metadata details of the image.
"""

import cv2
from utils.io_utils import load_image, get_image_metadata
from utils.plot_utils import show_image


def load_and_show_image(path):
    """
    Load an image from the given path and display it.

    Args:
        path (str): File path of the image to be loaded.

    Returns:
        image (numpy.ndarray): The loaded image.
    """
    image = load_image(path)
    show_image("Original Image", image)
    return image


def display_metadata(image, path=None):
    """
    Display image metadata such as dimensions, channels, and file size.

    Args:
        image (numpy.ndarray): The loaded image.
        path (str, optional): Image file path to include file size in metadata.
    """
    metadata = get_image_metadata(image, path)
    print("\n🧾 Image Metadata:")
    print("-" * 30)
    for key, value in metadata.items():
        print(f"{key}: {value}")
    print("-" * 30)


# -----------------------------
# Example usage for testing
# -----------------------------
if __name__ == "__main__":
    test_path = "../data/input/sample_leaf.jpg"  # Update relative to src folder
    img = load_and_show_image(test_path)
    display_metadata(img, test_path)
