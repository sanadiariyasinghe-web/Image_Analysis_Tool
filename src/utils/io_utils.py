"""
io_utils.py
------------
Utility functions for handling image input/output operations and metadata extraction
for the Domain-Specific Image Analysis & Enhancement Tool (Agriculture domain).

Functions:
    - load_image(path): Load an image from the given path.
    - save_image(path, image): Save an image to the specified path.
    - get_image_metadata(image, path=None): Extract metadata such as dimensions, channels, etc.
"""

import cv2
import os

def load_image(path):
    """
    Load an image from the given file path.

    Args:
        path (str): Path to the image file.

    Returns:
        image (numpy.ndarray): Loaded image.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found at: {path}")
    
    image = cv2.imread(path)
    if image is None:
        raise ValueError("Failed to load image. Check file format or path.")
    
    return image


def save_image(path, image):
    """
    Save an image to a given file path.

    Args:
        path (str): Output file path (e.g., 'data/output/result.jpg').
        image (numpy.ndarray): Image to save.
    """
    # Create output directory if it doesn’t exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    success = cv2.imwrite(path, image)
    if not success:
        raise IOError(f"Could not save image to: {path}")


def get_image_metadata(image, path=None):
    """
    Extract metadata from an image.

    Args:
        image (numpy.ndarray): Image array.
        path (str, optional): Image file path (to get file size if provided).

    Returns:
        dict: Metadata containing width, height, channels, and file size (if available).
    """
    height, width = image.shape[:2]
    channels = 1 if len(image.shape) == 2 else image.shape[2]
    
    metadata = {
        "Width": width,
        "Height": height,
        "Channels": channels
    }
    
    if path and os.path.exists(path):
        metadata["File Size (KB)"] = round(os.path.getsize(path) / 1024, 2)
    
    return metadata
