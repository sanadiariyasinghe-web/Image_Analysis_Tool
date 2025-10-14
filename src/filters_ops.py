"""
filters_ops.py
---------------
Performs image smoothing, noise reduction, and sharpening operations
for the Domain-Specific Image Analysis & Enhancement Tool (Agriculture domain).

Functions:
    - apply_average_filter(image, ksize):     Apply Averaging filter.
    - apply_gaussian_filter(image, ksize, sigma): Apply Gaussian Blur.
    - apply_median_filter(image, ksize):      Apply Median filter.
    - apply_sharpening(image):                Apply image sharpening filter.
"""

import cv2
import numpy as np
from utils.plot_utils import show_images_side_by_side


def apply_average_filter(image, ksize=5):
    """
    Apply an averaging (mean) filter to smooth the image.

    Args:
        image (numpy.ndarray): Input image.
        ksize (int): Kernel size (must be odd).

    Returns:
        avg_filtered (numpy.ndarray): Smoothed image.
    """
    avg_filtered = cv2.blur(image, (ksize, ksize))
    return avg_filtered


def apply_gaussian_filter(image, ksize=5, sigma=0):
    """
    Apply Gaussian blur for noise reduction.

    Args:
        image (numpy.ndarray): Input image.
        ksize (int): Kernel size (odd number preferred).
        sigma (float): Standard deviation for Gaussian kernel.
                       0 means auto-computed by OpenCV.

    Returns:
        gauss_filtered (numpy.ndarray): Blurred image.
    """
    gauss_filtered = cv2.GaussianBlur(image, (ksize, ksize), sigma)
    return gauss_filtered


def apply_median_filter(image, ksize=5):
    """
    Apply Median filtering to remove salt-and-pepper noise.

    Args:
        image (numpy.ndarray): Input image.
        ksize (int): Kernel size (odd number).

    Returns:
        median_filtered (numpy.ndarray): Denoised image.
    """
    median_filtered = cv2.medianBlur(image, ksize)
    return median_filtered


def apply_sharpening(image):
    """
    Apply sharpening to enhance edges and fine details.

    Args:
        image (numpy.ndarray): Input image.

    Returns:
        sharpened (numpy.ndarray): Sharpened image.
    """
    # Standard sharpening kernel
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened


# -----------------------------
# Example testing
# -----------------------------
if __name__ == "__main__":
    import os
    from utils.io_utils import load_image

    test_path = "../data/input/sample_leaf.jpg"  # Update relative to src folder
    if os.path.exists(test_path):
        img = load_image(test_path)

        avg = apply_average_filter(img, 5)
        gauss = apply_gaussian_filter(img, 5)
        median = apply_median_filter(img, 5)
        sharp = apply_sharpening(img)

        show_images_side_by_side(img, avg, "Original", "Averaging Filter")
        show_images_side_by_side(img, gauss, "Original", "Gaussian Blur")
        show_images_side_by_side(img, median, "Original", "Median Filter")
        show_images_side_by_side(img, sharp, "Original", "Sharpened")
    else:
        print("Test image not found! Please add one in '../data/input/'.")
