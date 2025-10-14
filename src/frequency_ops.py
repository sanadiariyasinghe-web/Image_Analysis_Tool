"""
frequency_ops.py
----------------
Performs frequency domain filtering using Fourier Transform for the
Domain-Specific Image Analysis & Enhancement Tool (Agriculture domain).

Functions:
    - ideal_low_pass(image, cutoff)
    - ideal_high_pass(image, cutoff)
    - gaussian_low_pass(image, sigma)
    - gaussian_high_pass(image, sigma)
"""

import cv2
import numpy as np
from utils.plot_utils import show_images_side_by_side


def fft_image(image):
    """
    Compute the Fourier Transform of the image and shift the zero frequency to center.

    Args:
        image (numpy.ndarray): Grayscale input image.

    Returns:
        fshift (numpy.ndarray): Shifted Fourier Transform of image.
    """
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    return fshift


def ifft_image(fshift):
    """
    Compute the inverse Fourier Transform to get back the spatial domain image.

    Args:
        fshift (numpy.ndarray): Shifted Fourier Transform of image.

    Returns:
        img_back (numpy.ndarray): Filtered image in spatial domain.
    """
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return np.uint8(np.clip(img_back, 0, 255))


def ideal_low_pass(image, cutoff=30):
    """
    Apply Ideal Low-Pass Filter.

    Args:
        image (numpy.ndarray): Grayscale input image.
        cutoff (int): Radius of the low-pass filter.

    Returns:
        filtered_img (numpy.ndarray): Low-pass filtered image.
    """
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    fshift = fft_image(image)

    # Create mask
    mask = np.zeros((rows, cols), np.uint8)
    cv2.circle(mask, (ccol, crow), cutoff, 1, -1)

    # Apply mask
    fshift_filtered = fshift * mask
    return ifft_image(fshift_filtered)


def ideal_high_pass(image, cutoff=30):
    """
    Apply Ideal High-Pass Filter.

    Args:
        image (numpy.ndarray): Grayscale input image.
        cutoff (int): Radius of the high-pass filter.

    Returns:
        filtered_img (numpy.ndarray): High-pass filtered image.
    """
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    fshift = fft_image(image)

    # Create mask
    mask = np.ones((rows, cols), np.uint8)
    cv2.circle(mask, (ccol, crow), cutoff, 0, -1)

    # Apply mask
    fshift_filtered = fshift * mask
    return ifft_image(fshift_filtered)


def gaussian_low_pass(image, sigma=10):
    """
    Apply Gaussian Low-Pass Filter.

    Args:
        image (numpy.ndarray): Grayscale input image.
        sigma (float): Standard deviation for Gaussian filter.

    Returns:
        filtered_img (numpy.ndarray): Low-pass filtered image.
    """
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    fshift = fft_image(image)

    # Create Gaussian mask
    x = np.linspace(-ccol, ccol, cols)
    y = np.linspace(-crow, crow, rows)
    X, Y = np.meshgrid(x, y)
    mask = np.exp(-(X**2 + Y**2) / (2 * sigma**2))

    fshift_filtered = fshift * mask
    return ifft_image(fshift_filtered)


def gaussian_high_pass(image, sigma=10):
    """
    Apply Gaussian High-Pass Filter.

    Args:
        image (numpy.ndarray): Grayscale input image.
        sigma (float): Standard deviation for Gaussian filter.

    Returns:
        filtered_img (numpy.ndarray): High-pass filtered image.
    """
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    fshift = fft_image(image)

    # Create Gaussian mask
    x = np.linspace(-ccol, ccol, cols)
    y = np.linspace(-crow, crow, rows)
    X, Y = np.meshgrid(x, y)
    mask = 1 - np.exp(-(X**2 + Y**2) / (2 * sigma**2))

    fshift_filtered = fshift * mask
    return ifft_image(fshift_filtered)


if __name__ == "__main__":
    # Example testing (optional)
    from utils.io_utils import load_image
    import os

    test_path = "data/input/sample_leaf.jpg"

    if os.path.exists(test_path):
        img = load_image(test_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ilp = ideal_low_pass(gray, cutoff=30)
        ihp = ideal_high_pass(gray, cutoff=30)
        glp = gaussian_low_pass(gray, sigma=15)
        ghp = gaussian_high_pass(gray, sigma=15)

        show_images_side_by_side(gray, ilp, "Original", "Ideal Low-Pass")
        show_images_side_by_side(gray, ihp, "Original", "Ideal High-Pass")
        show_images_side_by_side(gray, glp, "Original", "Gaussian Low-Pass")
        show_images_side_by_side(gray, ghp, "Original", "Gaussian High-Pass")
    else:
        print("Test image not found! Please add one in 'data/input/'.")
