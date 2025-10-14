"""
histogram_ops.py
----------------
Handles histogram analysis and contrast enhancement for the
Domain-Specific Image Analysis & Enhancement Tool (Agriculture domain).

Functions:
    - plot_grayscale_histogram(image): Plot histogram for grayscale image.
    - plot_color_histogram(image): Plot histograms for color channels.
    - equalize_grayscale(image): Apply histogram equalization to grayscale image.
    - equalize_color(image): Apply histogram equalization to each color channel.
"""

import cv2
import matplotlib.pyplot as plt
from utils.plot_utils import show_images_side_by_side


def plot_grayscale_histogram(image):
    """
    Plot histogram for a grayscale image.

    Args:
        image (numpy.ndarray): Grayscale image.
    """
    if len(image.shape) != 2:
        raise ValueError("Input image must be grayscale.")

    plt.figure(figsize=(6, 4))
    plt.hist(image.ravel(), bins=256, range=[0, 256], color='black')
    plt.title('Grayscale Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


def plot_color_histogram(image):
    """
    Plot histograms for color channels (B, G, R).

    Args:
        image (numpy.ndarray): BGR color image.
    """
    colors = ('b', 'g', 'r')
    plt.figure(figsize=(6, 4))

    for i, color in enumerate(colors):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=color, label=f'{color.upper()} channel')

    plt.title('Color Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()


def equalize_grayscale(image):
    """
    Apply histogram equalization to enhance contrast of a grayscale image.

    Args:
        image (numpy.ndarray): Input grayscale image.

    Returns:
        equalized (numpy.ndarray): Contrast-enhanced grayscale image.
    """
    if len(image.shape) != 2:
        raise ValueError("Histogram equalization requires a grayscale image.")

    equalized = cv2.equalizeHist(image)
    return equalized


def equalize_color(image):
    """
    Apply histogram equalization to each color channel separately.

    Args:
        image (numpy.ndarray): Input BGR color image.

    Returns:
        equalized (numpy.ndarray): Contrast-enhanced color image.
    """
    channels = cv2.split(image)
    eq_channels = [cv2.equalizeHist(ch) for ch in channels]
    equalized = cv2.merge(eq_channels)
    return equalized


# -----------------------------
# Example testing
# -----------------------------
if __name__ == "__main__":
    import os
    from utils.io_utils import load_image
    from color_conversion import bgr_to_grayscale

    test_path = "../data/input/sample_leaf.jpg"  # Update relative to src folder
    if os.path.exists(test_path):
        img = load_image(test_path)
        gray = bgr_to_grayscale(img)

        # Plot histograms
        plot_color_histogram(img)
        plot_grayscale_histogram(gray)

        # Apply equalization
        eq_gray = equalize_grayscale(gray)
        eq_color = equalize_color(img)

        # Show before-after results
        show_images_side_by_side(gray, eq_gray, "Original Gray", "Equalized Gray")
        show_images_side_by_side(img, eq_color, "Original Color", "Equalized Color")
    else:
        print("Test image not found! Please add one in '../data/input/'.")
