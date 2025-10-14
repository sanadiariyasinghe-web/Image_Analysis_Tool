"""
plot_utils.py
--------------
Utility functions for displaying and plotting images and histograms
for the Domain-Specific Image Analysis & Enhancement Tool (Agriculture domain).

Functions:
    - show_image(title, image): Display an image using matplotlib.
    - show_images_side_by_side(img1, img2, title1, title2): Compare two images.
    - plot_histogram(image): Plot grayscale or color histogram of an image.
"""

import cv2
import matplotlib.pyplot as plt

def show_image(title, image):
    """
    Display an image using matplotlib.

    Args:
        title (str): Window or figure title.
        image (numpy.ndarray): Image array.
    """
    # Convert BGR (OpenCV) to RGB (Matplotlib)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)
    else:
        plt.imshow(image, cmap='gray')

    plt.title(title)
    plt.axis('off')
    plt.show()


def show_images_side_by_side(img1, img2, title1="Original", title2="Processed"):
    """
    Display two images side by side for comparison.

    Args:
        img1 (numpy.ndarray): First image.
        img2 (numpy.ndarray): Second image.
        title1 (str): Title for first image.
        title2 (str): Title for second image.
    """
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    if len(img1.shape) == 3:
        plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img1, cmap='gray')
    plt.title(title1)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    if len(img2.shape) == 3:
        plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img2, cmap='gray')
    plt.title(title2)
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def plot_histogram(image):
    """
    Plot the grayscale or color histogram of an image.

    Args:
        image (numpy.ndarray): Input image (grayscale or BGR).
    """
    plt.figure(figsize=(6, 4))

    if len(image.shape) == 2:
        # Grayscale image
        plt.hist(image.ravel(), 256, [0, 256], color='black')
        plt.title('Grayscale Histogram')
    else:
        # Color image
        colors = ('b', 'g', 'r')
        for i, color in enumerate(colors):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            plt.plot(hist, color=color)
        plt.title('Color Histogram')

    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
