"""
segmentation_ops.py
-------------------
Implements segmentation techniques for the Domain-Specific Image Analysis
& Enhancement Tool (Agriculture domain).

Techniques:
    1. Global Thresholding
    2. Otsu’s Thresholding
    3. Region-based Segmentation (Watershed)
"""

import cv2
import numpy as np
from utils.plot_utils import show_images_side_by_side


def global_threshold(image, thresh_value=127):
    """
    Apply simple global thresholding.

    Args:
        image (numpy.ndarray): Grayscale input image.
        thresh_value (int): Threshold value (0–255).

    Returns:
        thresh_img (numpy.ndarray): Binary thresholded image.
    """
    _, thresh_img = cv2.threshold(image, thresh_value, 255, cv2.THRESH_BINARY)
    return thresh_img


def otsu_threshold(image):
    """
    Apply Otsu’s thresholding to automatically determine the best threshold value.

    Args:
        image (numpy.ndarray): Grayscale input image.

    Returns:
        otsu_img (numpy.ndarray): Binary image after Otsu’s thresholding.
        otsu_value (float): Computed optimal threshold value.
    """
    otsu_value, otsu_img = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return otsu_img, otsu_value


def watershed_segmentation(image):
    """
    Apply Watershed algorithm for region-based segmentation.
    Useful for separating overlapping plant leaves or field regions.

    Args:
        image (numpy.ndarray): Original BGR image.

    Returns:
        segmented (numpy.ndarray): Image with segmented boundaries highlighted.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Otsu thresholding to get binary mask
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Noise removal using Opening
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # Sure background (dilation)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Distance transform and threshold to get sure foreground
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Unknown region = background - foreground
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labeling
    _, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so background is not zero
    markers = markers + 1

    # Mark unknown regions with zero
    markers[unknown == 255] = 0

    # Apply watershed
    markers = cv2.watershed(image, markers)

    # Highlight boundaries
    segmented = image.copy()
    segmented[markers == -1] = [0, 0, 255]  # Mark boundaries in red

    return segmented


if __name__ == "__main__":
    # Example testing (optional)
    from utils.io_utils import load_image
    import os

    test_path = "data/samples/sample_leaf.jpg"  # Example path

    if os.path.exists(test_path):
        img = load_image(test_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. Global threshold
        global_img = global_threshold(gray, 120)

        # 2. Otsu’s threshold
        otsu_img, otsu_val = otsu_threshold(gray)
        print(f"Otsu computed threshold: {otsu_val:.2f}")

        # 3. Watershed segmentation
        watershed_img = watershed_segmentation(img)

        show_images_side_by_side(gray, global_img, "Grayscale", "Global Threshold")
        show_images_side_by_side(gray, otsu_img, "Grayscale", "Otsu Threshold")
        show_images_side_by_side(img, watershed_img, "Original", "Watershed Segmentation")
    else:
        print("Sample image not found! Please add one in 'data/samples/'.")
