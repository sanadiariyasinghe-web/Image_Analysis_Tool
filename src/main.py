"""
main.py
--------
Entry point for the Domain-Specific Image Analysis & Enhancement Tool
(Agriculture domain: Plant leaves, crop fields, aerial farm images).
"""

import os
import cv2

# Import custom modules (directly inside src folder)
from acquisition_display import load_and_show_image, display_metadata
from color_conversion import bgr_to_grayscale, bgr_to_hsv, bgr_to_binary
from histogram_ops import plot_color_histogram, plot_grayscale_histogram, equalize_grayscale, equalize_color
from geometric_ops import rotate_image, scale_image, translate_image, crop_image
from filters_ops import apply_average_filter, apply_gaussian_filter, apply_median_filter, apply_sharpening
from morphology_ops import apply_erosion, apply_dilation, apply_opening, apply_closing
from segmentation_ops import global_threshold, otsu_threshold, watershed_segmentation
from frequency_ops import ideal_low_pass, ideal_high_pass, gaussian_low_pass, gaussian_high_pass
from utils.plot_utils import show_images_side_by_side

# -----------------------------
# Configuration
# -----------------------------
IMAGE_PATH = "./data/input/sample_leaf.jpg"

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

# -----------------------------
# 1. Image Acquisition
# -----------------------------
image = load_and_show_image(IMAGE_PATH)
display_metadata(image, IMAGE_PATH)

# -----------------------------
# 2. Color Conversions
# -----------------------------
gray = bgr_to_grayscale(image)
hsv = bgr_to_hsv(image)
binary = bgr_to_binary(gray, threshold=127)

show_images_side_by_side(image, gray, "Original", "Grayscale")
show_images_side_by_side(image, hsv, "Original", "HSV")
show_images_side_by_side(gray, binary, "Grayscale", "Binary")

# -----------------------------
# 3. Histogram Analysis
# -----------------------------
plot_color_histogram(image)
plot_grayscale_histogram(gray)

eq_gray = equalize_grayscale(gray)
eq_color = equalize_color(image)
show_images_side_by_side(gray, eq_gray, "Gray Original", "Gray Equalized")
show_images_side_by_side(image, eq_color, "Color Original", "Color Equalized")

# -----------------------------
# 4. Geometric Transformations
# -----------------------------
rotated = rotate_image(image, 45)
scaled = scale_image(image, 0.5, 0.5)
translated = translate_image(image, 50, 30)
cropped = crop_image(image, 50, 50, 200, 200)

show_images_side_by_side(image, rotated, "Original", "Rotated")
show_images_side_by_side(image, scaled, "Original", "Scaled")
show_images_side_by_side(image, translated, "Original", "Translated")
show_images_side_by_side(image, cropped, "Original", "Cropped")

# -----------------------------
# 5. Smoothing & Sharpening
# -----------------------------
avg = apply_average_filter(image, ksize=5)
gauss = apply_gaussian_filter(image, ksize=5)
median = apply_median_filter(image, ksize=5)
sharpened = apply_sharpening(image)

show_images_side_by_side(image, avg, "Original", "Averaging Filter")
show_images_side_by_side(image, gauss, "Original", "Gaussian Filter")
show_images_side_by_side(image, median, "Original", "Median Filter")
show_images_side_by_side(image, sharpened, "Original", "Sharpened")

# -----------------------------
# 6. Morphological Operations
# -----------------------------
eroded = apply_erosion(binary, ksize=3)
dilated = apply_dilation(binary, ksize=3)
opened = apply_opening(binary, ksize=3)
closed = apply_closing(binary, ksize=3)

show_images_side_by_side(binary, eroded, "Binary", "Erosion")
show_images_side_by_side(binary, dilated, "Binary", "Dilation")
show_images_side_by_side(binary, opened, "Binary", "Opening")
show_images_side_by_side(binary, closed, "Binary", "Closing")

# -----------------------------
# 7. Segmentation
# -----------------------------
global_seg = global_threshold(gray, thresh_value=127)
otsu_seg, otsu_val = otsu_threshold(gray)
watershed_seg = watershed_segmentation(image)

print(f"Otsu threshold value: {otsu_val:.2f}")
show_images_side_by_side(gray, global_seg, "Gray", "Global Threshold")
show_images_side_by_side(gray, otsu_seg, "Gray", "Otsu Threshold")
show_images_side_by_side(image, watershed_seg, "Original", "Watershed Segmentation")

# -----------------------------
# 8. Frequency Filtering
# -----------------------------
ilp = ideal_low_pass(gray, cutoff=30)
ihp = ideal_high_pass(gray, cutoff=30)
glp = gaussian_low_pass(gray, sigma=15)
ghp = gaussian_high_pass(gray, sigma=15)

show_images_side_by_side(gray, ilp, "Gray", "Ideal Low-Pass")
show_images_side_by_side(gray, ihp, "Gray", "Ideal High-Pass")
show_images_side_by_side(gray, glp, "Gray", "Gaussian Low-Pass")
show_images_side_by_side(gray, ghp, "Gray", "Gaussian High-Pass")

print("\n✅ All operations executed successfully!")
