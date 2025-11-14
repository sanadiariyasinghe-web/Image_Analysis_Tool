"""
CustomTkinter GUI for Agriculture Image Analysis & Enhancement Tool
- Modern panel layout (left tools, right input/output)
- Buttons implement core PartA + many PartB operations
Save this file as src/gui.py and run with: python -m src.gui
"""

import os
import cv2
import numpy as np
import customtkinter as ctk
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox
from scipy import ndimage
import matplotlib.pyplot as plt

# ---------- Helper image utilities ----------
def to_pil(img_bgr):
    """Convert BGR (cv2) to PIL Image"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)

def resize_for_display(img_bgr, max_size=(480, 480)):
    pil = to_pil(img_bgr)
    pil.thumbnail(max_size, Image.ANTIALIAS)
    return pil

def ensure_color(image):
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image

# ---------- Image processing functions ----------
def bgr_to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def bgr_to_hsv_bgr(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)  # convert back to displayable BGR

def grayscale_to_binary(gray, thresh=127):
    _, binary = cv2.threshold(gray, int(thresh), 255, cv2.THRESH_BINARY)
    return binary

def plot_and_show_histogram(img):
    plt.figure(figsize=(5,3))
    if len(img.shape) == 2:
        plt.hist(img.ravel(), bins=256, range=(0,256))
        plt.title("Grayscale Histogram")
    else:
        color = ('b','g','r')
        for i,col in enumerate(color):
            hist = cv2.calcHist([img],[i],None,[256],[0,256])
            plt.plot(hist, label=col)
        plt.legend()
    plt.tight_layout()
    plt.show()

def equalize_grayscale(gray):
    return cv2.equalizeHist(gray)

def equalize_color(img):
    # equalize in YUV (better than naive per-channel)
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

def rotate_image(img, angle):
    h,w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2,h/2), angle, 1.0)
    return cv2.warpAffine(img, M, (w,h))

def scale_image(img, sx, sy):
    return cv2.resize(img, None, fx=sx, fy=sy, interpolation=cv2.INTER_LINEAR)

def translate_image(img, tx, ty):
    M = np.float32([[1,0,tx],[0,1,ty]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

def crop_image(img, x,y,w,h):
    return img[y:y+h, x:x+w].copy()

def apply_average(img, k=5):
    return cv2.blur(img, (k,k))

def apply_gaussian(img, k=5, sigma=0):
    return cv2.GaussianBlur(img, (k,k), sigma)

def apply_median(img, k=5):
    return cv2.medianBlur(img, k)

def apply_sharpen(img):
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    return cv2.filter2D(img, -1, kernel)

def sobel_edge(img):
    gray = bgr_to_grayscale(img)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = np.uint8(np.clip(mag,0,255))
    return cv2.cvtColor(mag, cv2.COLOR_GRAY2BGR)

def laplacian_edge(img):
    gray = bgr_to_grayscale(img)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap = np.uint8(np.clip(np.abs(lap),0,255))
    return cv2.cvtColor(lap, cv2.COLOR_GRAY2BGR)

def canny_edge(img, low=50, high=150):
    gray = bgr_to_grayscale(img)
    edges = cv2.Canny(gray, low, high)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

def morphology_op(binary, op, k=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k,k))
    if op == "erode":
        return cv2.erode(binary, kernel, iterations=1)
    if op == "dilate":
        return cv2.dilate(binary, kernel, iterations=1)
    if op == "open":
        return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    if op == "close":
        return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return binary

def global_threshold(gray, t=127):
    _, th = cv2.threshold(gray, int(t), 255, cv2.THRESH_BINARY)
    return th

def otsu_threshold(gray):
    _, th = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return th

def watershed_segmentation(img):
    # Basic watershed using distance transform — works for simple leaf images
    img_gray = bgr_to_grayscale(img)
    ret, thresh = cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2,5)
    ret2, sure_fg = cv2.threshold(dist_transform, 0.5*dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    ret3, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown==255] = 0
    img_copy = img.copy()
    cv2.watershed(img_copy, markers)
    img_copy[markers == -1] = [0,0,255]
    return img_copy

def ideal_low_pass(gray, cutoff=30):
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    rows, cols = gray.shape
    crow,ccol = rows//2, cols//2
    mask = np.zeros_like(gray)
    y,x = np.ogrid[:rows,:cols]
    mask_area = (x-ccol)**2 + (y-crow)**2 <= cutoff*cutoff
    mask[mask_area] = 1
    fshift2 = fshift * mask
    ishift = np.fft.ifftshift(fshift2)
    iimg = np.fft.ifft2(ishift)
    iimg = np.abs(iimg)
    iimg = np.uint8(np.clip(iimg,0,255))
    return iimg

def ideal_high_pass(gray, cutoff=30):
    low = ideal_low_pass(gray, cutoff)
    high = cv2.subtract(gray, low)
    return high

def gaussian_low_pass(gray, sigma=10):
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    rows, cols = gray.shape
    crow,ccol = rows//2, cols//2
    y,x = np.ogrid[:rows,:cols]
    gaussian = np.exp(-((x-ccol)**2+(y-crow)**2)/(2*(sigma**2)))
    fshift2 = fshift * gaussian
    ishift = np.fft.ifftshift(fshift2)
    iimg = np.fft.ifft2(ishift)
    iimg = np.abs(iimg)
    iimg = np.uint8(np.clip(iimg,0,255))
    return iimg

def gaussian_high_pass(gray, sigma=10):
    low = gaussian_low_pass(gray, sigma)
    high = cv2.subtract(gray, low)
    return high

# ---------- GUI ----------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Agriculture Image Analysis Tool")
        self.geometry("1200x720")
        self.minsize(1000,600)

        # State
        self.img_orig = None      # original BGR
        self.img_current = None   # current BGR
        self.img_output = None    # output BGR or gray (if gray stored as 2D)

        # Layout: left tool frame, right image frame
        self.grid_columnconfigure((0,1), weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Left tools panel (scrollable)
        self.tools_frame = ctk.CTkFrame(self, width=280)
        self.tools_frame.grid(row=0, column=0, sticky="nswe", padx=10, pady=10)
        self._build_tools(self.tools_frame)

        # Right image frame
        self.display_frame = ctk.CTkFrame(self)
        self.display_frame.grid(row=0, column=1, sticky="nswe", padx=10, pady=10)
        self._build_display(self.display_frame)

    def _build_tools(self, parent):
        # top: Upload + Save + Reset + Metadata
        top = ctk.CTkFrame(parent)
        top.pack(fill="x", pady=(8,10))
        ctk.CTkButton(top, text="Upload Image", command=self.load_image).pack(pady=6, padx=8, fill="x")
        ctk.CTkButton(top, text="Save Output", command=self.save_output).pack(pady=6, padx=8, fill="x")
        ctk.CTkButton(top, text="Reset to Original", command=self.reset_image).pack(pady=6, padx=8, fill="x")
        ctk.CTkButton(top, text="Show Metadata", command=self.show_metadata).pack(pady=6, padx=8, fill="x")

        # Processing groups (use collapsible frames visually)
        def add_button(label, cmd):
            btn = ctk.CTkButton(parent, text=label, command=cmd, corner_radius=6)
            btn.pack(fill="x", padx=8, pady=4)
            return btn

        # Color & Histogram
        ctk.CTkLabel(parent, text="Color & Histogram", fg_color=None, anchor="w").pack(fill="x", padx=8, pady=(12,2))
        add_button("Convert → Grayscale", self.do_grayscale)
        add_button("Convert → HSV (display)", self.do_hsv)
        add_button("Convert → Binary (127)", self.do_binary)
        add_button("Show Histogram", self.do_histogram)
        add_button("Equalize Grayscale", self.do_equalize_gray)
        add_button("Equalize Color", self.do_equalize_color)

        # Geometric
        ctk.CTkLabel(parent, text="Geometric", anchor="w").pack(fill="x", padx=8, pady=(12,2))
        # rotate / scale inputs
        self.rotate_entry = ctk.CTkEntry(parent, placeholder_text="Angle (deg) e.g. 45")
        self.rotate_entry.pack(fill="x", padx=8, pady=4)
        add_button("Rotate", self.do_rotate)

        self.scale_entry = ctk.CTkEntry(parent, placeholder_text="Scale (sx,sy) e.g. 0.5,0.5")
        self.scale_entry.pack(fill="x", padx=8, pady=4)
        add_button("Scale", self.do_scale)

        self.translate_entry = ctk.CTkEntry(parent, placeholder_text="Translate tx,ty e.g. 50,30")
        self.translate_entry.pack(fill="x", padx=8, pady=4)
        add_button("Translate", self.do_translate)

        self.crop_entry = ctk.CTkEntry(parent, placeholder_text="Crop x,y,w,h e.g. 50,50,200,200")
        self.crop_entry.pack(fill="x", padx=8, pady=4)
        add_button("Crop", self.do_crop)

        # Filters
        ctk.CTkLabel(parent, text="Smoothing & Sharpening", anchor="w").pack(fill="x", padx=8, pady=(12,2))
        self.kernel_entry = ctk.CTkEntry(parent, placeholder_text="Kernel size odd e.g. 3 or 5")
        self.kernel_entry.pack(fill="x", padx=8, pady=4)
        add_button("Averaging Filter", self.do_average)
        add_button("Gaussian Filter", self.do_gaussian)
        add_button("Median Filter", self.do_median)
        add_button("Sharpen", self.do_sharpen)

        # Edges
        ctk.CTkLabel(parent, text="Edge Detection", anchor="w").pack(fill="x", padx=8, pady=(12,2))
        add_button("Sobel", self.do_sobel)
        add_button("Laplacian", self.do_laplacian)
        add_button("Canny", self.do_canny)

        # Morphology
        ctk.CTkLabel(parent, text="Morphological Ops (use binary)", anchor="w").pack(fill="x", padx=8, pady=(12,2))
        add_button("Erode", lambda: self.do_morph("erode"))
        add_button("Dilate", lambda: self.do_morph("dilate"))
        add_button("Open", lambda: self.do_morph("open"))
        add_button("Close", lambda: self.do_morph("close"))

        # Segmentation
        ctk.CTkLabel(parent, text="Segmentation", anchor="w").pack(fill="x", padx=8, pady=(12,2))
        add_button("Global Threshold (127)", lambda: self.do_global_thresh(127))
        add_button("Otsu Threshold", self.do_otsu)
        add_button("Watershed", self.do_watershed)

        # Frequency
        ctk.CTkLabel(parent, text="Frequency Filtering (grayscale)", anchor="w").pack(fill="x", padx=8, pady=(12,2))
        self.cutoff_entry = ctk.CTkEntry(parent, placeholder_text="cutoff or sigma e.g. 30")
        self.cutoff_entry.pack(fill="x", padx=8, pady=4)
        add_button("Ideal LPF", lambda: self.do_ideal_lpf())
        add_button("Ideal HPF", lambda: self.do_ideal_hpf())
        add_button("Gaussian LPF", lambda: self.do_gaussian_lpf())
        add_button("Gaussian HPF", lambda: self.do_gaussian_hpf)

        # Compare / Reset / Save
        ctk.CTkLabel(parent, text="", anchor="w").pack(fill="x", padx=8, pady=(8,2))

    def _build_display(self, parent):
        # left subframe: input image title + canvas
        left = ctk.CTkFrame(parent)
        left.pack(side="left", fill="both", expand=True, padx=8, pady=8)

        right = ctk.CTkFrame(parent)
        right.pack(side="right", fill="both", expand=True, padx=8, pady=8)

        ctk.CTkLabel(left, text="Input Image").pack()
        self.input_label = ctk.CTkLabel(left, text="", width=480, height=480, fg_color=("white","gray20"))
        self.input_label.pack(expand=True, padx=8, pady=8)

        ctk.CTkLabel(right, text="Output Image").pack()
        self.output_label = ctk.CTkLabel(right, text="", width=480, height=480, fg_color=("white","gray20"))
        self.output_label.pack(expand=True, padx=8, pady=8)

    # ---------- UI actions ----------
    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.tif *.bmp")])
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", "Failed to load image.")
            return
        self.img_orig = img.copy()
        self.img_current = img.copy()
        self.img_output = None
        self.show_input()
        self.clear_output_label()

    def save_output(self):
        if self.img_output is None and self.img_current is None:
            messagebox.showinfo("Info", "No output to save.")
            return
        img_to_save = self.img_output if self.img_output is not None else self.img_current
        save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG","*.png"),("JPG","*.jpg")])
        if not save_path:
            return
        # if gray 2D array convert to BGR for writing
        if len(img_to_save.shape) == 2:
            img_to_save = cv2.cvtColor(img_to_save, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(save_path, img_to_save)
        messagebox.showinfo("Saved", f"Saved to {save_path}")

    def reset_image(self):
        if self.img_orig is not None:
            self.img_current = self.img_orig.copy()
            self.img_output = None
            self.show_input()
            self.clear_output_label()

    def show_metadata(self):
        if self.img_current is None:
            messagebox.showinfo("Info", "Load an image first.")
            return
        h,w = self.img_current.shape[:2]
        channels = 1 if len(self.img_current.shape)==2 else self.img_current.shape[2]
        # color depth approximate
        dtype = self.img_current.dtype
        file_size = "N/A"
        messagebox.showinfo("Metadata", f"Width: {w}\nHeight: {h}\nChannels: {channels}\nDType: {dtype}\nFile size: {file_size}")

    def show_input(self):
        if self.img_current is None:
            return
        pil = resize_for_display(self.img_current)
        imgtk = ImageTk.PhotoImage(pil)
        self.input_label.image = imgtk
        self.input_label.configure(image=imgtk)

    def show_output(self, out_img):
        if out_img is None:
            self.clear_output_label()
            return
        # if gray convert to BGR for display
        if len(out_img.shape) == 2:
            out_disp = cv2.cvtColor(out_img, cv2.COLOR_GRAY2BGR)
        else:
            out_disp = out_img
        self.img_output = out_img.copy()
        pil = resize_for_display(out_disp)
        imgtk = ImageTk.PhotoImage(pil)
        self.output_label.image = imgtk
        self.output_label.configure(image=imgtk)

    def clear_output_label(self):
        self.output_label.configure(image="", text="")

    # ---------- Operation wrappers ----------
    def do_grayscale(self):
        if self.img_current is None: return
        gray = bgr_to_grayscale(self.img_current)
        self.show_output(gray)

    def do_hsv(self):
        if self.img_current is None: return
        hsv_bgr = bgr_to_hsv_bgr(self.img_current)
        self.show_output(hsv_bgr)

    def do_binary(self):
        if self.img_current is None: return
        gray = bgr_to_grayscale(self.img_current)
        binary = grayscale_to_binary(gray, 127)
        self.show_output(binary)

    def do_histogram(self):
        if self.img_current is None: return
        img = self.img_current.copy()
        plot_and_show_histogram(img)

    def do_equalize_gray(self):
        if self.img_current is None: return
        gray = bgr_to_grayscale(self.img_current)
        eq = equalize_grayscale(gray)
        self.show_output(eq)

    def do_equalize_color(self):
        if self.img_current is None: return
        eq = equalize_color(self.img_current)
        self.show_output(eq)

    def do_rotate(self):
        if self.img_current is None: return
        try:
            angle = float(self.rotate_entry.get())
        except:
            messagebox.showerror("Error", "Invalid angle")
            return
        out = rotate_image(self.img_current, angle)
        self.show_output(out)

    def do_scale(self):
        if self.img_current is None: return
        try:
            sx,sy = self.scale_entry.get().split(",")
            sx = float(sx.strip()); sy = float(sy.strip())
        except:
            messagebox.showerror("Error", "Scale must be two floats: sx,sy")
            return
        out = scale_image(self.img_current, sx, sy)
        self.show_output(out)

    def do_translate(self):
        if self.img_current is None: return
        try:
            tx,ty = self.translate_entry.get().split(",")
            tx = int(tx.strip()); ty = int(ty.strip())
        except:
            messagebox.showerror("Error", "Translate must be two ints: tx,ty")
            return
        out = translate_image(self.img_current, tx, ty)
        self.show_output(out)

    def do_crop(self):
        if self.img_current is None: return
        try:
            x,y,w,h = [int(x.strip()) for x in self.crop_entry.get().split(",")]
        except:
            messagebox.showerror("Error", "Crop must be x,y,w,h (ints)")
            return
        out = crop_image(self.img_current, x,y,w,h)
        self.show_output(out)

    def do_average(self):
        if self.img_current is None: return
        try:
            k = int(self.kernel_entry.get())
        except:
            k = 5
        out = apply_average(self.img_current, max(1,k))
        self.show_output(out)

    def do_gaussian(self):
        if self.img_current is None: return
        try:
            k = int(self.kernel_entry.get())
        except:
            k = 5
        if k % 2 == 0: k += 1
        out = apply_gaussian(self.img_current, k)
        self.show_output(out)

    def do_median(self):
        if self.img_current is None: return
        try:
            k = int(self.kernel_entry.get())
        except:
            k = 5
        if k % 2 == 0: k += 1
        out = apply_median(self.img_current, max(3,k))
        self.show_output(out)

    def do_sharpen(self):
        if self.img_current is None: return
        out = apply_sharpen(self.img_current)
        self.show_output(out)

    def do_sobel(self):
        if self.img_current is None: return
        out = sobel_edge(self.img_current)
        self.show_output(out)

    def do_laplacian(self):
        if self.img_current is None: return
        out = laplacian_edge(self.img_current)
        self.show_output(out)

    def do_canny(self):
        if self.img_current is None: return
        out = canny_edge(self.img_current, 50,150)
        self.show_output(out)

    def do_morph(self, op):
        if self.img_current is None: return
        gray = bgr_to_grayscale(self.img_current)
        binary = grayscale_to_binary(gray,127)
        out = morphology_op(binary, op, k=3)
        self.show_output(out)

    def do_global_thresh(self, t=127):
        if self.img_current is None: return
        gray = bgr_to_grayscale(self.img_current)
        out = global_threshold(gray, t)
        self.show_output(out)

    def do_otsu(self):
        if self.img_current is None: return
        gray = bgr_to_grayscale(self.img_current)
        out = otsu_threshold(gray)
        self.show_output(out)

    def do_watershed(self):
        if self.img_current is None: return
        out = watershed_segmentation(self.img_current)
        self.show_output(out)

    def do_ideal_lpf(self):
        if self.img_current is None: return
        try:
            cutoff = int(self.cutoff_entry.get())
        except:
            cutoff = 30
        gray = bgr_to_grayscale(self.img_current)
        out = ideal_low_pass(gray, cutoff)
        self.show_output(out)

    def do_ideal_hpf(self):
        if self.img_current is None: return
        try:
            cutoff = int(self.cutoff_entry.get())
        except:
            cutoff = 30
        gray = bgr_to_grayscale(self.img_current)
        out = ideal_high_pass(gray, cutoff)
        self.show_output(out)

    def do_gaussian_lpf(self):
        if self.img_current is None: return
        try:
            sigma = float(self.cutoff_entry.get())
        except:
            sigma = 15.0
        gray = bgr_to_grayscale(self.img_current)
        out = gaussian_low_pass(gray, sigma)
        self.show_output(out)

    def do_gaussian_hpf(self):
        if self.img_current is None: return
        try:
            sigma = float(self.cutoff_entry.get())
        except:
            sigma = 15.0
        gray = bgr_to_grayscale(self.img_current)
        out = gaussian_high_pass(gray, sigma)
        self.show_output(out)

# Run
if __name__ == "__main__":
    app = App()
    app.mainloop()
