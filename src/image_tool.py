#!/usr/bin/env python3
"""
image_tool.py
Modern Agriculture Image Analysis & Enhancement Tool
Wide Sidebar (250px) + Real Icons + Tab-Based UI
Single-file version
"""

import os
import cv2
import numpy as np
import customtkinter as ctk
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt

# Dark theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

# ---------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------
def imread_fallback(path):
    try:
        img = cv2.imread(path)
        if img is not None:
            return img
    except: pass
    try:
        with open(path, "rb") as f:
            data = f.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    except:
        return None

def to_pil_from_bgr(img_bgr):
    if img_bgr is None:
        return None
    if img_bgr.ndim == 2:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def pil_thumbnail_for_display(pil_img, max_size=(480,480)):
    if pil_img is None:
        return None
    p = pil_img.copy()
    p.thumbnail(max_size, Image.LANCZOS)
    return p

def ensure_bgr(img):
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

def safe_plot_show():
    try:
        plt.show()
    except:
        plt.show(block=True)

# ---------------------------------------------------------
# Processing Functions
# ---------------------------------------------------------
def to_gray(img): return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def to_hsv_bgr(img):
    return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), cv2.COLOR_HSV2BGR)

def gray_to_binary(gray, thresh=127):
    _, th = cv2.threshold(gray, int(thresh), 255, cv2.THRESH_BINARY)
    return th

def plot_color_histogram(img):
    plt.figure(figsize=(6,3))
    for i,c in enumerate(("b","g","r")):
        hist = cv2.calcHist([img],[i],None,[256],[0,256])
        plt.plot(hist, color=c)
    plt.title("Color Histogram"); plt.tight_layout(); safe_plot_show()

def plot_gray_histogram(gray):
    plt.figure(figsize=(5,3))
    plt.hist(gray.ravel(),256,[0,256])
    plt.title("Grayscale Histogram"); plt.tight_layout(); safe_plot_show()

def equalize_gray(gray): return cv2.equalizeHist(gray)

def equalize_color(img):
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

def rotate_image(img, angle):
    h,w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2,h/2), float(angle), 1)
    return cv2.warpAffine(img, M, (w,h), borderMode=cv2.BORDER_REFLECT)

def scale_image(img, sx, sy):
    return cv2.resize(img, None, fx=float(sx), fy=float(sy))

def translate_image(img, tx, ty):
    M = np.float32([[1,0,int(tx)],[0,1,int(ty)]])
    return cv2.warpAffine(img,M,(img.shape[1],img.shape[0]),borderMode=cv2.BORDER_REFLECT)

def crop_image(img,x,y,w,h):
    return img[int(y):int(y+h), int(x):int(x+w)].copy()

def average_filter(img,k=5): return cv2.blur(img,(int(k),int(k)))
def gaussian_filter(img,k=5,sigma=0):
    if int(k)%2==0: k+=1
    return cv2.GaussianBlur(img,(int(k),int(k)),sigma)
def median_filter(img,k=5):
    if int(k)%2==0: k+=1
    return cv2.medianBlur(img,int(k))

def sharpen(img):
    kernel=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    return cv2.filter2D(img,-1,kernel)

def sobel_edge(img):
    g=to_gray(img)
    gx=cv2.Sobel(g,cv2.CV_64F,1,0)
    gy=cv2.Sobel(g,cv2.CV_64F,0,1)
    mag=np.sqrt(gx*gx+gy*gy)
    if mag.max()>0: mag=np.uint8((mag/mag.max())*255)
    return cv2.cvtColor(mag,cv2.COLOR_GRAY2BGR)

def laplacian_edge(img):
    g=to_gray(img)
    lap=cv2.Laplacian(g,cv2.CV_64F)
    lap=np.uint8(np.clip(np.abs(lap),0,255))
    return cv2.cvtColor(lap,cv2.COLOR_GRAY2BGR)

def canny_edge(img,low=50,high=150):
    g=to_gray(img)
    e=cv2.Canny(g,int(low),int(high))
    return cv2.cvtColor(e,cv2.COLOR_GRAY2BGR)

def morphology_op(binary,op="erode",k=3):
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(int(k),int(k)))
    if op=="erode": return cv2.erode(binary,kernel)
    if op=="dilate": return cv2.dilate(binary,kernel)
    if op=="open": return cv2.morphologyEx(binary,cv2.MORPH_OPEN,kernel)
    if op=="close": return cv2.morphologyEx(binary,cv2.MORPH_CLOSE,kernel)
    return binary

def global_threshold(gray,t=127):
    _,th=cv2.threshold(gray,int(t),255,cv2.THRESH_BINARY)
    return th

def otsu_threshold(gray):
    _,th=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return th

def watershed_segmentation(img):
    gray=to_gray(img)
    _,th=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel=np.ones((3,3),np.uint8)
    opening=cv2.morphologyEx(th,cv2.MORPH_OPEN,kernel,iterations=2)
    sure_bg=cv2.dilate(opening,kernel,iterations=3)
    dist=cv2.distanceTransform(opening,cv2.DIST_L2,5)
    _,sure_fg=cv2.threshold(dist,0.5*dist.max(),255,0)
    sure_fg=np.uint8(sure_fg)
    unknown=cv2.subtract(sure_bg,sure_fg)
    _,markers=cv2.connectedComponents(sure_fg)
    markers=markers+1
    markers[unknown==255]=0
    img2=img.copy()
    cv2.watershed(img2,markers)
    img2[markers==-1]=[0,0,255]
    return img2

def ideal_low_pass(gray,c=30):
    rows,cols=gray.shape
    crow,ccol=rows//2,cols//2
    f=np.fft.fftshift(np.fft.fft2(gray))
    Y,X=np.ogrid[:rows,:cols]
    mask=((X-ccol)**2+(Y-crow)**2)<=c*c
    back=np.abs(np.fft.ifft2(np.fft.ifftshift(f*mask)))
    if back.max()>0: back=np.uint8(back/back.max()*255)
    return back

def ideal_high_pass(gray,c=30): return cv2.subtract(gray, ideal_low_pass(gray,c))
def gaussian_low_pass(gray, s=10):
    rows,cols=gray.shape
    crow,ccol=rows//2,cols//2
    f=np.fft.fftshift(np.fft.fft2(gray))
    Y,X=np.ogrid[:rows,:cols]
    g=np.exp(-((X-ccol)**2+(Y-crow)**2)/(2*s*s))
    back=np.abs(np.fft.ifft2(np.fft.ifftshift(f*g)))
    if back.max()>0: back=np.uint8(back/back.max()*255)
    return back

def gaussian_high_pass(gray,s=10): return cv2.subtract(gray, gaussian_low_pass(gray,s))

# ---------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------
class ImageToolApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Image Analysis Tool")
        self.geometry("1280x760")

        self.img_orig=None
        self.img_current=None
        self.img_output=None
        self.img_path=None

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ---------------- Sidebar ----------------
        sidebar = ctk.CTkFrame(self, width=250)
        sidebar.grid(row=0, column=0, sticky="ns")
        sidebar.grid_propagate(False)

        self.tabs = [
            "📁  Acquisition",
            "🎨  Color",
            "📐  Geometry",
            "✂  Filters",
            "⚙  Morphology",      # <-- ADDED
            "🔍  Segmentation",
            "🌐  Frequency"
        ]

        self.current_tab = None
        for t in self.tabs:
            btn = ctk.CTkButton(
                sidebar, text=t, height=48, anchor="w",
                command=lambda x=t: self.switch_tab(x)
            )
            btn.pack(fill="x", padx=10, pady=4)

        # ---------------- Right Panel (Images) ----------------
        self.right = ctk.CTkFrame(self)
        self.right.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.right.grid_columnconfigure((0,1), weight=1)
        self.right.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(self.right, text="Input Image").grid(row=0, column=0)
        ctk.CTkLabel(self.right, text="Output Image").grid(row=0, column=1)

        self.input_disp = ctk.CTkLabel(self.right, text="", fg_color=("gray20"), width=500, height=500) 
        self.output_disp = ctk.CTkLabel(self.right, text="", fg_color=("gray20"), width=500, height=500)

        self.input_disp.grid(row=1,column=0,sticky="nsew",padx=6,pady=6)
        self.output_disp.grid(row=1,column=1,sticky="nsew",padx=6,pady=6)

        # TAB AREA
        self.tab_area = ctk.CTkFrame(sidebar, fg_color="transparent")
        self.tab_area.pack(fill="both", expand=True, padx=6, pady=6)

        self.build_tabs()
        self.switch_tab("📁  Acquisition")

    # ---------------- Tab Switching ----------------
    def switch_tab(self, name):
        if self.current_tab:
            self.current_tab.pack_forget()
        self.current_tab = self.frames[name]
        self.current_tab.pack(fill="both", expand=True)

    # ---------------- Build All Tabs ----------------
    def build_tabs(self):
        self.frames = {}

        # ----- Acquisition -----
        f = ctk.CTkFrame(self.tab_area)
        self.frames["📁  Acquisition"] = f
        ctk.CTkButton(f,text="Upload Image",command=self.load_image).pack(fill="x", pady=6)
        ctk.CTkButton(f,text="Save Output",command=self.save_output).pack(fill="x", pady=6)
        ctk.CTkButton(f,text="Reset to Original",command=self.reset_image).pack(fill="x", pady=6)
        ctk.CTkButton(f,text="Show Metadata",command=self.show_metadata).pack(fill="x", pady=6)

        # ----- Color -----
        f = ctk.CTkFrame(self.tab_area)
        self.frames["🎨  Color"] = f
        ctk.CTkButton(f,text="Convert → Gray",command=self.act_grayscale).pack(fill="x", pady=6)
        ctk.CTkButton(f,text="Convert → HSV",command=self.act_hsv).pack(fill="x", pady=6)
        ctk.CTkButton(f,text="Convert → Binary",command=self.act_binary).pack(fill="x", pady=6)
        ctk.CTkButton(f,text="Show Histogram",command=self.act_histogram).pack(fill="x", pady=6)
        ctk.CTkButton(f,text="Equalize Gray",command=self.act_equalize_gray).pack(fill="x", pady=6)
        ctk.CTkButton(f,text="Equalize Color",command=self.act_equalize_color).pack(fill="x", pady=6)

        # ----- Geometry -----
        f = ctk.CTkFrame(self.tab_area)
        self.frames["📐  Geometry"] = f
        self.entry_rotate = ctk.CTkEntry(f, placeholder_text="Angle 45")
        self.entry_scale = ctk.CTkEntry(f, placeholder_text="sx,sy 0.5,0.5")
        self.entry_trans = ctk.CTkEntry(f, placeholder_text="tx,ty 50,30")
        self.entry_crop = ctk.CTkEntry(f, placeholder_text="x,y,w,h 50,50,200,200")
        self.entry_rotate.pack(fill="x", pady=4)
        ctk.CTkButton(f,text="Rotate",command=self.act_rotate).pack(fill="x", pady=4)
        self.entry_scale.pack(fill="x", pady=4)
        ctk.CTkButton(f,text="Scale",command=self.act_scale).pack(fill="x", pady=4)
        self.entry_trans.pack(fill="x", pady=4)
        ctk.CTkButton(f,text="Translate",command=self.act_translate).pack(fill="x", pady=4)
        self.entry_crop.pack(fill="x", pady=4)
        ctk.CTkButton(f,text="Crop",command=self.act_crop).pack(fill="x", pady=4)

        # ----- Filters -----
        f = ctk.CTkFrame(self.tab_area)
        self.frames["✂  Filters"] = f
        self.entry_k = ctk.CTkEntry(f, placeholder_text="Kernel size 5")
        self.entry_k.pack(fill="x", pady=4)
        ctk.CTkButton(f,text="Average",command=self.act_average).pack(fill="x", pady=4)
        ctk.CTkButton(f,text="Gaussian",command=self.act_gaussian).pack(fill="x", pady=4)
        ctk.CTkButton(f,text="Median",command=self.act_median).pack(fill="x", pady=4)
        ctk.CTkButton(f,text="Sharpen",command=self.act_sharpen).pack(fill="x", pady=4)
        ctk.CTkButton(f,text="Sobel Edge",command=self.act_sobel).pack(fill="x", pady=4)
        ctk.CTkButton(f,text="Laplacian Edge",command=self.act_laplacian).pack(fill="x", pady=4)
        ctk.CTkButton(f,text="Canny Edge",command=self.act_canny).pack(fill="x", pady=4)

        # ----- Morphology (ADDED) -----
        f = ctk.CTkFrame(self.tab_area)
        self.frames["⚙  Morphology"] = f

        self.entry_morph = ctk.CTkEntry(f, placeholder_text="Kernel size 3")
        self.entry_morph.pack(fill="x", pady=6)

        ctk.CTkButton(f,text="Erosion",command=lambda:self.act_morph("erode")).pack(fill="x", pady=6)
        ctk.CTkButton(f,text="Dilation",command=lambda:self.act_morph("dilate")).pack(fill="x", pady=6)
        ctk.CTkButton(f,text="Opening",command=lambda:self.act_morph("open")).pack(fill="x", pady=6)
        ctk.CTkButton(f,text="Closing",command=lambda:self.act_morph("close")).pack(fill="x", pady=6)

        # ----- Segmentation -----
        f = ctk.CTkFrame(self.tab_area)
        self.frames["🔍  Segmentation"] = f
        ctk.CTkButton(f,text="Global Threshold",command=self.act_global).pack(fill="x", pady=6)
        ctk.CTkButton(f,text="Otsu Threshold",command=self.act_otsu).pack(fill="x", pady=6)
        ctk.CTkButton(f,text="Watershed",command=self.act_watershed).pack(fill="x", pady=6)

        # ----- Frequency -----
        f = ctk.CTkFrame(self.tab_area)
        self.frames["🌐  Frequency"] = f
        self.entry_cut = ctk.CTkEntry(f, placeholder_text="Cutoff/Sigma")
        self.entry_cut.pack(fill="x", pady=4)
        ctk.CTkButton(f,text="Ideal LPF",command=self.act_ideal_lpf).pack(fill="x", pady=4)
        ctk.CTkButton(f,text="Ideal HPF",command=self.act_ideal_hpf).pack(fill="x", pady=4)
        ctk.CTkButton(f,text="Gaussian LPF",command=self.act_gaussian_lpf).pack(fill="x", pady=4)
        ctk.CTkButton(f,text="Gaussian HPF",command=self.act_gaussian_hpf).pack(fill="x", pady=4)

    # ---------------- IMAGE DISPLAY ----------------
    def show_input(self):
        pil=pil_thumbnail_for_display(to_pil_from_bgr(self.img_current))
        if pil is None:return
        imgtk=ImageTk.PhotoImage(pil)
        self.input_disp.configure(image=imgtk)
        self.input_disp.image=imgtk

    def show_output(self,out):
        if out is None:return
        if out.ndim==2: out=cv2.cvtColor(out,cv2.COLOR_GRAY2BGR)
        pil=pil_thumbnail_for_display(to_pil_from_bgr(out))
        imgtk=ImageTk.PhotoImage(pil)
        self.output_disp.configure(image=imgtk)
        self.output_disp.image=imgtk
        self.img_output=out.copy()

    def clear_output(self):
        self.output_disp.configure(image="")
        self.output_disp.image=None

    # ---------------- File Ops ----------------
    def load_image(self):
        path=filedialog.askopenfilename(filetypes=[("Images","*.jpg *.png *.jpeg *.bmp *.tiff")])
        if not path: return
        img=imread_fallback(path)
        if img is None: messagebox.showerror("Err","Load failed");return
        img=ensure_bgr(img)
        self.img_path=path
        self.img_orig=img.copy()
        self.img_current=img.copy()
        self.img_output=None
        self.show_input()
        self.clear_output()

    def save_output(self):
        if self.img_output is None and self.img_current is None:
            return
        img=self.img_output if self.img_output is not None else self.img_current
        path=filedialog.asksaveasfilename(defaultextension=".png")
        if not path:return
        if img.ndim==2: img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        cv2.imwrite(path,img)

    def reset_image(self):
        if self.img_orig is None:return
        self.img_current=self.img_orig.copy()
        self.img_output=None
        self.show_input()
        self.clear_output()

    def show_metadata(self):
        if self.img_current is None:return
        h,w=self.img_current.shape[:2]
        c=1 if self.img_current.ndim==2 else self.img_current.shape[2]
        messagebox.showinfo("Meta", f"{w}x{h}\nChannels:{c}")

    # ---------------- FILTER ACTIONS ----------------
    def act_grayscale(self):
        if self.img_current is None:return
        self.show_output(to_gray(self.img_current))

    def act_hsv(self):
        if self.img_current is None:return
        self.show_output(to_hsv_bgr(self.img_current))

    def act_binary(self):
        if self.img_current is None:return
        g=to_gray(self.img_current)
        self.show_output(gray_to_binary(g,127))

    def act_histogram(self):
        if self.img_current is None:return
        if self.img_current.ndim==3:
            plot_color_histogram(self.img_current)
        else:
            plot_gray_histogram(self.img_current)

    def act_equalize_gray(self):
        if self.img_current is None:return
        g=to_gray(self.img_current)
        self.show_output(equalize_gray(g))

    def act_equalize_color(self):
        if self.img_current is None:return
        self.show_output(equalize_color(self.img_current))

    def act_rotate(self):
        if self.img_current is None:return
        try: angle=float(self.entry_rotate.get())
        except: angle=0
        self.show_output(rotate_image(self.img_current,angle))

    def act_scale(self):
        if self.img_current is None:return
        try:
            sx,sy=self.entry_scale.get().split(",")
            sx,sy=float(sx),float(sy)
        except:
            sx=sy=1
        self.show_output(scale_image(self.img_current,sx,sy))

    def act_translate(self):
        if self.img_current is None:return
        try:
            tx,ty=self.entry_trans.get().split(",")
            tx,ty=int(tx),int(ty)
        except:
            tx=ty=0
        self.show_output(translate_image(self.img_current,tx,ty))

    def act_crop(self):
        if self.img_current is None:return
        try:
            x,y,w,h=[int(v) for v in self.entry_crop.get().split(",")]
        except:
            return
        self.show_output(crop_image(self.img_current,x,y,w,h))

    def act_average(self):
        if self.img_current is None:return
        try:k=int(self.entry_k.get())
        except:k=5
        self.show_output(average_filter(self.img_current,k))

    def act_gaussian(self):
        if self.img_current is None:return
        try:k=int(self.entry_k.get())
        except:k=5
        self.show_output(gaussian_filter(self.img_current,k))

    def act_median(self):
        if self.img_current is None:return
        try:k=int(self.entry_k.get())
        except:k=5
        self.show_output(median_filter(self.img_current,k))

    def act_sharpen(self):
        if self.img_current is None:return
        self.show_output(sharpen(self.img_current))

    def act_sobel(self):
        if self.img_current is None:return
        self.show_output(sobel_edge(self.img_current))

    def act_laplacian(self):
        if self.img_current is None:return
        self.show_output(laplacian_edge(self.img_current))

    def act_canny(self):
        if self.img_current is None:return
        self.show_output(canny_edge(self.img_current,50,150))

    def act_morph(self, mode):
        if self.img_current is None:return
        try: k=int(self.entry_morph.get())
        except: k=3
        g=to_gray(self.img_current)
        _,b=cv2.threshold(g,127,255,cv2.THRESH_BINARY)
        self.show_output(morphology_op(b,mode,k))

    def act_global(self):
        if self.img_current is None:return
        g=to_gray(self.img_current)
        self.show_output(global_threshold(g,127))

    def act_otsu(self):
        if self.img_current is None:return
        g=to_gray(self.img_current)
        self.show_output(otsu_threshold(g))

    def act_watershed(self):
        if self.img_current is None:return
        self.show_output(watershed_segmentation(self.img_current))

    def act_ideal_lpf(self):
        if self.img_current is None:return
        try:c=int(self.entry_cut.get())
        except:c=30
        g=to_gray(self.img_current)
        self.show_output(ideal_low_pass(g,c))

    def act_ideal_hpf(self):
        if self.img_current is None:return
        try:c=int(self.entry_cut.get())
        except:c=30
        g=to_gray(self.img_current)
        self.show_output(ideal_high_pass(g,c))

    def act_gaussian_lpf(self):
        if self.img_current is None:return
        try:s=float(self.entry_cut.get())
        except:s=10
        g=to_gray(self.img_current)
        self.show_output(gaussian_low_pass(g,s))

    def act_gaussian_hpf(self):
        if self.img_current is None:return
        try:s=float(self.entry_cut.get())
        except:s=10
        g=to_gray(self.img_current)
        self.show_output(gaussian_high_pass(g,s))

# ---------------------------------------------------------
# Run App
# ---------------------------------------------------------
if __name__ == "__main__":
    app = ImageToolApp()
    app.mainloop()
