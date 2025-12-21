"""
Segmented Creator: Video Processing Application GUI

This module provides a comprehensive graphical user interface for the
Segmented Creator pipeline, which facilitates the creation of semantic
segmentation datasets from videos using SAM2 (Segment Anything Model 2).

Overview:
---------
The Segmented Creator tool provides an 8-step pipeline for creating
high-quality segmentation datasets:

Step 1 - Frame Extraction:
    Extracts and aligns frames from input video, correcting camera
    vibrations for consistent segmentation results.

Step 2 - Interactive Initial Segmentation:
    Allows users to interactively segment objects in the first frame
    using SAM2, creating initial prompts for automated propagation.

Step 3 - Automatic Mask Propagation:
    Uses the prompts from Step 2 to automatically segment objects
    throughout the video sequence using SAM2.

Step 4 - Discovering New Objects:
    Performs YOLO-based object detection with SAHI enhancement and
    DeepSort tracking to identify new objects appearing mid-video.

Step 5 - Interactive Mask Refinement:
    Provides interactive refinement of detected objects, allowing
    users to improve segmentation quality for important objects.

Step 6 - Enhanced Mask Propagation:
    Uses refined masks from Step 5 to provide enhanced segmentation
    propagation both forward and backward through the video.

Step 7 - Color-Based Semantic Segmentation:
    Creates color-coded semantic segmentation maps where each class
    has a distinct color for visual clarity.

Step 8 - Final Dataset Creation:
    Creates the final dataset by combining tracked objects with a
    manually prepared background frame (`static.png`) from the
    original stabilized video.

GUI Features:
-------------
- Process selection panel for choosing which step to execute
- Configuration management with automatic saving/loading
- Real-time status monitoring with terminal output capture
- Device information display (GPU/CPU detection)
- Parameter configuration for each processing step
- Thread-safe execution to prevent GUI freezing
- Configuration persistence across sessions

Usage:
------
1. Select a video file to process
2. Choose the desired step from the process selection panel
3. Configure required parameters (SAM2 checkpoint, model config, etc.)
4. Click "Execute Process" to run the selected step
5. Monitor progress through the status line
6. Repeat steps as needed to complete the pipeline

Dependencies:
-------------
- SAM2 (Segment Anything Model 2) for segmentation
- YOLO for object detection
- SAHI (Sliced Aided Hyper Inference) for enhanced detection
- DeepSort for object tracking
- Tkinter for GUI
- PyTorch for deep learning operations
- OpenCV for computer vision operations

Configuration:
--------------
The application automatically saves configuration to 'config.yaml',
including:
- Video path
- SAM2 checkpoint path
- Model configuration path
- Processing parameters (factor, num_images, num_objects, etc.)

This configuration is automatically loaded on subsequent runs.

Detailed Step Information:
-------------------------

Step 1 - Frame Extraction:
Script: first_step.py
Purpose: Prepares the base material for the entire process. Extracts each
frame from the video, aligns it with the previous frame to correct for
small camera vibrations, and saves it as an image.
Inputs: The video file selected in the GUI (--root)
Outputs: imgsA/ folder (individual frames), video_alineado.mp4 file

Step 2 - Initial Interactive Segmentation:
Script: second_step.py
Purpose: Teaches the model which objects you are interested in on the
first frame. This initial information is crucial for the model to track
these objects later.
How it works: Shows first frame (resized according to Factor), allows
clicking on objects, prompts for positive/negative clicks, SAM2 shows
masks in real-time, allows refinement with additional points.
Controls: Mouse click (adds points), 'a' key (accept object, assign class),
'ESC' key (finish and assign last class)
Inputs: First frame from imgsA/ folder, user interaction
Outputs: mask_prompts.csv file (reference points, labels, classes)

Step 3 - Mask Propagation:
Script: third_step.py
Purpose: Uses mask_prompts.csv to process video in batches and propagate
initial masks through frames, tracking objects.
How it works: Loads prompts, creates initial masks for first batch,
uses last frame mask as starting point for subsequent batches,
saves each generated mask as individual image.
Inputs: mask_prompts.csv, images from imgsA/ folder
Outputs: masks/ folder (individual mask files), segmentation/ folder
(grouped masks by frame)

Step 4 - Discovering New Objects:
Script: fourth_step.py
Purpose: Designed to find objects that appear mid-video and were not originally
segmented in Step 2. It uses YOLO for detection and DeepSort for tracking to
identify new objects of interest.
How it works: Processes video_alineado.mp4 frame by frame, applies
inverse masks to hide already segmented areas, uses SAHI to improve small object
detection, uses DeepSort to assign unique tracking IDs to newly detected objects.
Inputs: video_alineado.mp4, existing masks from segmentation/ folder
Outputs: track_dic.csv file (discovered objects, discovery frame, bounding boxes)

Step 5 - Interactive Mask Refinement:
Script: fifth_step.py
Purpose: Reviews detected objects from Step 4 and creates high-quality
masks using SAM2 interactively for important objects that may have been
missed in Step 3.
How it works: Reads track_dic.csv, shows cropped regions for each object,
allows positive/negative clicks for precise masking.
Controls: Mouse click (adds points), 'a' key (accept and assign class),
'0' key (skip object), 'ESC' key (finish process)
Inputs: track_dic.csv, images from imgsA/ folder
Outputs: traked/ folder (new interactive masks), mask_list.csv file
(list of generated masks)

Step 6 - Enhanced Mask Propagation:
Script: sixth_step.py
Purpose: Similar to Step 3 but uses refined masks from Step 5 to
propagate them forward and backward through the video.
How it works: For each mask in mask_list.csv, uses as starting point,
processes video in batches forward from mask frame, then backward from
same frame to complete preceding frames.
Inputs: mask_list.csv, images from imgsA/ folder
Outputs: masks/ folder (updated with refined object masks)

Step 7 - Color-Based Semantic Segmentation:
Script: seventh_step.py
Purpose: Unifies all generated masks into single images per frame where
each object class has unique color, creating semantic segmentation.
How it works: Reads class colors from class_dict.csv, combines all masks
from masks/ folder for each frame, paints each mask with its class color.
Inputs: All masks in masks/ folder, class_dict.csv for colors
Outputs: semantic/ folder (colored semantic segmentation images)

Step 8 - Final Dataset Creation:
Script: eighth_step.py
Purpose: Creates the final dataset by combining tracked objects with a
clean background base (`static.png`).
How it works: Uses `static.png` as the background. This frame is typically
a manually edited version of the first stabilized frame containing only
the background. Tracked objects are then overlaid onto this base.
Inputs: Images from semantic/ folder, static background image (static.png)
Outputs: dataset/ folder (final ready-to-use dataset)
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import os
import sys
import subprocess
import torch
import webbrowser
from PIL import Image, ImageTk


class Tooltip:
    """Simple tooltip class for Tkinter widgets."""

    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)
        self.widget.bind("<Button-1>", self.hide_tooltip)
        self.widget.bind("<FocusOut>", self.hide_tooltip)

    def show_tooltip(self, event=None):
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25

        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")

        label = ttk.Label(self.tooltip, text=self.text, background="#333333",
                          foreground="white", relief="solid", borderwidth=1,
                          font=("Arial", 9))
        label.pack()

    def hide_tooltip(self, event=None):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None

from vivadatacreator.runtime_utils import (
    OPTIMIZED_DEFAULTS,
    load_runtime_config,
    save_runtime_config,
)
from vivadatacreator.download_checkpoints import ensure_checkpoints
from vivadatacreator.sam2_resources import (
    CHECKPOINTS_DIR,
    DEFAULT_MODEL_KEY,
    build_model_map,
)

class TerminalCapture:
    """
    Captures stdout and stderr to update a status Label widget
    with the last output line.
    """
    def __init__(self, status_label_widget):
        self.status_label_widget = status_label_widget
        self.last_line = "Terminal ready"
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.capture_stdout()

    def capture_stdout(self):
        sys.stdout = self
        sys.stderr = self

    def write(self, text):
        self.original_stdout.write(text)
        cleaned_text = text.strip()
        if cleaned_text:
            self.last_line = cleaned_text
            # Update the status label with the last line
            self.status_label_widget.config(text=self.last_line)
            self.status_label_widget.update_idletasks()

    def flush(self):
        self.original_stdout.flush()

    def get_last_line(self):
        return self.last_line

class VideoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ViVa-DataCreator")
        # --- CHANGE: Reduced window size for a more compact layout ---
        self.root.geometry("1024x620")
        self.root.minsize(1024, 620)

        # Configure fonts
        self.FONT = ("Arial", 10)
        self.FONT_BOLD = ("Arial", 11, 'bold')

        self.setup_theme()

        self.device_info = self.get_device_info()

        # --- Variables ---
        self.video_path = None
        self.sam2_chkpt_path = None
        self.model_cfg_path = None
        self.processing_thread = None

        self.config_data = self.load_config()
        self.video_path = self.config_data.get("root") or self.video_path
        self.sam2_chkpt_path = self.config_data.get("sam2_chkpt")
        self.model_cfg_path = self.config_data.get("model_cfg")

        # Ensure checkpoints are available for SAM2
        if not CHECKPOINTS_DIR.exists():
            print("SAM2 checkpoints not found. Downloading...")
            ensure_checkpoints(CHECKPOINTS_DIR)

        self.create_widgets()
        self.on_option_selected()  # Set initial step description
    def setup_theme(self):
        """Sets up the Forest-dark theme from the .tcl file"""
        try:
            tcl_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'GUI', 'forest-dark.tcl')
            if os.path.exists(tcl_file):
                self.root.tk.call('source', tcl_file)
                style = ttk.Style()
                style.theme_use('forest-dark')
                self._configure_fonts(style)
                print("Forest-dark theme loaded successfully")
            else:
                print("forest-dark.tcl not found. Using default theme.")
        except Exception as e:
            print(f"Error loading theme: {e}")

    def _configure_fonts(self, style):
        """Configure fonts for cross-platform compatibility."""
        style.configure('TLabel', font=self.FONT)
        style.configure('TButton', font=self.FONT)
        style.configure('TEntry', font=self.FONT)
        style.configure('TCheckbutton', font=self.FONT)
        style.configure('TRadiobutton', font=self.FONT)
        style.configure('TScale', font=self.FONT)
        style.configure('TSpinbox', font=self.FONT)
        style.configure('TCombobox', font=self.FONT)
        style.configure('TLabelframe.Label', font=self.FONT_BOLD)

    def load_config(self):
        """Loads configuration from config.yaml if it exists"""
        config = load_runtime_config()
        print("Optimized defaults loaded" if config else "ℹ️ Using built-in defaults")
        return config

    def save_config(self):
        """Saves current configuration to config.yaml"""
        try:
            save_runtime_config(self.config_data)
            print("Configuration saved to config.yaml")
        except Exception as e:
            print(f"Error saving configuration: {e}")

    def get_device_info(self):
        """Gets device information (GPU/CPU)"""
        try:
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3 # GB
                return f"GPU: {gpu_name} ({gpu_memory:.1f} GB)"
            else:
                return "CPU: No GPU available"
        except ImportError:
            return "CPU: Torch is not available"

    def create_widgets(self):
        # --- CHANGE: Main layout restructured for compactness ---
        
        # 1. Status line at the very bottom
        status_line_frame = ttk.Frame(self.root)
        status_line_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(5, 10))

        # 2. Main container for the two-column layout
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        main_container.rowconfigure(0, weight=1)
        main_container.columnconfigure(0, minsize=220) # Left column for process selection
        main_container.columnconfigure(1, weight=1)    # Right column for everything else

        # 3. Create and place the two main panes
        # Left Panel Container
        left_panel = ttk.Frame(main_container)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        # Logo
        try:
            logo_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'docs', 'assets', 'logo.png')
            if os.path.exists(logo_path):
                # Open the image using PIL
                pil_image = Image.open(logo_path)
                
                # Calculate new height to maintain aspect ratio with width 200
                target_width = 200
                width_percent = (target_width / float(pil_image.size[0]))
                target_height = int((float(pil_image.size[1]) * float(width_percent)))
                
                # Resize image
                resized_image = pil_image.resize((target_width, target_height), Image.Resampling.LANCZOS)
                
                # Convert to PhotoImage
                self.logo_img = ImageTk.PhotoImage(resized_image)
                
                logo_label = ttk.Label(left_panel, image=self.logo_img)
                logo_label.pack(anchor='center', pady=(0, 10))
        except Exception as e:
            print(f"Error loading logo: {e}")

        process_pane = ttk.Labelframe(left_panel, text="Process Selection")
        process_pane.pack(fill=tk.X, pady=(0, 10))

        # Documentation Link
        doc_link = ttk.Label(left_panel, text="Documentation", foreground="#4a90e2", cursor="hand2", font=("Arial", 10, "underline"))
        doc_link.pack(anchor='center', pady=(0, 10))
        doc_link.bind("<Button-1>", lambda e: self.open_documentation("https://viva-safeland.github.io/viva_datacreator"))

        controls_pane = ttk.Frame(main_container)
        controls_pane.grid(row=0, column=1, sticky="nsew")

        # 4. Populate the panes
        self.create_process_selection_panel(process_pane)
        self.create_controls_panel(controls_pane)
        
        # 5. Populate the status line
        status_label_title = ttk.Label(status_line_frame, text="LAST MESSAGE:", font=self.FONT_BOLD)
        status_label_title.pack(side=tk.LEFT, padx=(5, 5))

        self.status_line_label = ttk.Label(
            status_line_frame,
            text="Terminal ready",
            font=self.FONT,
            anchor='w',
            wraplength=1000
        )
        self.status_line_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        self.terminal_capture = TerminalCapture(self.status_line_label)

    def open_documentation(self, url):
        """Opens the documentation URL suppressing browser output on Linux."""
        if sys.platform.startswith('linux'):
            try:
                subprocess.Popen(['xdg-open', url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as e:
                print(f"Error opening documentation with xdg-open: {e}")
                # Fallback to webbrowser if xdg-open fails
                webbrowser.open(url)
        else:
            webbrowser.open(url)


    def create_process_selection_panel(self, parent):
        """Creates the left panel with only the process selection radio buttons."""
        options = ["first_step", "second_step", "third_step", "fourth_step", "fifth_step", "sixth_step", "seventh_step", "eighth_step"]
        self.selected_option = tk.StringVar(value=options[0])

        for option in options:
            rb = ttk.Radiobutton(parent, text=f"{option.replace('_', ' ').title()}",
                                 variable=self.selected_option,
                                 value=option, command=self.on_option_selected)
            rb.pack(anchor='w', pady=5, padx=10)

    def create_controls_panel(self, parent):
        """Creates the right panel with status, configuration, and actions."""
        # --- Grid layout for the right panel ---
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1) # Allow config frame to expand

        # --- 1. Status and Info Section ---
        status_labelframe = ttk.Labelframe(parent, text="Status & Info")
        status_labelframe.grid(row=0, column=0, sticky="new", pady=(0, 10))
        status_labelframe.columnconfigure(0, weight=1)

        device_label = ttk.Label(status_labelframe, text=self.device_info, font=self.FONT_BOLD)
        device_label.pack(fill=tk.X, padx=10, pady=5)
        print(f"Device information: {self.device_info}")

        self.status_labels = {}
        info_labels = ["Status: Idle", "Selected Process: First Step"]
        for i, text in enumerate(info_labels):
            label = ttk.Label(status_labelframe, text=text, anchor='w')
            label.pack(fill=tk.X, padx=10, pady=2)
            self.status_labels[f"label_{i}"] = label

        # --- 2. Configuration Section ---
        config_frame = ttk.Frame(parent)
        config_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        config_frame.rowconfigure(1, weight=1)
        config_frame.columnconfigure(0, weight=1)

        # First sub-frame: File & Model
        file_model_frame = ttk.Labelframe(config_frame, text="File & Model")
        file_model_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        file_model_frame.columnconfigure(1, weight=1)

        # File-based Parameters
        file_params = [
            ("Video:", "video_path", "video_search_var", self.browse_video, "root", "")
        ]

        for i, (label_text, path_attr, var_attr, cmd, config_key, default) in enumerate(file_params):
            ttk.Label(file_model_frame, text=label_text).grid(row=i, column=0, sticky='w', pady=4, padx=5)
            entry_frame = ttk.Frame(file_model_frame)
            entry_frame.grid(row=i, column=1, sticky='ew', pady=4, padx=5)

            var = tk.StringVar()
            setattr(self, var_attr, var)
            path_value = self.config_data.get(config_key, default)

            if path_value:
                setattr(self, path_attr, path_value)
                if os.path.exists(path_value):
                    var.set(os.path.basename(path_value))
                else:
                    var.set(path_value)

            entry = ttk.Entry(entry_frame, textvariable=var, state='readonly')
            entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
            browse_btn = ttk.Button(entry_frame, text="...", command=cmd, width=3)
            browse_btn.pack(side=tk.RIGHT, padx=(5, 0))

        # Model Selection Combobox
        ttk.Label(file_model_frame, text="SAM2 Model:").grid(row=len(file_params), column=0, sticky='w', pady=4, padx=5)
        self.model_map = build_model_map()
        missing_configs = [
            f"{name}: {cfg}" for name, (cfg, _) in self.model_map.items() if not os.path.exists(cfg)
        ]
        if missing_configs:
            print("⚠️ Missing SAM2 config files:")
            for entry in missing_configs:
                print(f"   - {entry}")
            print("   Please reinstall viva-datacreator or ensure the SAM2 package is accessible.")
        model_options = list(self.model_map.keys())
        selected_cfg = self.config_data.get("model_cfg")
        selected_chkpt = self.config_data.get("sam2_chkpt")
        default_model = next(
            (
                name
                for name, (cfg, chkpt) in self.model_map.items()
                if cfg == selected_cfg and chkpt == selected_chkpt
            ),
            DEFAULT_MODEL_KEY,
        )
        self.model_var = tk.StringVar()
        self.model_var.set(default_model)
        self.model_cfg_path, self.sam2_chkpt_path = self.model_map[default_model]
        self.config_data["model_cfg"] = self.model_cfg_path
        self.config_data["sam2_chkpt"] = self.sam2_chkpt_path
        model_combo = ttk.Combobox(file_model_frame, textvariable=self.model_var, values=model_options, state='readonly')
        model_combo.grid(row=len(file_params), column=1, sticky='ew', pady=4, padx=5)
        model_combo.bind('<<ComboboxSelected>>', self.on_model_selected)

        # Second sub-frame: Parameters and Description
        params_desc_frame = ttk.Frame(config_frame)
        params_desc_frame.grid(row=1, column=0, sticky="nsew")
        params_desc_frame.columnconfigure(0, weight=1)
        params_desc_frame.columnconfigure(1, weight=1)

        # Left frame for parameters
        params_frame = ttk.Labelframe(params_desc_frame, text="Parameters")
        params_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=5)
        params_frame.columnconfigure(0, minsize=100)
        params_frame.columnconfigure(1, minsize=150)

        # Numeric Parameters
        numeric_params = [
            ("Factor:", "fac_var", "fac", str(OPTIMIZED_DEFAULTS["fac"]), "Scaling factor for resizing frames.\nUsed in Step 2 for interactive segmentation."),
            ("Num Images:", "n_imgs_var", "n_imgs", str(OPTIMIZED_DEFAULTS["n_imgs"]), "Number of images to process per batch.\nUsed in Step 3 for mask propagation."),
            ("Num Objects:", "n_obj_var", "n_obj", str(OPTIMIZED_DEFAULTS["n_obj"]), "Number of objects to process per batch.\nUsed in Step 3 for mask propagation."),
            ("SAHI Image Size:", "img_size_sahi_var", "img_size_sahi", str(OPTIMIZED_DEFAULTS["img_size_sahi"]), "Image slice size for SAHI object detection.\nUsed in Step 4 for detection enhancement."),
            ("SAHI Overlap:", "overlap_sahi_var", "overlap_sahi", str(OPTIMIZED_DEFAULTS["overlap_sahi"]), "Overlap ratio between SAHI slices.\nUsed in Step 4 for detection enhancement."),
        ]

        for i, (label, var_name, key, default, tooltip_text) in enumerate(numeric_params):
            ttk.Label(params_frame, text=label).grid(row=i, column=0, sticky='w', pady=4, padx=5)
            config_value = self.config_data.get(key, default)
            var = tk.StringVar(value=str(config_value))
            setattr(self, var_name, var)
            entry = ttk.Entry(params_frame, textvariable=var, width=15)
            entry.grid(row=i, column=1, sticky='w', pady=4, padx=5)
            Tooltip(entry, tooltip_text)

        # Right frame for descriptions
        desc_frame = ttk.Labelframe(params_desc_frame, text="Description")
        desc_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=5)

        # Step Description Label
        self.step_desc_label = ttk.Label(desc_frame, text="", wraplength=300, anchor='center', justify='center', font=self.FONT_BOLD)
        self.step_desc_label.pack(fill=tk.BOTH, expand=True)

        # --- 3. Actions Section ---
        actions_frame = ttk.Labelframe(parent, text="Actions")
        actions_frame.grid(row=2, column=0, sticky="sew")
        actions_frame.columnconfigure(0, weight=1)
        actions_frame.columnconfigure(1, weight=1)
        
        self.execute_btn = ttk.Button(actions_frame, text="Execute Process", command=self.execute_process, state=tk.DISABLED)
        self.execute_btn.grid(row=0, column=0, sticky='ew', padx=5, pady=5)
        
        clear_btn = ttk.Button(actions_frame, text="Clear Status Line", command=self.clear_status_line)
        clear_btn.grid(row=0, column=1, sticky='ew', padx=5, pady=5)
        
        if self.video_path:
            self.execute_btn.configure(state=tk.NORMAL)

    def clear_status_line(self):
        """Clears the status line"""
        self.status_line_label.config(text="")
        print("Status line cleared\n")
    
    def browse_video(self):
        initialdir = os.path.dirname(self.video_path) if self.video_path else None
        file_path = filedialog.askopenfilename(title="Select video", filetypes=[("Video files", "*.mp4 *.MP4 *.avi *.AVI *.mov *.MOV *.mkv *.MKV *.wmv *.WMV *.flv *.FLV *.webm *.WEBM"), ("All files", "*.*")], initialdir=initialdir)
        if file_path:
            if os.path.isfile(file_path):
                self.video_path = file_path
                self.config_data["root"] = file_path
                self.save_config()
                self.video_search_var.set(os.path.basename(file_path))
                self.update_status_labels()
                print(f"Video selected: {os.path.basename(file_path)}\n")
                if self.video_path:
                    self.execute_btn.configure(state=tk.NORMAL)
            else:
                messagebox.showerror("Error", "Selected file does not exist or is not a valid file.")
                print("Error: Selected file is invalid.\n")


    def on_model_selected(self, event):
        selected = self.model_var.get()
        self.model_cfg_path, self.sam2_chkpt_path = self.model_map[selected]
        self.config_data["model_cfg"] = self.model_cfg_path
        self.config_data["sam2_chkpt"] = self.sam2_chkpt_path
        self.save_config()
        print(f"Model selected: {selected}\n")
            
    def on_option_selected(self):
        option = self.selected_option.get()
        process_name = option.replace('_', ' ').title()
        self.status_labels["label_1"].configure(text=f"Selected Process: {process_name}")
        if self.video_path:
            self.execute_btn.configure(state=tk.NORMAL)

        # Update step description
        descriptions = {
            "first_step": "Prepares the base material for the entire process. Extracts each frame from the video, aligns it with the previous frame to correct for small camera vibrations, and saves it as an image. Outputs: imgsA/ folder with individual frames, video_alineado.mp4 file, static.png (first frame for background use).",
            "second_step": "Teaches the model which objects you are interested in on the first frame. Shows the first frame (resized according to Factor), allows clicking on objects with positive/negative points, SAM2 shows masks in real-time. Controls: Mouse click (adds points), 'a' key (accept object, assign class), 'ESC' key (finish). Outputs: mask_prompts.csv with reference points, labels, and classes.",
            "third_step": "Uses mask_prompts.csv to process video in batches and propagate initial masks through frames, tracking objects. Loads prompts, creates initial masks for first batch, uses last frame mask as starting point for subsequent batches. Outputs: masks/ folder with individual mask files, segmentation/ folder with grouped masks by frame.",
            "fourth_step": "Designed to find objects that appear mid-video and were not originally segmented in Step 2. Processes video_alineado.mp4 frame by frame, applies inverse masks to hide already segmented areas so the model focuses only on parts of the video that don't have a mask yet. Uses SAHI for small object detection and DeepSort for tracking newly detected objects. Outputs: track_dic.csv with newly detected objects, discovery frame, and bounding boxes.",
            "fifth_step": "Reviews detected objects from Step 4 and creates high-quality masks using SAM2 interactively for important objects that may have been missed in Step 3. Reads track_dic.csv, shows cropped regions for each object, allows positive/negative clicks for precise masking. Controls: Mouse click (adds points), 'a' key (accept and assign class), '0' key (skip object), 'ESC' key (finish). Outputs: traked/ folder with new masks, mask_list.csv with list of generated masks.",
            "sixth_step": "Similar to Step 3 but uses refined masks from Step 5 to propagate them forward and backward through the video. For each mask in mask_list.csv, uses as starting point, processes video in batches forward from mask frame, then backward from same frame. Outputs: masks/ folder updated with refined object masks.",
            "seventh_step": "Unifies all generated masks into single images per frame where each object class has unique color, creating semantic segmentation. Reads class colors from class_dict.csv, combines all masks from masks/ folder for each frame, paints each mask with its class color. Outputs: semantic/ folder with colored semantic segmentation images.",
            "eighth_step": "Creates the final dataset by combining tracked objects with a clean background base. This step uses 'static.png' as the background, which is typically a manually edited version of the first stabilized frame containing only the background. The segmented objects are then overlaid onto this backdrop. Outputs: dataset/ folder with final ready-to-use dataset."
        }
        self.step_desc_label.config(text=descriptions.get(option, ""))

    def update_status_labels(self):
        status_text = "Idle"
        option_text = f"Selected Process: {self.selected_option.get().replace('_', ' ').title()}"
        self.status_labels["label_0"].configure(text=f"Status: {status_text}")
        self.status_labels["label_1"].configure(text=option_text)
    
    def execute_process(self):
        if not self.video_path:
            messagebox.showwarning("Warning", "Please select a video first.")
            return
        
        if self.processing_thread and self.processing_thread.is_alive():
            messagebox.showwarning("Warning", "A process is already running.")
            return
        
        self.processing_thread = threading.Thread(target=self.run_selected_process, daemon=True)
        self.processing_thread.start()
        
    def run_selected_process(self):
        """Executes the selected process with all parameters"""
        try:
            step = self.selected_option.get()
            print(f"Starting process: {step}\n")
            self.status_labels["label_0"].configure(text="Status: Processing...")
            
            cmd = [
                "uv", "run", "python", "-m", f"vivadatacreator.{step}",
                "--root", self.video_path,
                "--fac", self.fac_var.get(),
                "--n-imgs", self.n_imgs_var.get(),
                "--n-obj", self.n_obj_var.get(),
                "--img-size-sahi", self.img_size_sahi_var.get(),
                "--overlap-sahi", self.overlap_sahi_var.get(),
            ]
            
            if self.sam2_chkpt_path:
                cmd.extend(["--sam2-chkpt", self.sam2_chkpt_path])
            if self.model_cfg_path:
                cmd.extend(["--model-cfg", self.model_cfg_path])
            
            print(f"Executing: {' '.join(cmd)}\n")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
                encoding='utf-8',
                errors='replace'
            )

            output_lines = []
            for line in process.stdout:
                output_lines.append(line.strip())
                print(line.strip())

            process.wait()

            if process.returncode == 0:
                final_msg = "Process completed successfully!"
                print(f"{final_msg}\n")
                self.status_labels["label_0"].configure(text="Status: Completed")
            else:
                # Show last 10 lines of output in error
                error_details = "\n".join(output_lines[-10:]) if output_lines else "No output captured"
                final_msg = f"Process failed with code: {process.returncode}\nError details:\n{error_details}"
                print(f"{final_msg}\n")
                self.status_labels["label_0"].configure(text="Status: Error")
                messagebox.showerror("Error", f"The process failed with code {process.returncode}\n\nError details:\n{error_details}")
                # Crash the program as requested
                sys.exit(1)
            
            self.status_line_label.config(text=final_msg)

        except FileNotFoundError:
            error_msg = "Error: 'uv' not found. Make sure 'uv' is installed and in your PATH."
            print(error_msg)
            self.status_labels["label_0"].configure(text="Status: Error - uv not found")
            self.status_line_label.config(text=error_msg)
            messagebox.showerror("Error", error_msg)
        except Exception as e:
            error_msg = f"Error executing process: {str(e)}"
            print(error_msg)
            self.status_labels["label_0"].configure(text="Status: Error")
            self.status_line_label.config(text=error_msg)
            messagebox.showerror("Error", error_msg)

    def on_closing(self):
        # Update config_data with current values before saving
        if self.video_path:
            self.config_data["root"] = self.video_path
        if self.sam2_chkpt_path:
            self.config_data["sam2_chkpt"] = self.sam2_chkpt_path
        if self.model_cfg_path:
            self.config_data["model_cfg"] = self.model_cfg_path

        # Update numeric parameters
        try:
            self.config_data["fac"] = int(self.fac_var.get())
        except ValueError:
            pass
        try:
            self.config_data["n_imgs"] = int(self.n_imgs_var.get())
        except ValueError:
            pass
        try:
            self.config_data["n_obj"] = int(self.n_obj_var.get())
        except ValueError:
            pass
        try:
            self.config_data["img_size_sahi"] = int(self.img_size_sahi_var.get())
        except ValueError:
            pass
        try:
            self.config_data["overlap_sahi"] = float(self.overlap_sahi_var.get())
        except ValueError:
            pass

        self.save_config()
        sys.stdout = self.terminal_capture.original_stdout
        sys.stderr = self.terminal_capture.original_stderr
        self.root.destroy()

def main():
    root = tk.Tk()
    app = VideoApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
