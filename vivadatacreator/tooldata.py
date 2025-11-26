import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter import simpledialog, Tk, filedialog
import matplotlib.pyplot as plt
import os
import csv
import shutil
import ast
#import re
import pandas as pd
from tqdm import tqdm
import threading
from ultralytics import YOLO

#from sam2.build_sam import build_sam2_video_predictor
#import torch
from PIL import Image
from pathlib import Path
from IPython.display import clear_output

from sahi.predict import get_sliced_prediction

# Lock for thread-safe model inference
inference_lock = threading.Lock()

##### Creating folders #####
def folder_creation(ruta):

    if ruta is None:
        print("No path provided. Please select a video file..")
        root, video_path = get_folder_path()
    else:
        print(f"Provided path: {ruta}")
        path_obj = Path(ruta)
        root = str(path_obj.parent)
        video_path = str(path_obj.resolve())

    ### Aligned Video ###
    video_dir = os.path.join(root, 'video_alineado.mp4')

    ### Auxiliary Frames ###
    aux_folder = os.path.join(root, 'aux_frame')
    os.makedirs(aux_folder, exist_ok=True)

    ### Video to Aligned Frames ###
    imgs_folder_A = os.path.join(root, 'imgsA')
    os.makedirs(imgs_folder_A, exist_ok=True)

    ### Save Masks ###
    mask_folder = os.path.join(root, 'masks')
    os.makedirs(mask_folder, exist_ok=True)

    ### Grouped Masks ###
    frames_folder = os.path.join(root, 'segmentation')
    os.makedirs(frames_folder, exist_ok=True)

    ##### New Directories #####

    ### Frame Auxiliar ###
    frame_aux = os.path.join(root, 'frame_aux')
    os.makedirs(frame_aux, exist_ok=True)

    ### Recorte Auxiliar ###
    recorte_folder = os.path.join(root, 'recorte')
    os.makedirs(recorte_folder, exist_ok=True)

    ### Objects Detected ###
    traked_folder = os.path.join(root, 'traked')
    os.makedirs(traked_folder, exist_ok=True)

    ### Semantic Segmentation ###
    semantic_folder = os.path.join(root, 'semantic')
    os.makedirs(semantic_folder, exist_ok=True)

    ### final dataset ###
    dataset_folder = os.path.join(root, 'dataset')
    os.makedirs(dataset_folder, exist_ok=True)

    return {
        "root": root,                       # root folder
        "video_path": video_path,           # original video
        "video_dir": video_dir,             # aligned video
        "aux_folder": aux_folder,
        "imgs_folder_A": imgs_folder_A,
        "mask_folder": mask_folder,
        "frames_folder": frames_folder,
        "frame_aux": frame_aux,
        "recorte_folder": recorte_folder,
        "traked_folder": traked_folder,
        "semantic_folder": semantic_folder,
        "dataset_folder": dataset_folder
    }

### Select root folder ###
def get_folder_path():
    root = Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(title="Select the video to process")

    if not file_path:  # If the user cancels
        return None, None
    
    path_obj = Path(file_path)
    return str(path_obj.parent), str(path_obj.resolve()) # (folder, file_path)

##### Align frames #####
def alinear_imagen(img1, img2):
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Detect features using ORB
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

    # Feature matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract keypoints to compute the transformation matrix
    pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute the homography matrix
    matrix, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 10.0)

    # Apply the transformation
    aligned_img = cv2.warpPerspective(img2, matrix, (img1.shape[1], img1.shape[0]))

    return aligned_img

### Create aligned video ###
def crear_video(output_dir, output_video="video_final.mp4", fps=30, codec='mp4v'):
    try:
        # Get list of ordered frames
        frames = sorted([f for f in os.listdir(output_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        if not frames:
            print("Error: No frames found in the directory.")
            return False

        # Get dimensions of the first frame
        example_frame = cv2.imread(os.path.join(output_dir, frames[0]))
        if example_frame is None:
            print("Error: Could not read example frame.")
            return False
            
        height, width, layers = example_frame.shape

        # Configure VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*codec)
        video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

        # Add each frame to the video
        for frame_file in tqdm(frames, desc="Creating video", colour=None):
            frame_path = os.path.join(output_dir, frame_file)
            frame = cv2.imread(frame_path)
            if frame is not None:
                video.write(frame)
            else:
                print(f"Warning: Could not read frame {frame_file}")

        # Release VideoWriter
        video.release()
        
        print(f"¡The video {output_video} has been created successfully!")
        return True
        
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return False

### Mostrar mascaras ###
def show_mask(image_or_ax, mask, obj_id=None, random_color=False, is_video=False):
    """
    Parameters:
    - image_or_ax: Either the input image (for non-video) or matplotlib axis (for video)
    - mask: The mask to display
    - obj_id: Object ID for color selection
    - random_color: Whether to use random color
    - is_video: Flag to determine if displaying on axis (True) or returning image (False)
    """
    if random_color:
        if is_video:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.concatenate([np.random.random(3) * 255, np.array([0.6 * 255])], axis=0).astype(np.uint8)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        if is_video:
            color = np.array([*cmap(cmap_idx)[:3], 0.6])

            h, w = mask.shape[-2:]
            mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        else:
            color = np.array([*cmap(cmap_idx)[:3]]) * 255
            color = color.astype(np.uint8)

            h, w = mask.shape[:2]
            mask_image = (mask.reshape(h, w, 1) * color.reshape(1, 1, -1)).astype(np.uint8)

        return mask_image

### Create mask in SAM2 predictor ###
def mask_creation(puntos_interes, input_label, ann_obj_id, predictor, inference_state, ann_frame_idx=0, frame_shape=None):
    # for labels, `1` means positive click and `0` means negative click
    points = np.array([puntos_interes], dtype=np.float32)
    labels = np.array(input_label, np.int32)
    
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )
    
    # Detect scale factor between displayed frame and SAM2 internal processing
    scale_h, scale_w = 1.0, 1.0
    if frame_shape is not None:
        mask_h, mask_w = out_mask_logits[0].shape[1:3]
        frame_h, frame_w = frame_shape[:2]
        scale_h = mask_h / frame_h
        scale_w = mask_w / frame_w
        print(f"✓ SAM2 scale factor detected: {scale_w:.2f}x (Frame: {frame_w}x{frame_h}, SAM2: {mask_w}x{mask_h})")

    return out_mask_logits, out_obj_ids, (scale_h, scale_w)

### Function to handle mouse click events
def on_click(event, x, y, flags, param):
    frame = param["frame"]
    estado = param["estado"]
    if event == cv2.EVENT_LBUTTONDOWN:
        root = tk.Tk()
        root.withdraw()
        etiqueta = simpledialog.askstring("Label", "Save with label 1 or 0?")
        root.destroy()

        if etiqueta in ['0', '1']:
            label = int(etiqueta)
            estado["input_label"] = np.append(estado["input_label"], label)
            
            # Store click coordinates in frame space
            estado["puntos_interes"].append([x, y])
            
            color = (0, 255, 0) if label == 1 else (0, 0, 255)
            
            # Draw point on the displayed frame
            cv2.circle(frame, (x, y), 2, color, -1)
            print(f"Point added: [{x}, {y}] with label: {etiqueta} and object ID: {estado['ann_obj_id']}")

            # Get or calculate SAM2 scale factor
            if "sam2_scale" not in estado:
                # First click: detect scale factor
                out_mask_logits, out_obj_ids, sam2_scale = mask_creation(
                    estado["puntos_interes"], estado["input_label"], estado["ann_obj_id"],
                    estado["predictor"], estado["inference_state"], frame_shape=frame.shape
                )
                estado["sam2_scale"] = sam2_scale
                
                # If scale factor is not 1.0, recalculate with scaled coordinates
                if sam2_scale[0] != 1.0 or sam2_scale[1] != 1.0:
                    scaled_points = [[int(x * sam2_scale[1]), int(y * sam2_scale[0])] for x, y in estado["puntos_interes"]]
                    out_mask_logits, out_obj_ids, _ = mask_creation(
                        scaled_points, estado["input_label"], estado["ann_obj_id"],
                        estado["predictor"], estado["inference_state"]
                    )
            else:
                # Subsequent clicks: use saved scale factor to adjust coordinates
                sam2_scale = estado["sam2_scale"]
                scaled_points = [[int(x * sam2_scale[1]), int(y * sam2_scale[0])] for x, y in estado["puntos_interes"]]
                
                out_mask_logits, out_obj_ids, _ = mask_creation(
                    scaled_points, estado["input_label"], estado["ann_obj_id"],
                    estado["predictor"], estado["inference_state"]
                )
            
            # Start with a copy of the frame that has all the points drawn
            overlay = frame.copy()
            
            for i, obj_id in enumerate(out_obj_ids):
                out_mask_logits_transposed = np.transpose((out_mask_logits[i] > 0.0).cpu().numpy(), (1, 2, 0))
                mask_image = show_mask(overlay, out_mask_logits_transposed, obj_id=obj_id)
                
                # Ensure mask_image matches frame dimensions
                if mask_image.shape[:2] != overlay.shape[:2]:
                    mask_image = cv2.resize(mask_image, (overlay.shape[1], overlay.shape[0]), interpolation=cv2.INTER_NEAREST)
                
                # Blend mask with the overlay
                overlay = cv2.addWeighted(overlay, 1.0, mask_image, 0.5, 0)
            
            cv2.imshow("Frame", overlay)

        estado["mask"] = out_mask_logits

### Mask spread on video ###
def actualizar_segmentos_video(predictor, inference_state, video_segments, reverse=False):
    if reverse == False:
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
    elif reverse == True:
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=True):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

### Function to split the text string into points, labels, and class
def split_cadena(cadena):
    cadena = cadena[1:-1]
    obj_pt = cadena.split(", array(")
    pt = ast.literal_eval(obj_pt[0])

    obj_lb = obj_pt[1].split("), '")
    lb = ast.literal_eval(obj_lb[0])

    cl = obj_lb[1][:-1]
    
    return pt, lb, cl

### Load first frame dictionary from a CSV file #####
def leer_prompts(archivo_csv):
    prompts = {}
    with open(archivo_csv, 'r') as archivo:
        reader = csv.reader(archivo)
        for fila in reader:
            prompts[fila[0]] = fila[1]
    return prompts

### Add an object to SAM2 from a mask #####
def add_object_mask(mask, ann_obj_id, predictor, inference_state, ann_frame_idx=0):
    _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
        inference_state= inference_state,
        frame_idx= ann_frame_idx,
        obj_id= ann_obj_id,
        mask= mask
    )

    return out_mask_logits, out_obj_ids

### Read a mask from a file and convert it to a boolean array ###
def read_mask(ruta_archivo):
    imagen = Image.open(ruta_archivo)
    datos = np.array(imagen)
    matriz_booleana = datos > 127

    return matriz_booleana

### Save masks to a folder ###
def save_masks(mask_folder, n_imgs, ind, video_segments, cl, only_cls = False):
    lim = min(n_imgs, len(video_segments))
    for out_frame_idx in tqdm(range(lim), desc="Saving masks", colour=None):
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            mask = Image.fromarray((out_mask[0] * 255).astype(np.uint8))
            if only_cls == True:
                mask.save(f"{mask_folder}/outmask_fr{int(out_frame_idx) + ind}_id{out_obj_id}_cl{cl}.png")
            else:
                mask.save(f"{mask_folder}/outmask_fr{int(out_frame_idx) + ind}_id{out_obj_id}_cl{cl[int(out_obj_id)-1]}.png")

#### Process prompts and create masks ###
def procesar_prompts(estado, ini_id, n_obj, n_imgs, fac, mask_folder):
    last_id = int(max(estado["prompts"], key=int)) # Last object in prompts

    for i in range(ini_id, last_id + 1, n_obj):
        estado["predictor"].reset_state(estado["inference_state"])

        lim = min(i + n_obj, last_id + 1)

        for j in range(i, lim):
            pt, lb, cla = split_cadena(estado["prompts"][str(j)])
            estado["cl"].append(cla)
            estado["puntos_interes"] = [[int(valor * fac) for valor in sublista] for sublista in pt]
            estado["input_label"] = lb
            estado["ann_obj_id"] = j

            out_mask_logits, out_obj_ids, _ = mask_creation(
                estado["puntos_interes"],
                estado["input_label"],
                estado["ann_obj_id"],
                estado["predictor"],
                estado["inference_state"]
            )

        actualizar_segmentos_video(estado["predictor"], estado["inference_state"], estado["video_segments"])
        save_masks(mask_folder, n_imgs, 0, estado["video_segments"], estado["cl"])

#### Group masks by object ID and save them ###
def group_masks(mask_folder, frames_folder):
    image_sums = {}
    files = [f for f in os.listdir(mask_folder) if f.startswith('outmask_fr')]

    # Use a single loop to process and save
    for filename in tqdm(files, desc="Processing and saving masks", colour=None):
        group_id = filename.split('_')[1]
        image_path = os.path.join(mask_folder, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if group_id not in image_sums:
            image_sums[group_id] = np.zeros_like(image)

        image_sums[group_id] += image

        # Save the summed image
        output_path = os.path.join(frames_folder, f'{group_id[2:]}.png')
        cv2.imwrite(output_path, image_sums[group_id])

### Detect objects with SAHI ###
def detect_with_sahi(frame, sahi_model, slice_size=256, overlap_ratio=0.2):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    with inference_lock:
        result = get_sliced_prediction(
            image=frame_rgb,
            detection_model=sahi_model,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=overlap_ratio,
            overlap_width_ratio=overlap_ratio
        )

    return result.object_prediction_list

### Convert SAHI predictions to DeepSort format
def convert_sahi_to_deepsort(sahi_predictions):
    detections = []
    
    for pred in sahi_predictions:
        bbox = pred.bbox.to_xywh()
        confidence = pred.score.value
        class_id = pred.category.id
        
        detections.append(([bbox[0], bbox[1], bbox[2], bbox[3]], confidence, class_id))
    
    return detections

### Update object tracking information
def update_tracking_info(tracks, frame_number, track_dict):
    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
            
        ltrb = track.to_tlbr()
        track_id = track.track_id
        x1, y1, x2, y2 = map(int, ltrb)
        
        # Try different methods to get class ID from DeepSort track
        class_id = -1
        if hasattr(track, 'get_det_class'):
            cls = track.get_det_class()
            if cls is not None:
                class_id = cls
        elif hasattr(track, 'det_class'):
            if track.det_class is not None:
                class_id = track.det_class
        elif hasattr(track, 'class_id'):
            if track.class_id is not None:
                class_id = track.class_id
        
        if track_id not in track_dict:
            track_dict[track_id] = {
                "frame_number": frame_number,
                "ltrb": ltrb,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "class_id": class_id
            }
    
    return track_dict

#### Save tracking information to a CSV file ###
def save_tracking_info(track_dict, output_filename):
    # YOLO COCO class names
    class_names = {
        0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
        5: "bus", 7: "truck", -1: "unknown"
    }
    
    with open(output_filename, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["track_id", "frame_number", "x1", "y1", "x2", "y2", "class_id", "class_name"])
        
        for track_id, info in track_dict.items():
            class_id = info.get("class_id", -1)
            class_name = class_names.get(class_id, "unknown")
            writer.writerow([track_id, info["frame_number"],
                            info["x1"], info["y1"], info["x2"], info["y2"],
                            class_id, class_name])
    
    print(f"Information saved in {output_filename}")

### Apply inverse mask to a frame ###
def apply_inverse_mask(frame, mask_path):    
    msk = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if msk is None:
        raise ValueError(f"Could not load mask at {mask_path}")

    inverse_mask = cv2.bitwise_not(msk)
    inverse_mask_rgb = cv2.merge([inverse_mask, inverse_mask, inverse_mask])
    result = cv2.multiply(frame, inverse_mask_rgb, scale=1/255)
    
    return result

### Load detection data from SAHI and DeepSort models
def load_and_prepare_data(csv_path, imgs_folder_A, drop_columns=['track_id']):
    df = pd.read_csv(csv_path)
    df = df.drop(columns=drop_columns)
    
    files = os.listdir(imgs_folder_A)
    image_files = [f for f in sorted(files) if f.endswith(('.png', '.jpg'))]
    
    return df, image_files, files

### Process a specific frame according to the data in the row ###
def process_frame(row, image_files, imgs_folder_A, frame_aux, frames_folder, recorte_folder):
    frame_number = row["frame_number"] -1
    archivo = f'{image_files[frame_number]}'
    
    # Copy and process image
    shutil.copy(os.path.join(imgs_folder_A, archivo), os.path.join(frame_aux, archivo))
    
    # Read coordinates
    x1, y1, x2, y2 = max(0, row["x1"]), max(0, row["y1"]), max(0, row["x2"]), max(0, row["y2"])
    
    # Load and process image
    imagen = cv2.imread(os.path.join(frame_aux, archivo))
    imagen = apply_inverse_mask(imagen, os.path.join(frames_folder, f"{frame_number}.png"))

    # Crop image
    recorte = imagen[y1:y2, x1:x2]
    cv2.imwrite(os.path.join(recorte_folder, archivo), recorte)
    
    return recorte, imagen, (x1, y1, x2, y2), archivo

### Handle user interaction with the cropped image ###
def handle_user_interaction(recorte, estado):
    """Handle user interaction with the image."""
    cv2.imshow("Frame", recorte)
    cv2.setMouseCallback("Frame", on_click, {"frame": recorte, "estado": estado})
    
    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == 97:  # 'a' key Finish adding points
            estado["class_id"] = get_class_from_user()
            estado["prompts"][estado["ann_obj_id"]] = (estado["puntos_interes"], estado["input_label"], estado["class_id"])
            break

        elif key == 48:  # '0' key Skip object
            estado["omitir"] = True
            cv2.destroyAllWindows()
            break

        elif key == 27:  # ESC key Finish process
            estado["terminar"] = True
            break

    cv2.destroyAllWindows()

#### Prompt the user to enter the object class ###
def get_class_from_user():
    """Show a Combobox to select the object class"""
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    opciones = list(pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'class_dict.csv'))['name']) 

    # Create popup window
    popup = tk.Toplevel()
    popup.title("Select class")
    popup.geometry("300x150")  # Window size
    tk.Label(popup, text="Select class:").pack(pady=10)  # Configure Combobox

    # Create Combobox
    combo = ttk.Combobox(popup, values=opciones)
    combo.set(opciones[0])  # Default text
    combo.pack(pady=10)
    
    def on_accept():
        popup.seleccion = combo.get()
        popup.destroy()

    tk.Button(popup, text="Accept", command=on_accept).pack()
    popup.wait_window() # Wait for user interaction

    return getattr(popup, 'seleccion', None)  # Return None if closed without selection

### Create a composite mask from the original mask data and dimensions ###
def create_composite_mask(mask_data, original_shape, bbox):
    """Create a composite mask."""
    x1, y1, x2, y2 = bbox
    new_mask = (mask_data[0][0] > 0.0).cpu().numpy()
    h, w = original_shape[:2]
    com_mask = np.zeros((h, w))
    com_mask[y1:y2, x1:x2] = new_mask
    return com_mask

### Cleanup temporary files ###
def cleanup_temp_files(frame_aux, recorte_folder):
    """Delete temporary files."""
    shutil.rmtree(frame_aux)
    os.makedirs(frame_aux, exist_ok=True)
    shutil.rmtree(recorte_folder)
    os.makedirs(recorte_folder, exist_ok=True)

def load_class_colors(csv_path):
    """Load class colors from a CSV file."""
    df = pd.read_csv(csv_path)
    class_colors = {}
    for _, row in df.iterrows():
        class_colors[row['name']] = (row['b'], row['g'], row['r'])  # OpenCV uses BGR, not RGB
    return class_colors

def group_masks_color(mask_folder, frames_folder):
    class_colors = load_class_colors(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'class_dict.csv'))

    image_sums = {}
    files = sorted([f for f in os.listdir(mask_folder) if f.startswith('outmask_fr')])

    for filename in tqdm(files, desc="Building semantic maps", colour=None):
        parts = filename.split('_')
        group_id = parts[1][2:]
        class_name = parts[3][2:-4]

        mask_path = os.path.join(mask_folder, filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        color = np.array(class_colors.get(class_name, (0, 0, 0)), dtype=np.uint8)
        if group_id not in image_sums:
            image_sums[group_id] = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

        available = np.all(image_sums[group_id] == 0, axis=2)
        apply_mask = (mask > 0) & available
        image_sums[group_id][apply_mask] = color

    for group_id, image in image_sums.items():
        output_path = os.path.join(frames_folder, f'{group_id}.png')
        cv2.imwrite(output_path, image)

def final_dataset(input_folder: str, output_folder: str, static_image_name: str):
    """
    Merge semantic segmentation maps with a static background image wherever
    the segmentation is black (unlabeled).
    """
    os.makedirs(output_folder, exist_ok=True)

    try:
        static_img = Image.open(static_image_name).convert('RGB')
        static_array = np.array(static_img)
        print(f"Static image '{static_image_name}' loaded.")
    except FileNotFoundError:
        print(f"Error: Static image '{static_image_name}' not found.")
        return

    files_to_process = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    print(f"Starting processing of {len(files_to_process)} images...")

    for filename in tqdm(files_to_process, desc="Processing images", colour=None):
        input_path = os.path.join(input_folder, filename)
        try:
            segmentation_img = Image.open(input_path).convert('RGB')
            seg_array = np.array(segmentation_img)

            if seg_array.shape != static_array.shape:
                base_array = np.array(static_img.resize(segmentation_img.size))
            else:
                base_array = static_array

            mask = np.all(seg_array == 0, axis=2)
            output_array = seg_array.copy()
            output_array[mask] = base_array[mask]

            Image.fromarray(output_array).save(os.path.join(output_folder, filename))
        except Exception as e:
            tqdm.write(f"Could not process '{filename}'. Error: {e}")

    print("\nProcessing complete!")
