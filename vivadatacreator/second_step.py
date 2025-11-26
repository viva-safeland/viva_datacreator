"""
Segmented Creator - Step 2: Interactive Initial Segmentation

This module handles the second step of the video processing pipeline, which involves:
- Interactive segmentation of objects in the first frame using SAM2
- User-guided annotation with positive/negative clicks
- Class assignment for segmented objects
- Saving prompts for automatic propagation in subsequent steps

This step creates the initial segmentation prompts that will be used
in Step 3 for automatic mask propagation throughout the video.
"""

import os
import cv2
import typer
import csv
import shutil
import numpy as np
import torch
import sys
from pathlib import Path
from typing import Optional
from click import get_current_context

import vivadatacreator.tooldata as td
from vivadatacreator.runtime_utils import (
    configure_opencv_threads,
    gather_hardware_profile,
    load_runtime_config,
    resolve_cli_value,
    save_runtime_config,
)
from vivadatacreator.sam2_resources import config_to_hydra_name
from sam2.build_sam import build_sam2_video_predictor  # type: ignore

# Create Typer app
app = typer.Typer()

# Global state for interactive segmentation
estado_global = {
    "prompts": {},
    "puntos_interes": [],
    "input_label": np.array([]),
    "ann_obj_id": 1,
    "mask": [],
    "predictor": None,
    "inference_state": None,
}


def load_config(config_path: str = "config.yaml") -> dict:
    return load_runtime_config(config_path)


def save_config(args, config_path: str = "config.yaml") -> None:
    if isinstance(args, dict):
        save_runtime_config(args, config_path)
    else:
        save_runtime_config(vars(args), config_path)


def create_step2_folders(video_path: str) -> dict:
    """Create folders needed for step 2 (progressive folder creation).
    
    This function creates folders progressively, adding only those needed
    for step 2 operations.
    
    Args:
        video_path: Path to the input video file
        
    Returns:
        Dictionary containing paths to created folders
    """
    if video_path is None:
        raise ValueError("Video path is required for step 2")
    
    path_obj = Path(video_path)
    root = str(path_obj.parent)
    video_path = str(path_obj.resolve())
    
    # Check if step 1 folders exist
    imgs_folder_A = os.path.join(root, 'imgsA')
    if not os.path.exists(imgs_folder_A):
        raise FileNotFoundError(
            f"Step 1 must be completed first. Folder '{imgs_folder_A}' not found."
        )
    
    # Create additional folders needed for step 2
    aux_folder = os.path.join(root, 'aux_frame')
    os.makedirs(aux_folder, exist_ok=True)
    
    frame_aux = os.path.join(root, 'frame_aux')
    os.makedirs(frame_aux, exist_ok=True)
    
    return {
        "root": root,
        "video_path": video_path,
        "imgs_folder_A": imgs_folder_A,
        "aux_folder": aux_folder,
        "frame_aux": frame_aux,
    }


def configurar_sam2_predictor(model_cfg: str, sam2_checkpoint: str) -> object:
    """Configure SAM2 predictor with proper device settings.
    
    Args:
        model_cfg: Path to the SAM2 model configuration file
        sam2_checkpoint: Path to the SAM2 model checkpoint file
        
    Returns:
        Configured SAM2 predictor instance
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        # Enable bfloat16 globally
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

        # Enable tf32 on Ampere GPUs (Compute Capability >= 8)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    try:
        hydra_model_cfg = config_to_hydra_name(model_cfg)
        predictor = build_sam2_video_predictor(hydra_model_cfg, sam2_checkpoint, device=device)
        print("SAM2 predictor configured correctly.")
        return predictor
    except Exception as e:
        print(f"Error configuring SAM2 predictor: {e}")
        sys.exit("Failed to configure SAM2 predictor. Check the configuration and checkpoint files.")


def process_step(folders: dict, fac: int) -> None:
    """Process interactive segmentation step.
    
    This function handles the interactive user interface for initial object
    segmentation. Users can click on objects to create positive/negative
    prompts and assign classes.
    
    Args:
        folders: Dictionary containing folder paths
        fac: Scaling factor for frame resizing
    """
    # Clean and recreate auxiliary folder
    shutil.rmtree(folders['aux_folder'])
    os.makedirs(folders['aux_folder'], exist_ok=True)

    # Load first frame
    files = os.listdir(folders['imgs_folder_A'])
    image_files = [f for f in sorted(files) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))]

    # Initialize and show frame
    img = cv2.imread(os.path.join(folders['imgs_folder_A'], image_files[0]))
    h, w = img.shape[:2]
    scaled_frame = cv2.resize(img, (int(w / fac), int(h / fac)))
    cv2.imwrite(os.path.join(folders['frame_aux'], image_files[0]), scaled_frame)
    
    # Initialize SAM2 predictor state
    estado_global["inference_state"] = estado_global["predictor"].init_state(video_path=folders['frame_aux'])
    
    # Set up interactive window
    cv2.imshow("Frame", scaled_frame)
    cv2.setMouseCallback("Frame", td.on_click, param={"frame": scaled_frame, "estado": estado_global})

    print("\n🎯 Interactive Segmentation Instructions:")
    print("  • Left-click: Add positive point (includes object part)")
    print("  • Right-click: Add negative point (excludes object part)")
    print("  • Press 'a': Accept current object and assign class")
    print("  • Press 'ESC': Finish segmentation and exit")
    print("  • Enter class ID (0-25) when prompted")
    print()

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 97:  # 'a' key
            class_id = td.get_class_from_user()
            if class_id is not None:
                estado_global["prompts"][estado_global["ann_obj_id"]] = (
                    estado_global["puntos_interes"], 
                    estado_global["input_label"], 
                    class_id
                )
                print(f"✅ Object {estado_global['ann_obj_id']} saved with class: {class_id}")
                estado_global["puntos_interes"] = []
                estado_global["input_label"] = np.array([])
                estado_global["ann_obj_id"] += 1

        elif key == 27:  # ESC key
            class_id = td.get_class_from_user()
            if class_id is not None:
                estado_global["prompts"][estado_global["ann_obj_id"]] = (
                    estado_global["puntos_interes"], 
                    estado_global["input_label"], 
                    class_id
                )
                print(f"✅ Final object {estado_global['ann_obj_id']} saved with class: {class_id}")
            break

    cv2.destroyAllWindows()

    # Save prompts to CSV file
    csv_path = os.path.join(folders['root'], 'mask_prompts.csv')
    with open(csv_path, 'w', newline='') as archivo:
        writer = csv.writer(archivo)
        for clave, valor in estado_global["prompts"].items():
            writer.writerow([clave, valor])
    
    print(f"💾 Prompts saved to: {csv_path}")
    print(f"📊 Total objects segmented: {len(estado_global['prompts'])}")


@app.command()
def main(
    root: Optional[str] = typer.Option(None, "--root", help="Path to the video file"),
    fac: Optional[int] = typer.Option(None, "--fac", help="Scaling factor for resizing images"),
    model_cfg: Optional[str] = typer.Option(None, "--model-cfg", help="Path to the SAM2 model configuration file"),
    sam2_chkpt: Optional[str] = typer.Option(None, "--sam2-chkpt", help="Path to the SAM2 model checkpoint file"),
    n_imgs: Optional[int] = typer.Option(None, "--n-imgs", help="Number of images to process per batch"),
    n_obj: Optional[int] = typer.Option(None, "--n-obj", help="Number of objects to process per batch"),
    img_size_sahi: Optional[int] = typer.Option(None, "--img-size-sahi", help="Image size for the SAHI model"),
    overlap_sahi: Optional[float] = typer.Option(None, "--overlap-sahi", help="Overlap threshold for SAHI detections"),
) -> None:
    """Perform interactive initial segmentation of objects.
    
    This is Step 2 of the Segmented Creator pipeline. It provides an interactive
    interface for users to segment objects in the first frame using SAM2.
    
    Users can:
    - Click on objects to create positive/negative prompts
    - Assign classes to segmented objects
    - Save prompts for automatic propagation in Step 3
    
    The interactive window will appear where you can perform segmentation.
    """
    config = load_config()

    root = resolve_cli_value("root", root, config)
    fac = resolve_cli_value("fac", fac, config)
    sam2_chkpt = resolve_cli_value("sam2_chkpt", sam2_chkpt, config)
    model_cfg = resolve_cli_value("model_cfg", model_cfg, config)
    
    # Validate required arguments
    if root is None:
        typer.echo("Error: Video path (--root) is required", err=True)
        raise typer.Exit(code=1)
    if fac is None:
        typer.echo("Error: Scaling factor (--fac) is required", err=True)
        raise typer.Exit(code=1)
    if sam2_chkpt is None:
        typer.echo("Error: SAM2 checkpoint (--sam2-chkpt) is required", err=True)
        raise typer.Exit(code=1)
    
    try:
        # Create necessary folders for step 2
        folders = create_step2_folders(root)

        ctx_params = dict(get_current_context().params)
        ctx_params.update(
            root=root,
            fac=fac,
            sam2_chkpt=sam2_chkpt,
            model_cfg=model_cfg,
            n_imgs=n_imgs,
            n_obj=n_obj,
            img_size_sahi=img_size_sahi,
            overlap_sahi=overlap_sahi,
        )
        save_config(ctx_params)

        hardware = gather_hardware_profile()
        configure_opencv_threads(min(8, hardware["cpu_physical"]))
        typer.echo(
            f"🛠️ Running Step 2 on {hardware['cpu_physical']} cores "
            f"(GPU: {'Yes' if hardware['gpu_available'] else 'No'})"
        )
        
        # Configure SAM2 predictor
        estado_global["predictor"] = configurar_sam2_predictor(model_cfg, sam2_chkpt)

        # Process interactive segmentation
        process_step(folders, fac)
        
        typer.echo("✅ Step 2 completed successfully!")
        typer.echo(f"📁 Prompts saved to: {folders['root']}/mask_prompts.csv")
        
    except FileNotFoundError as e:
        typer.echo(f"❌ {e}", err=True)
        typer.echo("💡 Make sure to run Step 1 first to create the necessary folders.", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"❌ Error processing video: {e}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
