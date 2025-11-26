"""
Segmented Creator - Step 5: Interactive Mask Refinement

This module handles the fifth step of the video processing pipeline, which involves:
- Interactive refinement of detected objects using tracking data from Step 4
- User-guided segmentation of objects that may have been missed in Step 3
- High-quality mask creation for important objects
- Saving refined masks and their metadata

This step allows users to review and improve the automatic segmentation
from Step 3 by providing high-quality interactive segmentation.
"""

import os
import cv2
import typer
import sys
import shutil
import csv
import numpy as np
import torch
from pathlib import Path
from IPython.display import clear_output
from click import get_current_context
from typing import Optional

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

# Global state for interactive refinement
estado_global = {
    "puntos_interes": [],
    "input_label": np.array([]),
    "terminar": False,
    "omitir": False,
    "inference_state": None,
    "predictor": None,
    "mask": [],
    "ann_obj_id": 1,
    "prompts": {},
    "class_id": None,
    "id_obj": None,
}


def load_config(config_path: str = "config.yaml") -> dict:
    return load_runtime_config(config_path)


def save_config(args, config_path: str = "config.yaml") -> None:
    if isinstance(args, dict):
        save_runtime_config(args, config_path)
    else:
        save_runtime_config(vars(args), config_path)


def create_step5_folders(video_path: str) -> dict:
    """Create folders needed for step 5 (progressive folder creation)."""
    if video_path is None:
        raise ValueError("Video path is required for step 5")
    
    path_obj = Path(video_path)
    root = str(path_obj.parent)
    video_path = str(path_obj.resolve())
    
    # Check previous step folders
    imgs_folder_A = os.path.join(root, 'imgsA')
    frames_folder = os.path.join(root, 'segmentation')
    mask_folder = os.path.join(root, 'masks')
    
    if not os.path.exists(imgs_folder_A):
        raise FileNotFoundError(
            f"Steps 1, 3, and 4 must be completed first. Folder '{imgs_folder_A}' not found."
        )
    if not os.path.exists(frames_folder):
        raise FileNotFoundError(
            f"Step 3 must be completed first. Folder '{frames_folder}' not found."
        )
    if not os.path.exists(mask_folder):
        raise FileNotFoundError(
            f"Step 3 must be completed first. Folder '{mask_folder}' not found."
        )
    
    # Check for Step 4 output
    tracking_file = os.path.join(root, 'track_dic.csv')
    if not os.path.exists(tracking_file):
        raise FileNotFoundError(
            f"Step 4 must be completed first. File '{tracking_file}' not found."
        )
    
    # Create step 5 folders
    aux_folder = os.path.join(root, 'aux_frame')
    os.makedirs(aux_folder, exist_ok=True)
    
    frame_aux = os.path.join(root, 'frame_aux')
    os.makedirs(frame_aux, exist_ok=True)
    
    recorte_folder = os.path.join(root, 'recorte')
    os.makedirs(recorte_folder, exist_ok=True)
    
    traked_folder = os.path.join(root, 'traked')
    os.makedirs(traked_folder, exist_ok=True)
    
    return {
        "root": root,
        "video_path": video_path,
        "imgs_folder_A": imgs_folder_A,
        "frames_folder": frames_folder,
        "mask_folder": mask_folder,
        "aux_folder": aux_folder,
        "frame_aux": frame_aux,
        "recorte_folder": recorte_folder,
        "traked_folder": traked_folder,
    }


def configurar_sam2_predictor(model_cfg: str, sam2_checkpoint: str):
    """Configure SAM2 predictor with proper device settings."""
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
        sys.exit("Failed to configure SAM2 predictor. Please check the configuration and checkpoint files.")


def process_step(folders: dict, args: dict) -> None:
    """Process interactive mask refinement.
    
    This function handles the interactive interface for refining masks
    based on the tracking data from Step 4.
    
    Args:
        folders: Dictionary containing folder paths
        args: Command line arguments containing processing parameters
    """
    global estado_global

    csv_path = folders["root"] + "/track_dic.csv"

    # Get next available object ID
    files = [f for f in os.listdir(folders["mask_folder"]) 
             if f.endswith(('.png', '.jpg', '.jpeg')) and f.startswith('outmask_fr0')]
    if files:
        estado_global["id_obj"] = int(max(files, key=lambda x: int(x.split('_')[2][2:])).split('_')[2][2:]) + 1
    else:
        estado_global["id_obj"] = 1

    # Load tracking data
    df, image_files, files = td.load_and_prepare_data(csv_path, folders["imgs_folder_A"])
    td.cleanup_temp_files(folders["frame_aux"], folders["recorte_folder"])

    # Setup CSV file for mask list
    archivo_csv = folders["root"] + '/mask_list.csv'
    campos = ['ruta', 'frame_number', 'clase', 'id']
    
    if not os.path.exists(archivo_csv):
        with open(archivo_csv, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=campos)
            writer.writeheader()

    print(f"🎯 Starting interactive mask refinement...")
    print(f"📊 Processing {len(df)} detected objects")
    print("💡 Controls:")
    print("  • Left-click: Add positive point")
    print("  • Right-click: Add negative point")
    print("  • Press 'a': Accept current object and assign class")
    print("  • Press '0': Skip current object")
    print("  • Press 'ESC': Finish and exit")
    print()

    with open(archivo_csv, mode='a', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=campos)
        
        for j in range(len(df)):
            # Process current row
            row = df.iloc[j]
            
            # Get detected class if available
            detected_class = row.get("class_name", "unknown") if "class_name" in df.columns else "unknown"
            print(f'🎬 Processing object {j + 1} of {len(df)} - YOLO detected: {detected_class.upper()}')

            # Reset state for new object
            estado_global["puntos_interes"] = []
            estado_global["input_label"] = np.array([])
            estado_global["terminar"] = False
            estado_global["omitir"] = False

            frame_number = row["frame_number"] - 1
            
            recorte, imagen, bbox, archivo = td.process_frame(
                row, image_files, folders["imgs_folder_A"], folders["frame_aux"], 
                folders["frames_folder"], folders["recorte_folder"]
            )

            # Initialize SAM2 predictor
            estado_global["inference_state"] = estado_global["predictor"].init_state(folders["recorte_folder"])
            estado_global["predictor"].reset_state(estado_global["inference_state"])
            
            # Handle user interaction
            td.handle_user_interaction(recorte, estado_global)

            if estado_global["terminar"] == True:
                print("🏁 Finishing the process.")
                break

            if estado_global["omitir"] == True:
                estado_global["omitir"] = False
                print("⏭️ Object skipped.")
            else:
                # Save mask
                com_mask = td.create_composite_mask(estado_global["mask"], imagen.shape, bbox)
                nam_aux = f"fr{frame_number}_id{estado_global['id_obj']}_cl{estado_global['class_id']}.png"

                cv2.imwrite(os.path.join(folders["traked_folder"], nam_aux), com_mask * 255)

                writer.writerow({
                    'ruta': nam_aux,
                    'frame_number': frame_number,
                    'clase': estado_global["class_id"],
                    'id': estado_global["id_obj"]
                })

                print(f"✅ Object {estado_global['id_obj']} saved with class: {estado_global['class_id']}")
                estado_global["id_obj"] += 1
                com_mask = []
            
            clear_output(wait=False)
            td.cleanup_temp_files(folders["frame_aux"], folders["recorte_folder"])
            estado_global["predictor"].reset_state(estado_global["inference_state"])

            shutil.rmtree(folders["aux_folder"])
            os.makedirs(folders["aux_folder"], exist_ok=True)

    print(f"✅ Step 5 completed!")
    print(f"📁 Refined masks saved to: {folders['traked_folder']}")
    print(f"📋 Mask list saved to: {archivo_csv}")


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
    """Perform interactive refinement of detected objects.
    
    This is Step 5 of the Segmented Creator pipeline. It allows users to
    review and refine the segmentation results from Steps 3 and 4 by
    providing high-quality interactive segmentation for important objects.
    
    The process:
    1. Loads tracking data from Step 4 (track_dic.csv)
    2. Shows cropped regions of detected objects
    3. Allows interactive refinement using SAM2
    4. Saves high-quality masks and metadata
    
    This step is crucial for creating a high-quality dataset by ensuring
    important objects are properly segmented.
    """
    config = load_config()

    root = resolve_cli_value("root", root, config)
    fac = resolve_cli_value("fac", fac, config)
    sam2_chkpt = resolve_cli_value("sam2_chkpt", sam2_chkpt, config)
    model_cfg = resolve_cli_value("model_cfg", model_cfg, config)
    n_imgs = resolve_cli_value("n_imgs", n_imgs, config)
    n_obj = resolve_cli_value("n_obj", n_obj, config)
    img_size_sahi = resolve_cli_value("img_size_sahi", img_size_sahi, config)
    overlap_sahi = resolve_cli_value("overlap_sahi", overlap_sahi, config)
    
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
        # Create necessary folders for step 5
        folders = create_step5_folders(root)

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
            f"🛠️ Step 5 running on {hardware['cpu_physical']} CPU cores "
            f"(GPU: {'Yes' if hardware['gpu_available'] else 'No'})"
        )

        # Configure SAM2 predictor
        estado_global["predictor"] = configurar_sam2_predictor(model_cfg, sam2_chkpt)

        # Process interactive refinement
        process_step(folders, ctx_params)
        
        typer.echo("✅ Step 5 completed successfully!")
        typer.echo(f"📁 Refined masks saved to: {folders['traked_folder']}")
        
    except FileNotFoundError as e:
        typer.echo(f"❌ {e}", err=True)
        typer.echo("💡 Make sure to run Steps 1, 3, and 4 first.", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"❌ Error processing video: {e}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
