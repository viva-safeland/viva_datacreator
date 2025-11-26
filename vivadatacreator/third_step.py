"""
Segmented Creator - Step 3: Automatic Mask Propagation

This module handles the third step of the video processing pipeline, which involves:
- Loading initial segmentation prompts from Step 2
- Automatically propagating masks throughout the video using SAM2
- Processing video in batches for memory efficiency
- Creating mask files and grouped segmentation frames

This step uses the interactive prompts created in Step 2 to automatically
segment objects throughout the entire video sequence.
"""

import os
import cv2
import typer
import csv
import shutil
import numpy as np
import torch
import sys
import re
from pathlib import Path
from typing import Dict, Any, Optional
from tqdm import tqdm
from sam2.build_sam import build_sam2_video_predictor  # type: ignore
from click import get_current_context

import vivadatacreator.tooldata as td
from vivadatacreator.runtime_utils import (
    configure_opencv_threads,
    copy_files_concurrently,
    gather_hardware_profile,
    load_runtime_config,
    recommended_workers,
    resolve_cli_value,
    save_runtime_config,
)
from vivadatacreator.sam2_resources import config_to_hydra_name

# Create Typer app
app = typer.Typer()

# Global state for mask propagation
estado_global = {
    "prompts": {},
    "puntos_interes": [],
    "input_label": np.array([]),
    "ann_obj_id": 1,
    "mask": [],
    "predictor": None,
    "inference_state": None,
    "cl": [],
    "video_segments": {}
}


def load_config(config_path: str = "config.yaml") -> dict:
    return load_runtime_config(config_path)


def save_config(args, config_path: str = "config.yaml") -> None:
    if isinstance(args, dict):
        save_runtime_config(args, config_path)
    else:
        save_runtime_config(vars(args), config_path)


def create_step3_folders(video_path: str) -> dict:
    """Create folders needed for step 3 (progressive folder creation)."""
    if video_path is None:
        raise ValueError("Video path is required for step 3")
    
    path_obj = Path(video_path)
    root = str(path_obj.parent)
    video_path = str(path_obj.resolve())
    
    # Check previous step folders
    imgs_folder_A = os.path.join(root, 'imgsA')
    if not os.path.exists(imgs_folder_A):
        raise FileNotFoundError(
            f"Steps 1 and 2 must be completed first. Folder '{imgs_folder_A}' not found."
        )
    
    # Check for Step 2 outputs
    prompts_file = os.path.join(root, 'mask_prompts.csv')
    if not os.path.exists(prompts_file):
        raise FileNotFoundError(
            f"Step 2 must be completed first. File '{prompts_file}' not found."
        )
    
    # Create step 3 folders
    aux_folder = os.path.join(root, 'aux_frame')
    os.makedirs(aux_folder, exist_ok=True)
    
    mask_folder = os.path.join(root, 'masks')
    os.makedirs(mask_folder, exist_ok=True)
    
    frames_folder = os.path.join(root, 'segmentation')
    os.makedirs(frames_folder, exist_ok=True)
    
    # Create aligned video path
    video_dir = os.path.join(root, 'video_alineado.mp4')
    
    return {
        "root": root,
        "video_path": video_path,
        "video_dir": video_dir,
        "imgs_folder_A": imgs_folder_A,
        "aux_folder": aux_folder,
        "mask_folder": mask_folder,
        "frames_folder": frames_folder,
    }


def refresh_aux_folder(path: str) -> None:
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)


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
        sys.exit("Failed to configure SAM2 predictor. Check the configuration and checkpoint files.")


def process_step(folders: dict, fac: int, n_imgs: int, n_obj: int, runtime: Dict[str, Any]) -> None:
    """Process automatic mask propagation.
    
    This function processes the video in batches, using the prompts from Step 2
    to automatically propagate masks throughout the video sequence.
    
    Args:
        folders: Dictionary containing folder paths
        fac: Scaling factor for frame resizing
        n_imgs: Number of images to process per batch
        n_obj: Number of objects to process per batch
    """
    # Get list of image files
    files = os.listdir(folders['imgs_folder_A'])
    t_imgs = len(files)
    image_files = [f for f in sorted(files) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))]

    i = 0  # Processed image counter
    cnt = 0  # Object counter
    ini_id = 1  # Initial ID for objects

    print(f"🎬 Starting mask propagation for {t_imgs} images...")
    print(f"📦 Processing in batches of {n_imgs} images")
    print(f"🎯 Processing {n_obj} objects per batch")

    copy_workers = runtime["copy_workers"]

    while i < t_imgs:
        lote = image_files[i:i + n_imgs]

        refresh_aux_folder(folders["aux_folder"])
        copy_files_concurrently(
            folders["imgs_folder_A"],
            folders["aux_folder"],
            lote,
            copy_workers,
        )

        print(f"📋 Processing batch: images {i} to {min(i + n_imgs - 1, t_imgs - 1)}")

        # Initialize SAM2 inference state
        estado_global["inference_state"] = estado_global["predictor"].init_state(
            video_path=folders['aux_folder']
        )

        # Create masks
        estado_global["video_segments"] = {}
        if i == 0:
            # First batch: load prompts and process initial objects
            estado_global["prompts"] = td.leer_prompts(
                os.path.join(folders['root'], 'mask_prompts.csv')
            )
            td.procesar_prompts(estado_global, ini_id, n_obj, n_imgs, fac, folders['mask_folder'])
        else:
            # Subsequent batches: use previous masks as starting points
            for filename in sorted(os.listdir(folders['mask_folder'])):
                if filename.startswith(f'outmask_fr{i - 1}'):
                    if cnt >= n_obj:
                        td.actualizar_segmentos_video(
                            estado_global["predictor"], 
                            estado_global["inference_state"], 
                            estado_global["video_segments"]
                        )
                        td.save_masks(
                            folders['mask_folder'], 
                            n_imgs, 
                            i, 
                            estado_global["video_segments"], 
                            estado_global["cl"]
                        )
                        estado_global["predictor"].reset_state(estado_global["inference_state"])
                        cnt = 0
                    
                    new_mask = td.read_mask(os.path.join(folders['mask_folder'], filename))
                    id_obj = re.search(r"id(\d+)", filename).group(1)
                    cl_obj = re.search(r"id(\w+)", filename).group(1)
                    td.add_object_mask(
                        new_mask, 
                        id_obj, 
                        estado_global["predictor"], 
                        estado_global["inference_state"], 
                        0
                    )
                    cnt += 1
            
            td.actualizar_segmentos_video(
                estado_global["predictor"], 
                estado_global["inference_state"], 
                estado_global["video_segments"]
            )
            td.save_masks(
                folders['mask_folder'], 
                n_imgs, 
                i, 
                estado_global["video_segments"], 
                estado_global["cl"]
            )
            cnt = 0

        # Clean up auxiliary folder
        files_aux = sorted(os.listdir(folders['aux_folder']))
        for archivo in files_aux[:-1]:  # Keep last file for next batch
            os.remove(os.path.join(folders['aux_folder'], archivo))

        i += n_imgs
        estado_global["predictor"].reset_state(estado_global["inference_state"])

    # Group masks by frame
    print("🔗 Grouping masks by frame...")
    td.group_masks(folders['mask_folder'], folders['frames_folder'])
    
    print(f"✅ Step 3 completed!")
    print(f"📁 Masks saved to: {folders['mask_folder']}")
    print(f"🖼️ Grouped frames saved to: {folders['frames_folder']}")


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
    """Automatically propagate masks throughout the video.
    
    This is Step 3 of the Segmented Creator pipeline. It uses the interactive
    prompts created in Step 2 to automatically segment objects throughout
    the entire video sequence using SAM2.
    
    The process:
    1. Loads prompts from Step 2 (mask_prompts.csv)
    2. Processes video in batches for memory efficiency
    3. Propagates masks forward through the video
    4. Groups masks by frame for visualization
    
    This step creates the foundation for the final semantic segmentation.
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
        # Create necessary folders for step 3
        folders = create_step3_folders(root)
        
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
        copy_workers = recommended_workers(None, hardware, upper_cap=8)
        configure_opencv_threads(copy_workers)
        typer.echo(
            f"🛠️ Step 3 using {copy_workers} copy workers "
            f"(CPU {hardware['cpu_physical']} cores, GPU: {'Yes' if hardware['gpu_available'] else 'No'})"
        )

        runtime = {"copy_workers": copy_workers}

        estado_global["predictor"] = configurar_sam2_predictor(model_cfg, sam2_chkpt)

        process_step(folders, fac, n_imgs, n_obj, runtime)
        
        typer.echo("✅ Step 3 completed successfully!")
        
    except FileNotFoundError as e:
        typer.echo(f"❌ {e}", err=True)
        typer.echo("💡 Make sure to run Steps 1 and 2 first.", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"❌ Error processing video: {e}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
