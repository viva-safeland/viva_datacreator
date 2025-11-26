"""
Segmented Creator - Step 6: Enhanced Mask Propagation

This module handles the sixth step of the video processing pipeline, which involves:
- Using refined masks from Step 5 as starting points
- Propagating masks both forward and backward through the video
- Ensuring complete segmentation coverage for refined objects
- Creating comprehensive mask files for all frames

This step extends the automatic segmentation from Step 3 by providing
higher-quality starting masks from the interactive refinement in Step 5.
"""

import os
import typer
import vivadatacreator.tooldata as td
import sys
import shutil
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from IPython.display import clear_output
from click import get_current_context
from typing import Dict, Any, Optional

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
from sam2.build_sam import build_sam2_video_predictor  # type: ignore

# Create Typer app
app = typer.Typer()

# Global state for enhanced propagation
estado_global = {
    "inference_state": None,
    "predictor": None,
    "video_segments": {},
}


def load_config(config_path: str = "config.yaml") -> dict:
    return load_runtime_config(config_path)


def save_config(args, config_path: str = "config.yaml") -> None:
    if isinstance(args, dict):
        save_runtime_config(args, config_path)
    else:
        save_runtime_config(vars(args), config_path)


def create_step6_folders(video_path: str) -> dict:
    """Create folders needed for step 6 (progressive folder creation)."""
    if video_path is None:
        raise ValueError("Video path is required for step 6")
    
    path_obj = Path(video_path)
    root = str(path_obj.parent)
    video_path = str(path_obj.resolve())
    
    # Check previous step folders
    imgs_folder_A = os.path.join(root, 'imgsA')
    mask_folder = os.path.join(root, 'masks')
    traked_folder = os.path.join(root, 'traked')
    
    if not os.path.exists(imgs_folder_A):
        raise FileNotFoundError(
            f"Steps 1, 3, and 5 must be completed first. Folder '{imgs_folder_A}' not found."
        )
    if not os.path.exists(mask_folder):
        raise FileNotFoundError(
            f"Step 3 must be completed first. Folder '{mask_folder}' not found."
        )
    if not os.path.exists(traked_folder):
        raise FileNotFoundError(
            f"Step 5 must be completed first. Folder '{traked_folder}' not found."
        )
    
    # Check for Step 5 output
    mask_list_file = os.path.join(root, 'mask_list.csv')
    if not os.path.exists(mask_list_file):
        raise FileNotFoundError(
            f"Step 5 must be completed first. File '{mask_list_file}' not found."
        )
    
    # Create step 6 folders
    aux_folder = os.path.join(root, 'aux_frame')
    os.makedirs(aux_folder, exist_ok=True)
    
    frame_aux = os.path.join(root, 'frame_aux')
    os.makedirs(frame_aux, exist_ok=True)
    
    recorte_folder = os.path.join(root, 'recorte')
    os.makedirs(recorte_folder, exist_ok=True)
    
    return {
        "root": root,
        "video_path": video_path,
        "imgs_folder_A": imgs_folder_A,
        "mask_folder": mask_folder,
        "traked_folder": traked_folder,
        "aux_folder": aux_folder,
        "frame_aux": frame_aux,
        "recorte_folder": recorte_folder,
    }


def refresh_folder(path: str) -> None:
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


def process_step(folders: dict, args: dict, runtime: dict) -> None:
    """Process enhanced mask propagation.
    
    This function uses the refined masks from Step 5 to propagate
    high-quality segmentations both forward and backward through the video.
    
    Args:
        folders: Dictionary containing folder paths
        args: Command line arguments containing processing parameters
    """
    archivo_csv = folders["root"] + '/mask_list.csv'
    csv_path = folders["root"] + f"/track_dic.csv"
    df_a = pd.read_csv(archivo_csv)

    # Check if there are any refined objects to process
    if len(df_a) == 0:
        print("⚠️ No refined objects found in mask_list.csv")
        print("💡 This happens when all objects were skipped in Step 5")
        print("ℹ️ Step 6 is only needed if you refined objects in Step 5")
        print("➡️ You can proceed directly to Step 7")
        return

    # Load image files and prepare data
    _, image_files, files = td.load_and_prepare_data(csv_path, folders["imgs_folder_A"])
    t_imgs = len(files)

    print(f"🎯 Starting enhanced mask propagation...")
    print(f"📊 Processing {len(df_a)} refined objects")
    print(f"🎬 Working with {t_imgs} frames")

    for k in range(len(df_a)):
        print(f'🔄 Processing object {k + 1} of {len(df_a)}')

        # Process each row of the DataFrame
        row = df_a.iloc[k]
        ruta_acc = row["ruta"]
        frame_number = row["frame_number"]
        class_id = row["clase"]
        id_obj = row["id"]

        # Load refined mask
        com_mask = td.read_mask(os.path.join(folders["traked_folder"], ruta_acc))
        com_mask_aux = com_mask.copy()

        ##### Process images forward #####
        i = frame_number
        while i < t_imgs:
            lote = image_files[i:i + args['n_imgs']]

            refresh_folder(folders["aux_folder"])
            copy_files_concurrently(
                folders["imgs_folder_A"],
                folders["aux_folder"],
                lote,
                runtime["copy_workers"],
            )

            print(f'📋 Forward propagation: images {i} to {min(i + args['n_imgs'] - 1, t_imgs - 1)}')

            # Start inference
            estado_global["inference_state"] = estado_global["predictor"].init_state(
                video_path=folders["aux_folder"]
            )
            estado_global["video_segments"] = {}

            # Add mask and propagate
            td.add_object_mask(
                com_mask, id_obj, estado_global["predictor"], 
                estado_global["inference_state"], 0
            )
            td.actualizar_segmentos_video(
                estado_global["predictor"], estado_global["inference_state"], 
                estado_global["video_segments"]
            )
            td.save_masks(
                folders["mask_folder"], args['n_imgs'], i,
                estado_global["video_segments"], class_id, True
            )

            # Clean up auxiliary folder
            files_aux = sorted(os.listdir(folders["aux_folder"]))
            for archivo in files_aux[:-1]:
                os.remove(os.path.join(folders["aux_folder"], archivo))

            # Load new mask for next iteration
            i += args['n_imgs']
            estado_global["predictor"].reset_state(estado_global["inference_state"])
            _, com_mask = next(iter(estado_global["video_segments"][len(estado_global["video_segments"])-1].items()))
            com_mask = com_mask[0].astype(np.uint8)

        refresh_folder(folders["aux_folder"])
        com_mask = com_mask_aux

        ##### Process images backward #####
        i = frame_number
        while i > 0:
            lote = image_files[max(0, i - args['n_imgs']):i+1]

            refresh_folder(folders["aux_folder"])
            copy_files_concurrently(
                folders["imgs_folder_A"],
                folders["aux_folder"],
                lote,
                runtime["copy_workers"],
            )

            print(f'📋 Backward propagation: images {max(0, i - args['n_imgs'])} to {i}')

            estado_global["inference_state"] = estado_global["predictor"].init_state(
                video_path=folders["aux_folder"]
            )
            estado_global["video_segments"] = {}

            td.add_object_mask(
                com_mask, id_obj, estado_global["predictor"], 
                estado_global["inference_state"], len(lote)-1
            )
            td.actualizar_segmentos_video(
                estado_global["predictor"], estado_global["inference_state"], 
                estado_global["video_segments"], reverse=True
            )
            td.save_masks(
                folders["mask_folder"], args['n_imgs'], max(0, i - args['n_imgs']),
                estado_global["video_segments"], class_id, True
            )

            # Clean up auxiliary folder
            files_aux = sorted(os.listdir(folders["aux_folder"]))
            for archivo in files_aux[1:]:
                os.remove(os.path.join(folders["aux_folder"], archivo))

            i -= args['n_imgs']
            estado_global["predictor"].reset_state(estado_global["inference_state"])
            _, com_mask = next(iter(estado_global["video_segments"][0].items()))
            com_mask = com_mask[0].astype(np.uint8)

        clear_output(wait=False)
        
        shutil.rmtree(folders["aux_folder"])
        os.makedirs(folders["aux_folder"], exist_ok=True)
        estado_global["predictor"].reset_state(estado_global["inference_state"])
        estado_global["video_segments"] = {}
        com_mask = []

    td.cleanup_temp_files(folders["frame_aux"], folders["recorte_folder"])
    estado_global["predictor"].reset_state(estado_global["inference_state"])

    shutil.rmtree(folders["aux_folder"])
    os.makedirs(folders["aux_folder"], exist_ok=True)
    
    print(f"✅ Step 6 completed!")
    print(f"🎯 Enhanced mask propagation finished for all refined objects")
    print(f"📁 Updated masks saved to: {folders['mask_folder']}")


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
    """Perform enhanced mask propagation using refined masks.
    
    This is Step 6 of the Segmented Creator pipeline. It uses the
    high-quality refined masks from Step 5 to provide enhanced
    segmentation propagation both forward and backward through the video.
    
    The process:
    1. Loads refined masks from Step 5 (mask_list.csv)
    2. Uses each refined mask as a starting point for propagation
    3. Propagates masks forward through the remaining video
    4. Propagates masks backward to earlier frames
    5. Updates mask files with enhanced segmentations
    
    This step ensures complete segmentation coverage for important
    objects that were refined in Step 5.
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
        folders = create_step6_folders(root)

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
            f"🛠️ Step 6 using {copy_workers} file workers "
            f"(CPU {hardware['cpu_physical']} cores, GPU: {'Yes' if hardware['gpu_available'] else 'No'})"
        )
        runtime = {"copy_workers": copy_workers}

        # Configure SAM2 predictor
        estado_global["predictor"] = configurar_sam2_predictor(model_cfg, sam2_chkpt)

        process_step(folders, ctx_params, runtime)
        
        typer.echo("✅ Step 6 completed successfully!")
        typer.echo(f"📁 Enhanced masks saved to: {folders['mask_folder']}")
        
    except FileNotFoundError as e:
        typer.echo(f"❌ {e}", err=True)
        typer.echo("💡 Make sure to run Steps 1, 3, and 5 first.", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"❌ Error processing video: {e}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
