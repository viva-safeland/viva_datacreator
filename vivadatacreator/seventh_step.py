"""
Segmented Creator - Step 7: Color-Based Semantic Segmentation

This module handles the seventh step of the video processing pipeline, which involves:
- Loading masks from Step 6
- Creating color-coded semantic segmentation maps
- Organizing masks by semantic classes
- Preparing for final dataset creation

This step transforms the individual object masks into a coherent
semantic segmentation map where each class has a distinct color.
"""

import os
from pathlib import Path
from typing import Optional

import typer
from click import get_current_context

import vivadatacreator.tooldata as td
from vivadatacreator.runtime_utils import (
    load_runtime_config,
    resolve_cli_value,
    save_runtime_config,
)

# Create Typer app
app = typer.Typer()


def load_config(config_path: str = "config.yaml") -> dict:
    return load_runtime_config(config_path)


def save_config(args, config_path: str = "config.yaml") -> None:
    if isinstance(args, dict):
        save_runtime_config(args, config_path)
    else:
        save_runtime_config(vars(args), config_path)


def create_step7_folders(video_path: str) -> dict:
    """Create folders needed for step 7 (progressive folder creation)."""
    if video_path is None:
        raise ValueError("Video path is required for step 7")
    
    path_obj = Path(video_path)
    root = str(path_obj.parent)
    video_path = str(path_obj.resolve())
    
    # Check previous step folders
    mask_folder = os.path.join(root, 'masks')
    if not os.path.exists(mask_folder):
        raise FileNotFoundError(
            f"Steps 3 and 6 must be completed first. Folder '{mask_folder}' not found."
        )
    
    # Create step 7 folders
    semantic_folder = os.path.join(root, 'semantic')
    os.makedirs(semantic_folder, exist_ok=True)
    
    return {
        "root": root,
        "video_path": video_path,
        "mask_folder": mask_folder,
        "semantic_folder": semantic_folder,
    }


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
    """Create color-coded semantic segmentation maps.
    
    This is Step 7 of the Segmented Creator pipeline. It organizes
    and colors the masks from Step 6 to create semantic segmentation
    maps where each class has a distinct color.
    
    The process:
    1. Loads masks from Step 6 (masks folder)
    2. Groups masks by semantic class
    3. Applies color coding to each class
    4. Creates semantic segmentation maps
    
    This step prepares the segmentation data for the final dataset creation.
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
    
    try:
        folders = create_step7_folders(root)

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
        
        typer.echo("🎨 Creating color-coded semantic segmentation maps...")
        
        # Create semantic segmentation
        td.group_masks_color(folders["mask_folder"], folders["semantic_folder"])
        
        typer.echo("✅ Step 7 completed successfully!")
        typer.echo(f"📁 Semantic segmentation maps saved to: {folders['semantic_folder']}")
        
    except FileNotFoundError as e:
        typer.echo(f"❌ {e}", err=True)
        typer.echo("💡 Make sure to run Steps 3 and 6 first.", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"❌ Error processing: {e}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
