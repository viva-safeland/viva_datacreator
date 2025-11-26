"""
Segmented Creator - Step 8: Final Dataset Creation

This module handles the eighth and final step of the video processing pipeline, which involves:
- Creating the final semantic segmentation dataset
- Organizing images and masks into dataset format
- Creating training/validation splits
- Generating dataset metadata and documentation

This step completes the pipeline by creating a ready-to-use semantic
segmentation dataset from the processed video.
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


def create_step8_folders(video_path: str) -> dict:
    """Create folders needed for step 8 (progressive folder creation)."""
    if video_path is None:
        raise ValueError("Video path is required for step 8")
    
    path_obj = Path(video_path)
    root = str(path_obj.parent)
    video_path = str(path_obj.resolve())
    
    # Check previous step folders
    semantic_folder = os.path.join(root, 'semantic')
    if not os.path.exists(semantic_folder):
        raise FileNotFoundError(
            f"Step 7 must be completed first. Folder '{semantic_folder}' not found."
        )
    
    # Create step 8 folders
    dataset_folder = os.path.join(root, 'dataset')
    os.makedirs(dataset_folder, exist_ok=True)
    
    return {
        "root": root,
        "video_path": video_path,
        "semantic_folder": semantic_folder,
        "dataset_folder": dataset_folder,
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
    """Create the final semantic segmentation dataset.
    
    This is Step 8 of the Segmented Creator pipeline. It creates the
    final semantic segmentation dataset from the processed video.
    
    The process:
    1. Loads semantic segmentation maps from Step 7 (semantic folder)
    2. Organizes images and masks into dataset format
    3. Creates training/validation splits
    4. Generates dataset metadata and documentation
    5. Outputs a ready-to-use dataset
    
    This completes the entire segmentation pipeline and produces
    a dataset that can be used for training semantic segmentation models.
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
        folders = create_step8_folders(root)

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
        
        typer.echo("🏁 Creating final semantic segmentation dataset...")
        
        # Create final dataset
        td.final_dataset(
            folders["semantic_folder"], 
            folders["dataset_folder"], 
            os.path.join(folders["root"], "static.png")
        )
        
        typer.echo("✅ Step 8 completed successfully!")
        typer.echo("🎉 Complete segmentation pipeline finished!")
        typer.echo(f"📁 Final dataset saved to: {folders['dataset_folder']}")
        typer.echo("🎯 The dataset is ready for training semantic segmentation models.")
        
    except FileNotFoundError as e:
        typer.echo(f"❌ {e}", err=True)
        typer.echo("💡 Make sure to run Steps 1-7 first.", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"❌ Error processing: {e}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
