"""
Segmented Creator - Step 1: Frame Extraction

This module handles the first step of the video processing pipeline, which involves:
- Extracting frames from the input video
- Aligning frames to correct camera vibrations
- Saving frames as individual images
- Creating an aligned video file
- Creating a static background image (static.png) from the first frame

The frame alignment uses feature matching to stabilize the video sequence,
which is crucial for consistent segmentation across frames.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import typer
from click import get_current_context
from tqdm import tqdm

import vivadatacreator.tooldata as td
from vivadatacreator.runtime_utils import (
    configure_opencv_threads,
    gather_hardware_profile,
    load_runtime_config,
    recommended_workers,
    resolve_cli_value,
    save_runtime_config,
)

# Create Typer app
app = typer.Typer()


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    return load_runtime_config(config_path)


def save_config(args: Dict[str, Any], config_path: str = "config.yaml") -> None:
    save_runtime_config(args, config_path)


def align_frame(frame0, frame):
    """Align a single frame to the reference frame."""
    return td.alinear_imagen(frame0, frame)


def create_initial_folders(video_path: str) -> dict:
    """Create only the essential folders needed for step 1.
    
    This function creates folders progressively as needed, starting with
    only the folders required for frame extraction.
    
    Args:
        video_path: Path to the input video file
        
    Returns:
        Dictionary containing paths to created folders
    """
    if video_path is None:
        raise ValueError("Video path is required for step 1")
    
    path_obj = Path(video_path)
    root = str(path_obj.parent)
    video_path = str(path_obj.resolve())
    
    # Create aligned video path
    video_dir = os.path.join(root, 'video_alineado.mp4')
    
    # Create folder for aligned frames (step 1 requirement)
    imgs_folder_A = os.path.join(root, 'imgsA')
    os.makedirs(imgs_folder_A, exist_ok=True)
    
    return {
        "root": root,
        "video_path": video_path,
        "video_dir": video_dir,
        "imgs_folder_A": imgs_folder_A,
    }


def plan_alignment_runtime(args: Dict[str, Any], frame_shape: tuple[int, int, int]) -> Dict[str, Any]:
    hardware = gather_hardware_profile()
    worker_count = recommended_workers(args.get("max_workers"), hardware, upper_cap=hardware["cpu_logical"])
    bytes_per_frame = frame_shape[0] * frame_shape[1] * frame_shape[2]
    requested_chunk = args.get("chunk_size")
    if requested_chunk:
        chunk_size = max(worker_count, int(requested_chunk))
    else:
        budget = max(bytes_per_frame, int(hardware["available_ram_bytes"] * 0.04))
        chunk_size = budget // max(bytes_per_frame, 1)
        chunk_size = max(worker_count, min(256, chunk_size))

    configure_opencv_threads(worker_count)

    return {
        "hardware": hardware,
        "workers": worker_count,
        "chunk_size": chunk_size,
        "bytes_per_frame": bytes_per_frame,
    }


def log_runtime_plan(runtime: Dict[str, Any], total_frames: int) -> None:
    hardware = runtime["hardware"]
    typer.echo("🛠️  Hardware profile detected:")
    typer.echo(f"   • CPU: {hardware['cpu_physical']} cores ({hardware['cpu_logical']} threads)")
    typer.echo(f"   • RAM available: {hardware['available_ram_bytes'] / (1024 ** 3):.2f} GB")
    if hardware["gpu_available"]:
        typer.echo(f"   • GPU: {hardware['gpu_name']} ({hardware['total_vram_bytes'] / (1024 ** 3):.1f} GB VRAM)")
    typer.echo("")
    typer.echo("📋 Runtime plan:")
    typer.echo(f"   • Frames to process: {total_frames}")
    typer.echo(f"   • Thread workers: {runtime['workers']}")
    typer.echo(f"   • Chunk size: {runtime['chunk_size']} frames (~{runtime['chunk_size'] * runtime['bytes_per_frame'] / (1024 ** 2):.1f} MB)")
    typer.echo("")


def procesar_video(folders: dict, args: Dict[str, Any]) -> None:
    """Process video and extract aligned frames using streaming + threading."""
    cap = cv2.VideoCapture(folders["video_path"])
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    ret, frame0 = cap.read()
    if not ret or frame0 is None:
        typer.echo("❌ Unable to read first frame from the video.", err=True)
        return

    runtime = plan_alignment_runtime(args, frame0.shape)
    log_runtime_plan(runtime, total_frames or 1)

    frame_filename = os.path.join(folders["imgs_folder_A"], f"{0:05d}.jpg")
    cv2.imwrite(frame_filename, frame0)

    # Save the first frame as static.png for use in later steps
    static_filename = os.path.join(folders["root"], "static.png")
    cv2.imwrite(static_filename, frame0)

    processed = 1
    chunk_size = runtime["chunk_size"]

    def _align_single(frame):
        return td.alinear_imagen(frame0, frame)

    remaining = max(0, (total_frames or processed) - 1)
    with ThreadPoolExecutor(max_workers=runtime["workers"]) as pool, tqdm(
        total=remaining,
        desc="Saving aligned frames",
        unit="frame",
        colour=None,
    ) as progress:
        while True:
            chunk = []
            for _ in range(chunk_size):
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                chunk.append(frame)
            if not chunk:
                break

            aligned = list(pool.map(_align_single, chunk))
            for aligned_frame in aligned:
                frame_path = os.path.join(folders["imgs_folder_A"], f"{processed:05d}.jpg")
                cv2.imwrite(frame_path, aligned_frame)
                processed += 1
            progress.update(len(aligned))

    cap.release()
    typer.echo(f"✅ Saved {processed} aligned frames into '{folders['imgs_folder_A']}'.")
    td.crear_video(folders["imgs_folder_A"], folders["video_dir"], fps=30, codec="mp4v")


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
    max_workers: Optional[int] = typer.Option(None, "--max-workers", help="Thread count for frame alignment"),
    chunk_size: Optional[int] = typer.Option(None, "--chunk-size", help="Frames processed per chunk (auto if omitted)"),
) -> None:
    """Extract and align frames from input video.

    This is Step 1 of the Segmented Creator pipeline. It processes the input video
    to extract individual frames, applies alignment to correct camera vibrations,
    and creates both a folder of aligned images, an aligned video file, and a
    static background image (static.png) from the first frame.

    The alignment process uses feature matching (ORB) to stabilize frames and
    reduce camera shake, which is essential for consistent segmentation results.
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
    max_workers = resolve_cli_value("max_workers", max_workers, config)
    chunk_size = resolve_cli_value("chunk_size", chunk_size, config)
    
    # Validate required arguments
    if root is None:
        typer.echo("Error: Video path (--root) is required for step 1", err=True)
        raise typer.Exit(code=1)
    
    try:
        # Create necessary folders for step 1
        folders = create_initial_folders(root)
        
        # Update root in case it was inferred
        if root != folders["video_path"]:
            root = folders["video_path"]
        
        # Configuration is already loaded and updated
        
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
            max_workers=max_workers,
            chunk_size=chunk_size,
        )
        save_config(ctx_params)

        procesar_video(folders, ctx_params)
        
        typer.echo("✅ Step 1 completed successfully!")
        typer.echo(f"📁 Frames saved to: {folders['imgs_folder_A']}")
        typer.echo(f"🎥 Aligned video created: {folders['video_dir']}")
        
    except Exception as e:
        typer.echo(f"❌ Error processing video: {e}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
