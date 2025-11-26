"""
Segmented Creator - Step 4 (Adaptive & Resource-Aware)

This module performs object detection (YOLO + SAHI) and tracking (DeepSort) while
automatically adapting to the host hardware (CPU cores, RAM availability and GPU
capabilities). Key improvements:

- Hardware introspection adjusts batch sizes, SAHI slice sizes and device usage.
- Optional mask preloading is enabled only when sufficient RAM is available.
- Frames are streamed instead of fully buffered to keep RAM usage bounded.
- Tracking reuses the aligned video stream to avoid storing all frames in memory.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import typer
from click import get_current_context
from deep_sort_realtime.deepsort_tracker import DeepSort
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from tqdm import tqdm

import vivadatacreator.tooldata as td
from vivadatacreator.runtime_utils import (
    configure_opencv_threads,
    gather_hardware_profile,
    load_runtime_config,
    resolve_cli_value,
    save_runtime_config,
)

app = typer.Typer()

DEFAULT_CLASSES = [0, 1, 2, 3, 5, 7]  # person, bicycle, car, motorcycle, bus, truck


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    return load_runtime_config(config_path)


def save_config(args: Dict[str, Any], config_path: str = "config.yaml") -> None:
    save_runtime_config(args, config_path)


def create_step4_folders(video_path: str) -> Dict[str, str]:
    """Validate prerequisites from steps 1 & 3 and resolve required folders."""
    if video_path is None:
        raise ValueError("Video path is required for step 4")

    path_obj = Path(video_path)
    root = str(path_obj.parent)
    video_path = str(path_obj.resolve())

    imgs_folder_A = os.path.join(root, "imgsA")
    frames_folder = os.path.join(root, "segmentation")
    if not os.path.exists(imgs_folder_A):
        raise FileNotFoundError(
            f"Steps 1 and 3 must be completed first. Folder '{imgs_folder_A}' not found."
        )
    if not os.path.exists(frames_folder):
        raise FileNotFoundError(
            f"Step 3 must be completed first. Folder '{frames_folder}' not found."
        )

    video_dir = os.path.join(root, "video_alineado.mp4")
    if not os.path.exists(video_dir):
        raise FileNotFoundError(
            f"Aligned video not found at '{video_dir}'. Run Step 1 first."
        )

    return {
        "root": root,
        "video_path": video_path,
        "video_dir": video_dir,
        "imgs_folder_A": imgs_folder_A,
        "frames_folder": frames_folder,
    }


def bytes_to_gb(value: int) -> float:
    return round(value / (1024 ** 3), 2)


def estimate_mask_memory(mask_sample_path: Optional[str], total_frames: int) -> Tuple[int, Optional[Tuple[int, int]]]:
    """Estimate how much RAM would be needed to cache every segmentation mask."""
    if not mask_sample_path or not os.path.exists(mask_sample_path):
        return 0, None

    sample = cv2.imread(mask_sample_path, cv2.IMREAD_GRAYSCALE)
    if sample is None:
        return 0, None

    mask_rgb_bytes = sample.size * 3  # convert to 3 channels
    return mask_rgb_bytes * total_frames, sample.shape


def resolve_device(device_preference: str, hardware: Dict[str, Any]) -> str:
    pref = (device_preference or "auto").lower()
    if pref == "cpu":
        return "cpu"
    if pref == "cuda":
        if hardware["gpu_available"]:
            return "cuda"
        typer.echo("⚠️ CUDA requested but not available. Falling back to CPU.")
        return "cpu"
    return "cuda" if hardware["gpu_available"] else "cpu"


def choose_img_size(requested: Optional[int], auto_tune: bool, device: str, hardware: Dict[str, Any]) -> int:
    if requested and requested > 0:
        return requested
    if not auto_tune:
        return max(256, requested or 640)
    if device == "cuda":
        vram_gb = bytes_to_gb(hardware["total_vram_bytes"])
        if vram_gb >= 20:
            return 896
        if vram_gb >= 16:
            return 768
        if vram_gb >= 10:
            return 704
        return 640
    return 512


def choose_overlap(requested: Optional[float], auto_tune: bool, img_size: int) -> float:
    if requested is not None and requested > 0:
        return requested
    if not auto_tune:
        return requested or 0.2
    if img_size >= 896:
        return 0.1
    if img_size >= 768:
        return 0.12
    if img_size >= 640:
        return 0.15
    return 0.2


def choose_batch_size(requested: Optional[int], auto_tune: bool, device: str, hardware: Dict[str, Any]) -> int:
    if requested and requested > 0:
        return requested
    if not auto_tune:
        return 8
    if device == "cuda":
        vram_gb = bytes_to_gb(hardware["total_vram_bytes"])
        if vram_gb >= 20:
            return 16
        if vram_gb >= 16:
            return 12
        if vram_gb >= 12:
            return 10
        if vram_gb >= 8:
            return 8
        return 6
    cpu_physical = hardware["cpu_physical"]
    return max(2, min(6, cpu_physical // 2))


def should_preload_masks(strategy: str, mask_bytes: int, hardware: Dict[str, Any], ram_budget_fraction: float) -> bool:
    strat = (strategy or "auto").lower()
    if strat == "memory":
        return True
    if strat == "disk":
        return False
    if mask_bytes == 0:
        return False
    budget = hardware["available_ram_bytes"] * ram_budget_fraction
    return mask_bytes <= budget


class MaskProvider:
    """Provides inverse masks either from RAM cache or disk on demand."""

    def __init__(self, frames_folder: str, total_frames: int, preload: bool):
        self.frames_folder = frames_folder
        self.total_frames = total_frames
        self.cache: Optional[Dict[int, np.ndarray]] = None
        if preload:
            self.cache = self._preload_all()

    def _preload_all(self) -> Dict[int, np.ndarray]:
        typer.echo("🧠 Pre-caching masks into RAM...")
        cache: Dict[int, np.ndarray] = {}
        for frame_number in tqdm(
            range(self.total_frames),
            desc="Caching masks",
            unit="mask",
            colour=None,
        ):
            mask_rgb = self._load_mask_rgb(frame_number)
            if mask_rgb is not None:
                cache[frame_number] = mask_rgb
        typer.echo(f"✅ Cached {len(cache)} masks.")
        return cache

    def _load_mask_rgb(self, frame_number: int) -> Optional[np.ndarray]:
        mask_path = os.path.join(self.frames_folder, f"{frame_number}.png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None
        inverse = cv2.bitwise_not(mask)
        return cv2.merge([inverse, inverse, inverse])

    def get(self, frame_number: int) -> Optional[np.ndarray]:
        if self.cache is not None:
            return self.cache.get(frame_number)
        return self._load_mask_rgb(frame_number)


def apply_mask(frame: np.ndarray, mask_rgb: Optional[np.ndarray]) -> np.ndarray:
    if mask_rgb is None:
        return frame
    return cv2.multiply(frame, mask_rgb, scale=1 / 255.0)


def build_runtime_plan(args: Dict[str, Any], hardware: Dict[str, Any], mask_bytes: int) -> Dict[str, Any]:
    device = resolve_device(args.get("device", "auto"), hardware)
    img_size = choose_img_size(args.get("img_size_sahi"), args["auto_tune"], device, hardware)
    overlap = choose_overlap(args.get("overlap_sahi"), args["auto_tune"], img_size)
    batch_size = choose_batch_size(args.get("batch_size"), args["auto_tune"], device, hardware)
    ram_budget = max(0.1, min(0.9, args["ram_budget"]))
    preload_masks = should_preload_masks(args.get("mask_cache", "auto"), mask_bytes, hardware, ram_budget)

    cv_threads = args.get("max_workers")
    if cv_threads is None:
        cv_threads = min(8, max(1, hardware["cpu_physical"]))
    configure_opencv_threads(cv_threads)

    return {
        "device": device,
        "img_size_sahi": img_size,
        "overlap_sahi": overlap,
        "batch_size": batch_size,
        "preload_masks": preload_masks,
        "mask_memory_bytes": mask_bytes,
        "confidence_threshold": args["confidence_threshold"],
        "yolo_weights": args["yolo_weights"],
        "classes": DEFAULT_CLASSES,
    }


def log_runtime_plan(hardware: Dict[str, Any], runtime: Dict[str, Any], total_frames: int, mask_shape: Optional[Tuple[int, int]]) -> None:
    typer.echo("")
    typer.echo("🛠️  Hardware Profile")
    typer.echo(f"   • CPU: {hardware['cpu_physical']} cores ({hardware['cpu_logical']} logical)")
    typer.echo(f"   • RAM: {bytes_to_gb(hardware['total_ram_bytes'])} GB total / "
               f"{bytes_to_gb(hardware['available_ram_bytes'])} GB free")
    if hardware["gpu_available"]:
        typer.echo(f"   • GPU: {hardware['gpu_name']} ({bytes_to_gb(hardware['total_vram_bytes'])} GB VRAM)")
    else:
        typer.echo("   • GPU: Not detected (running on CPU)")

    typer.echo("\n📋 Runtime Plan")
    typer.echo(f"   • Frames to process: {total_frames}")
    if mask_shape:
        typer.echo(f"   • Mask resolution: {mask_shape[1]}x{mask_shape[0]}")
    typer.echo(f"   • Device: {runtime['device']}")
    typer.echo(f"   • SAHI slice: {runtime['img_size_sahi']} px, overlap {runtime['overlap_sahi']:.2f}")
    typer.echo(f"   • Batch size: {runtime['batch_size']} frames")
    typer.echo(f"   • Mask cache: {'RAM' if runtime['preload_masks'] else 'Disk streaming'} "
               f"({bytes_to_gb(runtime['mask_memory_bytes'])} GB if cached)")
    typer.echo("")


def build_detection_model(runtime: Dict[str, Any]) -> AutoDetectionModel:
    return AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=runtime["yolo_weights"],
        confidence_threshold=runtime["confidence_threshold"],
        device=runtime["device"],
    )


def _store_detection(result_store: List[Optional[List[Tuple[List[float], float, int]]]],
                     frame_number: int,
                     detections: List[Tuple[List[float], float, int]]) -> None:
    if frame_number >= len(result_store):
        result_store.extend([None] * (frame_number + 1 - len(result_store)))
    result_store[frame_number] = detections


def _consume_detection_batch(batch: List[Tuple[int, np.ndarray]],
                             detection_model: AutoDetectionModel,
                             runtime: Dict[str, Any],
                             result_store: List[Optional[List[Tuple[List[float], float, int]]]]) -> None:
    for frame_number, frame in batch:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        prediction = get_sliced_prediction(
            image=frame_rgb,
            detection_model=detection_model,
            slice_height=runtime["img_size_sahi"],
            slice_width=runtime["img_size_sahi"],
            overlap_height_ratio=runtime["overlap_sahi"],
            overlap_width_ratio=runtime["overlap_sahi"],
        )
        detections = [
            (list(pred.bbox.to_xywh()), float(pred.score.value), int(pred.category.id))
            for pred in prediction.object_prediction_list
        ]
        _store_detection(result_store, frame_number, detections)


def run_detection(folders: Dict[str, str],
                  total_frames: int,
                  runtime: Dict[str, Any]) -> List[Optional[List[Tuple[List[float], float, int]]]]:
    detection_model = build_detection_model(runtime)
    mask_provider = MaskProvider(folders["frames_folder"], total_frames, runtime["preload_masks"])

    cap = cv2.VideoCapture(folders["video_dir"])
    detections: List[Optional[List[Tuple[List[float], float, int]]]] = []
    batch: List[Tuple[int, np.ndarray]] = []

    typer.echo("🔍 Running adaptive object detection...")
    with tqdm(total=total_frames, desc="Detecting Objects", unit="frame", colour=None) as pbar:
        frame_number = 0
        while frame_number < total_frames:
            ret, frame = cap.read()
            if not ret:
                typer.echo(f"⚠️ Video ended early at frame {frame_number}.")
                break

            mask_rgb = mask_provider.get(frame_number)
            masked_frame = apply_mask(frame, mask_rgb)
            batch.append((frame_number, masked_frame))

            if len(batch) >= runtime["batch_size"]:
                _consume_detection_batch(batch, detection_model, runtime, detections)
                pbar.update(len(batch))
                batch.clear()

            frame_number += 1

        if batch:
            _consume_detection_batch(batch, detection_model, runtime, detections)
            pbar.update(len(batch))
            batch.clear()

    cap.release()
    return detections[:frame_number]


def run_tracking(folders: Dict[str, str],
                 detections: List[Optional[List[Tuple[List[float], float, int]]]],
                 runtime: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    tracker = DeepSort(max_age=30, n_init=3, nn_budget=100, override_track_class=None)
    track_dict: Dict[int, Dict[str, Any]] = {}

    cap = cv2.VideoCapture(folders["video_dir"])
    total_frames = len(detections)

    typer.echo("📊 Updating tracks sequentially...")
    with tqdm(total=total_frames, desc="Tracking Objects", unit="frame", colour=None) as pbar:
        for frame_number in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                typer.echo(f"⚠️ Video ended early at frame {frame_number} while tracking.")
                break

            frame_detections = detections[frame_number] or []
            filtered = [det for det in frame_detections if det[2] in runtime["classes"]]
            tracks = tracker.update_tracks(filtered, frame=frame)
            track_dict = td.update_tracking_info(tracks, frame_number, track_dict)
            pbar.update(1)

    cap.release()
    return track_dict


def process_step_adaptive(folders: Dict[str, str], args: Dict[str, Any]) -> None:
    mask_files = sorted(glob.glob(os.path.join(folders["frames_folder"], "*.png")))
    total_frames = len(mask_files)
    if total_frames == 0:
        raise FileNotFoundError(
            f"No segmentation masks found in '{folders['frames_folder']}'. "
            "Run step 3 before launching step 4."
        )

    mask_bytes, mask_shape = estimate_mask_memory(mask_files[0], total_frames)
    hardware = gather_hardware_profile()
    runtime = build_runtime_plan(args, hardware, mask_bytes)
    log_runtime_plan(hardware, runtime, total_frames, mask_shape)

    detections = run_detection(folders, total_frames, runtime)
    track_dict = run_tracking(folders, detections, runtime)

    csv_path = os.path.join(folders["root"], "track_dic.csv")
    td.save_tracking_info(track_dict, csv_path)

    typer.echo("")
    typer.echo(f"✅ Step 4 completed successfully! {len(track_dict)} objects tracked.")
    typer.echo(f"💾 Tracking data saved to: {csv_path}")


@app.command()
def main(
    root: Optional[str] = typer.Option(None, "--root", help="Path to the aligned video file"),
    fac: Optional[int] = typer.Option(None, "--fac", help="Scaling factor for resizing images"),
    model_cfg: Optional[str] = typer.Option(
        None,
        "--model-cfg",
        help="Path to the SAM2 model configuration file",
    ),
    sam2_chkpt: Optional[str] = typer.Option(
        None,
        "--sam2-chkpt",
        help="Path to SAM2 checkpoint",
    ),
    n_imgs: Optional[int] = typer.Option(
        None,
        "--n-imgs",
        help="Number of images to process per batch (Step 3 compat)",
    ),
    n_obj: Optional[int] = typer.Option(
        None,
        "--n-obj",
        help="Number of objects to process per batch (Step 3 compat)",
    ),
    img_size_sahi: Optional[int] = typer.Option(
        None,
        "--img-size-sahi",
        help="SAHI slice size (set 0 for auto)",
    ),
    overlap_sahi: Optional[float] = typer.Option(
        None,
        "--overlap-sahi",
        help="SAHI overlap ratio (set 0 for auto)",
    ),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", help="Detection batch size (auto if omitted)"),
    max_workers: Optional[int] = typer.Option(None, "--max-workers", help="Threads for OpenCV operations"),
    auto_tune: Optional[bool] = typer.Option(
        None,
        "--auto-tune/--no-auto-tune",
        help="Enable hardware-aware parameter tuning",
    ),
    mask_cache: Optional[str] = typer.Option(
        None,
        "--mask-cache",
        help="Mask caching strategy: auto | memory | disk",
    ),
    ram_budget: Optional[float] = typer.Option(
        None,
        "--ram-budget",
        help="Fraction of available RAM that can be used for mask caching (0.1-0.9)",
    ),
    device: Optional[str] = typer.Option(
        None,
        "--device",
        help="Device preference: auto | cuda | cpu",
    ),
    yolo_weights: Optional[str] = typer.Option(
        None,
        "--yolo-weights",
        help="Path to YOLO weights used by SAHI",
    ),
    confidence_threshold: Optional[float] = typer.Option(
        None,
        "--confidence-threshold",
        help="Detection confidence threshold",
    ),
) -> None:
    """Hardware-aware object detection (YOLO + SAHI) and tracking (DeepSort)."""
    config = load_config()
    root = resolve_cli_value("root", root, config)
    fac = resolve_cli_value("fac", fac, config)
    sam2_chkpt = resolve_cli_value("sam2_chkpt", sam2_chkpt, config)
    model_cfg = resolve_cli_value("model_cfg", model_cfg, config)
    n_imgs = resolve_cli_value("n_imgs", n_imgs, config)
    n_obj = resolve_cli_value("n_obj", n_obj, config)
    img_size_sahi = resolve_cli_value("img_size_sahi", img_size_sahi, config)
    overlap_sahi = resolve_cli_value("overlap_sahi", overlap_sahi, config)
    batch_size = resolve_cli_value("batch_size", batch_size, config)
    max_workers = resolve_cli_value("max_workers", max_workers, config)
    auto_tune = resolve_cli_value("auto_tune", auto_tune, config)
    mask_cache = resolve_cli_value("mask_cache", mask_cache, config)
    ram_budget = resolve_cli_value("ram_budget", ram_budget, config)
    device = resolve_cli_value("device", device, config)
    yolo_weights = resolve_cli_value("yolo_weights", yolo_weights, config)
    confidence_threshold = resolve_cli_value("confidence_threshold", confidence_threshold, config)

    if root is None:
        typer.echo("Error: Video path (--root) is required", err=True)
        raise typer.Exit(code=1)

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
        batch_size=batch_size,
        max_workers=max_workers,
        auto_tune=auto_tune,
        mask_cache=mask_cache,
        ram_budget=ram_budget,
        device=device,
        yolo_weights=yolo_weights,
        confidence_threshold=confidence_threshold,
    )

    try:
        folders = create_step4_folders(root)
        save_config(ctx_params)
        process_step_adaptive(folders, ctx_params)
    except FileNotFoundError as err:
        typer.echo(f"❌ {err}", err=True)
        typer.echo("💡 Make sure to run Steps 1 and 3 first.", err=True)
        raise typer.Exit(code=1)
    except Exception as err:  # noqa: BLE001
        typer.echo(f"❌ Error processing video: {err}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
