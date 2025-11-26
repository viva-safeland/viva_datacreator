from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import psutil
import shutil
import torch
import yaml

from vivadatacreator.sam2_resources import default_checkpoint_path, default_model_cfg

OPTIMIZED_DEFAULTS: Dict[str, Any] = {
    "fac": 3,
    "n_imgs": 200,
    "n_obj": 20,
    "img_size_sahi": 512,
    "overlap_sahi": 0.2,
    "sam2_chkpt": default_checkpoint_path(),
    "model_cfg": default_model_cfg(),
    "auto_tune": True,
    "mask_cache": "auto",
    "ram_budget": 0.6,
    "device": "auto",
    "yolo_weights": "yolo11x.pt",
    "confidence_threshold": 0.35,
}


def load_runtime_config(config_path: str = "config.yaml", extra_defaults: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load the persisted CLI configuration and overlay optimized defaults."""
    config: Dict[str, Any] = dict(OPTIMIZED_DEFAULTS)
    if extra_defaults:
        config.update({k: v for k, v in extra_defaults.items() if v is not None})
    if os.path.exists(config_path):
        with open(config_path, "r") as file:
            file_data = yaml.safe_load(file) or {}
        config.update({k: v for k, v in file_data.items() if v is not None})
    return config


def save_runtime_config(values: Dict[str, Any], config_path: str = "config.yaml") -> None:
    """Persist CLI arguments, ignoring None entries."""
    payload = load_runtime_config(config_path, extra_defaults={})
    payload.update({k: v for k, v in values.items() if v is not None})
    with open(config_path, "w") as file:
        yaml.safe_dump(payload, file)


def _count_cores(logical: bool = True) -> int:
    value = psutil.cpu_count(logical=logical)
    if value is None:
        value = psutil.cpu_count(logical=True) or 4
    return max(1, int(value))


def gather_hardware_profile() -> Dict[str, Any]:
    """Collect CPU, RAM and GPU statistics for adaptive planning."""
    vm = psutil.virtual_memory()
    profile: Dict[str, Any] = {
        "total_ram_bytes": vm.total,
        "available_ram_bytes": vm.available,
        "cpu_physical": _count_cores(logical=False),
        "cpu_logical": _count_cores(logical=True),
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_name": None,
        "total_vram_bytes": 0,
        "free_vram_bytes": 0,
    }

    if profile["gpu_available"]:
        props = torch.cuda.get_device_properties(0)
        profile["gpu_name"] = props.name
        profile["total_vram_bytes"] = props.total_memory
        try:
            free, _ = torch.cuda.mem_get_info()
            profile["free_vram_bytes"] = free
        except RuntimeError:
            profile["free_vram_bytes"] = props.total_memory

    return profile


def recommended_workers(requested: Optional[int], hardware: Dict[str, Any], upper_cap: int = 12) -> int:
    """Resolve worker/thread count honoring hardware limits."""
    if requested is not None:
        return max(1, min(upper_cap, int(requested)))
    return max(1, min(upper_cap, hardware["cpu_physical"]))


def configure_opencv_threads(worker_count: int) -> None:
    import cv2  # Local import to prevent optional dependency at import time

    cv2.setNumThreads(int(worker_count))


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def ensure_folder(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def ensure_parent_folder(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def batched(iterable: Iterable[Any], batch_size: int):
    """Yield iterable items in fixed-size batches."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def resolve_cli_value(key: str, cli_value: Any, config: Dict[str, Any]) -> Any:
    """Resolve a parameter using CLI > config > optimized defaults priority."""
    if cli_value is not None:
        return cli_value
    if key in config and config[key] is not None:
        return config[key]
    return OPTIMIZED_DEFAULTS.get(key)


def copy_files_concurrently(src_dir: str, dst_dir: str, filenames: Iterable[str], workers: int) -> None:
    ensure_folder(dst_dir)
    safe_workers = max(1, workers)
    with ThreadPoolExecutor(max_workers=safe_workers) as pool:
        list(pool.map(lambda name: shutil.copy2(os.path.join(src_dir, name), os.path.join(dst_dir, name)), filenames))
