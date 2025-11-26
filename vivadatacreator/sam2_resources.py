from __future__ import annotations

from functools import lru_cache
from importlib.util import find_spec
from pathlib import Path
from typing import Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

SAM2_MODEL_PRESETS: Dict[str, Tuple[str, str]] = {
    "Hiera Tiny": ("sam2.1/sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt"),
    "Hiera Small": ("sam2.1/sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt"),
    "Hiera Base+": ("sam2.1/sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt"),
    "Hiera Large": ("sam2.1/sam2.1_hiera_l.yaml", "sam2.1_hiera_large.pt"),
}
DEFAULT_MODEL_KEY = "Hiera Large"


def _locate_sam2_package() -> Optional[Path]:
    """Return the installation directory for the sam2 package if available."""
    spec = find_spec("sam2")
    if spec is None:
        return None
    if spec.submodule_search_locations:
        return Path(spec.submodule_search_locations[0]).resolve()
    if spec.origin:
        return Path(spec.origin).resolve().parent
    return None


def resolve_config_path(relative_config: str) -> Optional[Path]:
    """Return the absolute path to a SAM2 config file inside the installed package."""
    base = _locate_sam2_package()
    if base is None:
        return None
    candidate = base / "configs" / relative_config
    return candidate if candidate.exists() else None


@lru_cache(maxsize=1)
def build_model_map() -> Dict[str, Tuple[str, str]]:
    """Map friendly preset names to (config_path, checkpoint_path)."""
    entries: Dict[str, Tuple[str, str]] = {}
    for label, (cfg_rel, ckpt_file) in SAM2_MODEL_PRESETS.items():
        cfg_path = resolve_config_path(cfg_rel)
        entries[label] = (
            str(cfg_path) if cfg_path else str(Path("sam2") / "configs" / cfg_rel),
            str(CHECKPOINTS_DIR / ckpt_file),
        )
    return entries


def default_model_cfg() -> str:
    """Return the default SAM2 model configuration path."""
    return build_model_map()[DEFAULT_MODEL_KEY][0]


def default_checkpoint_path() -> str:
    """Return the default SAM2 checkpoint path."""
    return build_model_map()[DEFAULT_MODEL_KEY][1]


def config_to_hydra_name(config_path_or_name: str) -> str:
    """Convert an absolute config path into the Hydra-relative string SAM2 expects."""
    if not config_path_or_name:
        return config_path_or_name

    value_path = Path(config_path_or_name).expanduser()
    base = _locate_sam2_package()

    if base and value_path.is_absolute():
        try:
            relative = value_path.resolve().relative_to(base.resolve())
            return relative.as_posix()
        except ValueError:
            pass

    return config_path_or_name.replace("\\", "/")
