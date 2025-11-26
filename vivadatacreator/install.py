#!/usr/bin/env python3
"""
Bootstrap helper for Segmented Creator.

SAM2 is installed via the project dependencies (see pyproject.toml). This
script verifies the dependency is present and triggers the checkpoint
downloader so local runs are ready to go.
"""

from __future__ import annotations

import subprocess
import sys

from vivadatacreator.download_checkpoints import ensure_checkpoints

SAM2_SPEC = "sam2 @ git+https://github.com/facebookresearch/sam2.git"


def ensure_sam2_installed() -> None:
    """Install SAM2 from Git if it is not already available."""
    try:
        import sam2  # type: ignore  # noqa: F401
        print("SAM2 package already available.")
        return
    except ImportError:
        print("SAM2 package not found. Installing from Git...")

    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", SAM2_SPEC],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        print("Failed to install SAM2. Please run 'uv pip install -e .' manually.")
        sys.exit(1)
    print("SAM2 package installed successfully.")


def main() -> None:
    ensure_sam2_installed()
    ensure_checkpoints()
    print("Installation complete...")


if __name__ == "__main__":
    main()
