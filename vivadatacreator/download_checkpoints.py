
from __future__ import annotations

import argparse
from pathlib import Path
import urllib.request
from typing import Iterable, Mapping

from tqdm import tqdm

from vivadatacreator.sam2_resources import CHECKPOINTS_DIR

CHECKPOINT_URLS: Mapping[str, str] = {
    "sam2.1_hiera_tiny.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
    "sam2.1_hiera_small.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
    "sam2.1_hiera_base_plus.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
    "sam2.1_hiera_large.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
}


class DownloadProgressBar(tqdm):
    """Tqdm progress bar that works with urllib callbacks."""

    def update_to(self, blocks: int = 1, block_size: int = 1, total_size: int | None = None) -> None:
        if total_size is not None:
            self.total = total_size
        self.update(blocks * block_size - self.n)


def download_checkpoint(filename: str, url: str, download_dir: Path) -> bool:
    """Download a checkpoint file if it does not already exist."""
    download_dir.mkdir(parents=True, exist_ok=True)
    filepath = download_dir / filename
    if filepath.exists():
        print(f"Checkpoint {filename} already exists. Skipping download.")
        return False

    print(f"Downloading {filename} to {filepath}...")
    with DownloadProgressBar(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=filename) as progress:
        urllib.request.urlretrieve(url, filename=filepath, reporthook=progress.update_to)
    print(f"Finished downloading {filename}.")
    return True


def ensure_checkpoints(download_dir: Path | None = None, filenames: Iterable[str] | None = None) -> None:
    """Ensure all required checkpoints exist, downloading any missing files."""
    target_dir = Path(download_dir) if download_dir else CHECKPOINTS_DIR
    requested = list(filenames) if filenames else list(CHECKPOINT_URLS.keys())
    for filename in requested:
        url = CHECKPOINT_URLS[filename]
        download_checkpoint(filename, url, target_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the SAM2 checkpoints required by Segmented Creator.")
    parser.add_argument(
        "--dir",
        type=Path,
        default=CHECKPOINTS_DIR,
        help="Directory where checkpoints will be stored (default: %(default)s)",
    )
    args = parser.parse_args()
    ensure_checkpoints(args.dir)


if __name__ == "__main__":
    main()
