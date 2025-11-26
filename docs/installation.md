# Installation

This guide provides instructions to install ViVa-SAFELAND.

!!! info "System Requirements"
    The code automatically creates a Python 3.12 virtual environment using `uv`, even if a different Python version is installed on your system. 

## 1. Setting `UV`, the Python project manager

To facilitate the creation of virtual environments and manage Python packages and their dependencies we use a state of the art framework [uv](https://docs.astral.sh/uv/), its installation is straightforward and can be done via the following command:

=== "macOS/Linux"
    Using `curl`
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
    Using `wget`
    ```bash
    wget -qO- https://astral.sh/uv/install.sh | sh
    ```
=== "Windows"
    Use `irm` to download the script and execute it with `iex`:
    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

## 2. Install ViVa-DataCreator

Choose one of the following installation methods:

=== "From PyPI"
    !!! abstract "Recommended for most users"
        Install the latest stable release from the Python Package Index (PyPI).

    ```bash
    uv venv --python 3.12
    uv pip install viva-datacreator --upgrade
    ```

=== "From Source"
    !!! abstract "Recommended for developers"
        Install the latest development version directly from the GitHub repository.

    ```bash
    git clone https://github.com/viva-safeland/viva_datacreator.git
    cd viva_datacreator
    uv sync
    ```

!!! tip "Automatic Setup"
    The application will automatically download the required SAM2 checkpoints on first launch if they are not already available. No manual download is required.

## 4. Video Requirements

To use ViVa-SAFELAND, you need video files for processing. The application accepts various video formats. Below the recommended specifications are listed:

Video specifications:

- **Format:** MP4, AVI, MOV, MKV, WMV, FLV, WebM
- **Resolution:** 1080p or higher recommended for better segmentation quality
- **Frame Rate:** 30 FPS recommended
- **Content:** Videos containing objects you want to segment (people, vehicles, animals, etc.)