# ViVa-SAFELAND: Dataset Creation Tool

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python versions](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)

<div align="center">
  <img
    src="https://github.com/user-attachments/assets/328d3d54-91af-4334-af12-6155a1e19718"
    width="800"
    alt="ViVa-SAFELAND Logo"
  />
</div>

**ViVa-SAFELAND** is an open-source tool for creating semantic segmentation datasets by tracking objects of interest from videos. It leverages the **SAM 2 (Segment Anything Model 2)** and **YOLO** AI models to perform segmentation and object detection, guiding users through an 8-step process to generate complete datasets ready for model training.

<figure style="text-align: center;">
  <img
    src="https://github.com/user-attachments/assets/5feba7c9-e32a-4f66-97fc-1a22f3f2d0c1"
    alt="ViVa-SAFELAND GUI"
    width="800"
  />
  <figcaption>ViVa-SAFELAND: Graphical User Interface for Dataset Creation</figcaption>
</figure>

This tool focuses on generating semantic segmentation datasets through object tracking, utilizing SAM 2 to enhance segmentation accuracy.

## Key Features

-   **Video-to-Dataset Conversion:** Transform videos into high-quality segmentation datasets with minimal manual effort.
-   **SAM 2 Integration:** Utilize the latest Segment Anything Model 2 for accurate and interactive segmentation.
-   **8-Step Guided Process:** Step-by-step workflow ensuring comprehensive dataset creation from frame extraction to final composition.
-   **Interactive Refinement:** Manually refine segmentations for precision and quality control.
-   **Object Tracking Integration:** Utilize YOLO and DeepSort for tracking objects of interest across video frames.
-   **Batch Processing:** Efficiently handle large videos through configurable batch processing.
-   **Customizable Classes:** Define and assign custom object classes with unique colors.
-   **Safety-Focused:** Designed for safe and reliable dataset generation without hardware risks.

## Documentation

For detailed usage instructions, examples, and API documentation, please refer to the [ViVa-DataCreator Documentation](https://viva-safeland.github.io/viva_datacreator/).



## Citation

If you use ViVa-SAFELAND in your research, please cite our work:

```bibtex
@software{soriano_garcia_viva_safeland_2025,
  author = {Miguel Soriano-García, Diego Mercado-Ravell, Israel Becerra and Julio De La Torre-Vanegas},
  title = {ViVa-DataCreator: Dataset Creation Tool},
  year = {2025},
  url = {https://github.com/viva-safeland/viva_datacreator}
}
```
