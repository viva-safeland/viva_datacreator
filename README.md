# ViVa-DataCreator: An Open-Source Human-in-the-Loop Data Annotation Engine for Semantic Segmentation
[![PyPI version](https://badge.fury.io/py/viva-datacreator.svg)](https://badge.fury.io/py/viva-datacreator)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python versions](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)

<div align="center">
  <img
    src="https://github.com/user-attachments/assets/328d3d54-91af-4334-af12-6155a1e19718"
    width="600"
    alt="ViVa-DataCreator Logo"
  />
</div>

**ViVa-DataCreator** is an open-source tool for creating semantic segmentation datasets by tracking objects of interest from videos. It leverages the *Segment Anything Model 2 (SAM2)* and *You Only Look Once (YOLO)* AI models to perform segmentation and object detection, guiding users through an 8-step process to generate complete datasets ready for model training.

<figure style="text-align: center;">
  <img
    src="https://github.com/user-attachments/assets/5feba7c9-e32a-4f66-97fc-1a22f3f2d0c1"
    alt="ViVa-DataCreator GUI"
    width="800"
  />
  <figcaption>ViVa-DataCreator: Graphical User Interface for Dataset Creation</figcaption>
</figure>

This tool focuses on generating semantic segmentation datasets through object tracking, utilizing SAM 2 to enhance segmentation accuracy.

## Key Features

-   **Video-to-Dataset Conversion:** Transform videos into high-quality segmentation datasets with minimal manual effort.
-   **SAM 2 Integration:** Utilize the latest Segment Anything Model 2 for accurate and interactive segmentation.
-   **8-Step Flexible Process:** A comprehensive workflow that guides you through dataset creation, allowing you to move between steps as needed.
-   **Interactive Refinement:** Manually refine segmentations for precision and quality control.
-   **Object Tracking Integration:** Utilize YOLO and DeepSort for tracking objects of interest across video frames.
-   **Batch Processing:** Efficiently handle large videos through configurable batch processing.
-   **Customizable Classes:** Define and assign custom object classes with unique colors.
-   **Safety-Focused:** Designed for safe and reliable dataset generation without hardware risks.

## Documentation

For detailed usage instructions, examples, and API documentation, please refer to the [ViVa-DataCreator Documentation](https://viva-safeland.github.io/viva_datacreator/).



## Citation

If you use ViVa-DataCreator in your research, please consider adding the following citations:

**ViVa-DataCreator**
```bibtex
@software{soriano2025datacreator,
  author = {Miguel Soriano-García, Diego Mercado-Ravell, Israel Becerra and Julio De La Torre-Vanegas},
  title = {ViVa-DataCreator: An Open-Source Human-in-the-Loop Data Annotation Engine for Semantic Segmentation},
  year = {2025},
  url = {https://github.com/viva-safeland/viva_datacreator}
}
```

**ViVa-SAFELAND Simulator**
```bibtex
@article{soriano2025viva,
  title={ViVa-SAFELAND: a New Freeware for Safe Validation of Vision-based Navigation in Aerial Vehicles},
  author={Miguel S. Soriano-Garcia and Diego A. Mercado-Ravell},
  journal={arXiv preprint arXiv:2503.14719},
  year={2024}
}
```

**Related Application**
```bibtex
@misc{delatorre2025riskaware,
      title={Vision-Based Risk Aware Emergency Landing for UAVs in Complex Urban Environments}, 
      author={Julio de la Torre-Vanegas and Miguel Soriano-Garcia and Israel Becerra and Diego Mercado-Ravell},
      year={2025},
      eprint={2505.20423},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2505.20423}, 
}
```
