# Team hERMes - Efficient Robot Movement

This project utilizes [uv](https://github.com/astral-sh/uv), an extremely fast Python package installer and resolver, for managing Python virtual environments and dependencies.

## Install uv (Recommended Python Environment Manager)

It is recommended to use `uv` for creating the virtual environment and managing packages, as the following setup steps are tailored for it. However, if you are familiar with other Python environment management tools (like the standard `venv` module and `pip`), you can adapt these instructions to your preferred workflow.

If you choose to use `uv` and don't have it installed, you can find installation instructions on the [official `uv` installation guide](https://github.com/astral-sh/uv#installation).


## 1. Create and activate a virtual environment
```bash
uv venv
source .venv/bin/activate
```

## 2. Install PyTorch for your specific hardware
---> IMPORTANT! <---
Install the correct PyTorch version BEFORE installing packages from requirements.txt. The version depends on your GPU (NVIDIA/AMD) or if you are using CPU only.

Go to the [Official PyTorch Website (Get Started Locally)](https://pytorch.org/get-started/locally/) to find the latest, correct installation command for your specific system configuration (OS, Package Manager (pip), Compute Platform (CUDA/ROCm/CPU)).

A. For NVIDIA GPUs, the command will look like this:
```bash
# EXAMPLE ONLY - GET THE CORRECT COMMAND FROM PYTORCH.ORG
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
```

B. For AMD GPUs, the command will look like this:
```bash
# EXAMPLE ONLY - GET THE CORRECT COMMAND FROM PYTORCH.ORG
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/rocm6.3
```


## 3. Install the required packages
```bash
pip install -r requirements.txt
```

## 3.1. Dataset Structure (Not in Git)

This project expects a `dataset` folder in the root directory. This folder is not tracked by Git (as per the `.gitignore` if one exists or by convention for large data) and you will need to create it and populate it with your data.

The structure used in this project within the `dataset` folder is as follows:

```
./dataset/
├── data.yaml                 # Dataset configuration file
├── train/                    # Training data
│   ├── images/               # Training images (e.g., img1.jpg, img2.png, ...)
│   └── labels/               # Training labels (e.g., img1.txt, img2.txt, ...)
├── valid/                    # Validation data
│   ├── images/               # Validation images
│   └── labels/               # Validation labels
└── test/                     # Optional: Test data (if you have it)
    ├── images/               # Test images
    └── labels/               # Test labels

# Other files like *.cache files might also be present.
```

**Explanation:**

*   **`dataset/`**: The main folder containing all your dataset files.
*   **`data.yaml`**: This configuration file is crucial and should be located directly inside the `dataset` folder (i.e., `dataset/data.yaml`). It tells Ultralytics YOLO where to find your training, validation, and (optionally) test data, and also lists your class names. `main.py` refers to this file.
    *   An example `data.yaml` compatible with the structure above would look something like this:
        ```yaml
        # Ensure paths are relative to the directory containing data.yaml, or use absolute paths.
        # For this project structure, if data.yaml is in ./dataset/:
        train: train/images    # Path to training images folder
        val: valid/images      # Path to validation images folder
        # test: test/images    # Optional: Path to test images folder

        # Class names
        names:
          0: class_name_1
          1: class_name_2
          # ... more classes
        ```
    *   YOLO will typically infer the label directory paths by replacing `/images/` with `/labels/` in the paths provided for `train`, `val`, and `test` image directories.

*   **`train/images/`**, **`valid/images/`**, **`test/images/`**: These folders contain your raw image files (e.g., `.jpg`, `.png`).
*   **`train/labels/`**, **`valid/labels/`**, **`test/labels/`**: These folders contain the corresponding annotation files (usually `.txt` files in YOLO format) for each image. Each `.txt` file should have the same name as its corresponding image file.

Ensure your paths within `data.yaml` correctly point to your `train`, `valid`, and (if used) `test` image directories.

## Project Scripts

This project contains several Python scripts for training and testing YOLOv8 models:

*   `main.py`: 
    *   This is the primary script for training the YOLOv8 instance segmentation model.
    *   It handles GPU detection (NVIDIA CUDA and AMD ROCm/HIP), loads a pre-trained model, and starts the fine-tuning process based on the configuration variables set within the script (e.g., `EPOCHS`, `IMG_SIZE`, `PRETRAINED_MODEL`, `PROJECT_NAME`, `RUN_NAME`).
    *   **Important for AMD GPUs**: This script includes specific environment variable settings (`HSA_OVERRIDE_GFX_VERSION` and `AMD_SERIALIZE_KERNEL`) to aid compatibility and debugging on AMD ROCm platforms. Ensure `HSA_OVERRIDE_GFX_VERSION` is set to match your GPU's architecture (e.g., "12.0.1" for gfx1201) by checking `rocminfo | grep gfx`.

*   `test.py`:
    *   A simple script to load a trained YOLO model and run inference on a webcam feed.
    *   It displays the annotated video stream in real-time.
    *   **Note**: This script may need its model path updated to point to the specific weights generated by `main.py` (e.g., `"pren_robot_segmentation/fine_tune_yolov8s3/weights/best.pt"` or similar, found in `runs/segment/PROJECT_NAME/RUN_NAME/weights/best.pt`). It also does not currently include the AMD-specific environment variables from `main.py`, which might be necessary for it to run correctly on an AMD GPU that requires them.

*   `test-pytorch.py`:
    *   A diagnostic script to verify the PyTorch installation and check for GPU availability (CUDA/ROCm and Apple MPS).
    *   It prints basic information about the PyTorch version, Python version, and detected GPU(s).
    *   This script provides a general check but does not include the AMD-specific environment variables found in `main.py`.
