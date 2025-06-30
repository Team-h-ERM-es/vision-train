from os import putenv
putenv("HSA_OVERRIDE_GFX_VERSION", "12.0.1")
#putenv("AMD_SERIALIZE_KERNEL", "3") # for more detailed HIP kernel error debugging

import torch
import os
import pathlib
from ultralytics import YOLO
from typing import Tuple


SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
DATA_YAML_PATH = SCRIPT_DIR / 'dataset' / 'data.yaml'

# yolov8n-seg.pt (nano - fastest, lowest accuracy)
# yolov8s-seg.pt (small - good balance)
# yolov8m-seg.pt (medium)
# yolov8l-seg.pt (large)
# yolov8x-seg.pt (extra large - slowest, highest accuracy)
PRETRAINED_MODEL = 'yolov8m-seg.pt'

EPOCHS = 300
IMG_SIZE = 640
BATCH_SIZE = 8
LEARNING_RATE = 0.001
OPTIMIZER = 'AdamW'

PROJECT_NAME = 'pren_robot_segmentation'
RUN_NAME = 'fine_tune_yolov8'


def check_gpu():
    """Checks if GPU is available and prints info for both NVIDIA and AMD GPUs."""
    gpu_available = False
    device_type = 'cpu'

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        current_device_idx = torch.cuda.current_device()
        actual_gpu_name = torch.cuda.get_device_name(current_device_idx)
        
        gpu_available = True
        device_type = 'cuda'

        if "nvidia" in actual_gpu_name.lower():
            print(f"NVIDIA GPU detected: {device_count} device(s) available.")
        elif "amd" in actual_gpu_name.lower() or "radeon" in actual_gpu_name.lower():
            print(f"AMD GPU (via ROCm/HIP) detected: {device_count} device(s) available.")
        else:
            print(f"Unknown GPU (via CUDA-compatible interface) detected: {device_count} device(s) available.")
        
        print(f"Using device {current_device_idx}: {actual_gpu_name}")
    
    elif hasattr(torch, 'has_rocm') and torch.has_rocm:
        device_count = torch.cuda.device_count()
        current_device_idx = torch.cuda.current_device()
        actual_gpu_name = torch.cuda.get_device_name(current_device_idx)
        
        gpu_available = True
        device_type = 'cuda' 

        print(f"AMD GPU (ROCm) detected: {device_count} device(s) available.")
        print(f"Using device {current_device_idx}: {actual_gpu_name}")
    
    if not gpu_available:
        print("WARNING: No GPU detected. Training will run on CPU (very slow).")
    
    return gpu_available, device_type

# ---------------- Dynamic configuration helpers ---------------- #

def infer_dataset_task(data_yaml_path: pathlib.Path) -> str:
    """Infer whether the dataset is for object **detection** or **instance segmentation**.

    Detection label files contain 5 numbers per line (class + bbox xywh),
    whereas segmentation label files contain >5 numbers (class + polygon points).
    This heuristic inspects the first non-empty label file from the *train* split.

    Returns:
        str: "segment" or "detect" (default).  Falls back to "detect" on failure.
    """
    import yaml

    try:
        with data_yaml_path.open("r") as f:
            data_cfg = yaml.safe_load(f)

        # Work out absolute path for the train split directory
        base_path = pathlib.Path(data_cfg.get("path", data_yaml_path.parent))
        train_rel = pathlib.Path(data_cfg.get("train", ""))
        train_dir = (train_rel if train_rel.is_absolute() else base_path / train_rel).resolve()

        labels_dir = train_dir / "labels"
        if not labels_dir.exists():
            return "detect"

        # Pick the first label file that is not empty
        for txt_file in labels_dir.glob("*.txt"):
            try:
                first_line = txt_file.read_text().strip().splitlines()[0]
            except IndexError:
                continue  # empty file – keep searching

            if not first_line:
                continue

            token_count = len(first_line.split())
            # Segmentation annotations have more than 5 tokens per line
            return "segment" if token_count > 5 else "detect"
    except Exception as err:
        # Silent fallback – user will be warned later if mismatch occurs
        print(f"[WARN] Unable to infer dataset task automatically: {err}")

    return "detect"


def prompt_pretrained_model(default_task: str) -> Tuple[str, str]:
    """Interactively ask the user for which pretrained model to use.

    The user can specify:
      • YOLO major version (v8 or v11)
      • Model size (n, s, m, l, x)
    The task suffix ("-seg") is appended automatically for *segment* tasks.

    Args:
        default_task (str): "detect" or "segment" – decides default suffix.

    Returns:
        Tuple[str, str]: filename of the requested pretrained model (e.g. ("yolov8m-seg.pt", "8"))
    """
    task_suffix = "-seg" if default_task == "segment" else ""

    print("\n=== Pre-trained model selection ===")
    version = input("YOLO version [8 | 11] (default 8): ").strip() or "8"
    if version not in {"8", "11"}:
        print("Invalid version specified – falling back to 8.")
        version = "8"

    size = input("Model size [n/s/m/l/x] (default m): ").strip().lower() or "m"
    if size not in {"n", "s", "m", "l", "x"}:
        print("Invalid size specified – falling back to m.")
        size = "m"

    model_filename = f"yolov{version}{size}{task_suffix}.pt"
    print(f"Chosen pretrained model: {model_filename}\n")
    return model_filename, version

def train_model():
    """Loads model and starts training."""
    use_gpu, device_type = check_gpu()
    device = 0 if use_gpu else "cpu"

    if not DATA_YAML_PATH.exists():
        print(f"ERROR: data.yaml not found at: {DATA_YAML_PATH}")
        print("Please create the data.yaml file and ensure the path is correct.")
        return

    # Dynamically decide dataset task & allow user to override model choice
    dataset_task = infer_dataset_task(DATA_YAML_PATH)

    # Ask the user which pre-trained model to use *before* announcing training start
    pretrained_model, yolo_version = prompt_pretrained_model(dataset_task)

    print("-" * 30)
    print(f"Starting YOLOv{yolo_version} {dataset_task.capitalize()} Training")
    print("-" * 30)

    print(f"Detected dataset task: {dataset_task}")

    print(f"Using Dataset YAML: {DATA_YAML_PATH}")
    print(f"Using Pretrained Model: {pretrained_model}")
    print(f"Training for {EPOCHS} epochs.")
    print(f"Image Size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Batch Size: {'Auto' if BATCH_SIZE == -1 else BATCH_SIZE}")
    print(f"Device: {device_type} ({device})")
    save_root = "segment" if dataset_task == "segment" else "detect"
    print(f"Saving results to: runs/{save_root}/{PROJECT_NAME}/{RUN_NAME}")
    print("-" * 30)

    try:
        model = YOLO(pretrained_model)

        results = model.train(
            data=DATA_YAML_PATH,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            device=device,
            project=PROJECT_NAME,
            name=RUN_NAME,
            # --- Optional parameters ---
            # optimizer=OPTIMIZER,  # Specify optimizer
            # lr0=LEARNING_RATE,    # Initial learning rate
            # patience=20,          # Epochs to wait for no improvement before early stopping
            # workers=4,            # Number of dataloader workers (adjust based on CPU/system)
            # exist_ok=False,       # Error if project/name already exists
            # augment=True,         # Use default data augmentation (recommended)
            # verbose=True,         # Print detailed logs
        )

        print("-" * 30)
        print("Training completed successfully!")
        print(f"Results saved in: {results.save_dir}")
        print("Check the directory for weights (best.pt), logs, validation results, etc.")
        print("You can monitor training progress using TensorBoard: tensorboard --logdir runs/segment")
        print("-" * 30)

    except Exception as e:
        print(f"\n[ERROR] An error occurred during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train_model()