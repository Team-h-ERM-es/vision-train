from os import putenv
putenv("HSA_OVERRIDE_GFX_VERSION", "12.0.1")
#putenv("AMD_SERIALIZE_KERNEL", "3") # for more detailed HIP kernel error debugging

import torch
import os
import pathlib
from ultralytics import YOLO


SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
DATA_YAML_PATH = SCRIPT_DIR / 'dataset' / 'data.yaml'

# yolov8n-seg.pt (nano - fastest, lowest accuracy)
# yolov8s-seg.pt (small - good balance)
# yolov8m-seg.pt (medium)
# yolov8l-seg.pt (large)
# yolov8x-seg.pt (extra large - slowest, highest accuracy)
PRETRAINED_MODEL = 'yolov8m-seg.pt'

EPOCHS = 200
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

def train_model():
    """Loads model and starts training."""
    print("-" * 30)
    print("Starting YOLOv8 Instance Segmentation Training")
    print("-" * 30)

    use_gpu, device_type = check_gpu()
    device = 0 if use_gpu else 'cpu'

    if not os.path.exists(DATA_YAML_PATH):
        print(f"ERROR: data.yaml not found at: {DATA_YAML_PATH}")
        print("Please create the data.yaml file and ensure the path is correct.")
        return

    print(f"Using Dataset YAML: {DATA_YAML_PATH}")
    print(f"Using Pretrained Model: {PRETRAINED_MODEL}")
    print(f"Training for {EPOCHS} epochs.")
    print(f"Image Size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Batch Size: {'Auto' if BATCH_SIZE == -1 else BATCH_SIZE}")
    print(f"Device: {device_type} ({device})")
    print(f"Saving results to: runs/segment/{PROJECT_NAME}/{RUN_NAME}")
    print("-" * 30)

    try:
        model = YOLO(PRETRAINED_MODEL)

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
        print(f"\nAn error occurred during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train_model()