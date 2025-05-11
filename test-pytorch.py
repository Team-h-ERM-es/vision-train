import torch
import platform

print(f"--- PyTorch Installation Verification ---")
print(f"PyTorch Version: {torch.__version__}")
print(f"Python Version: {platform.python_version()}")

gpu_available = False
mps_available = False

if torch.cuda.is_available():
    try:
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        print(f"\nCUDA/ROCm available: True")
        print(f"GPU Device Count: {device_count}")
        print(f"Current Device ID: {current_device}")
        print(f"Device Name: {device_name}")
        if "nvidia" in device_name.lower():
            print("Type: NVIDIA GPU (CUDA)")
        elif "amd" in device_name.lower() or "radeon" in device_name.lower():
            print("Type: AMD GPU (ROCm / HIP)")
        else:
            print("Type: Unknown CUDA-compatible")
        gpu_available = True
    except Exception as e:
        print(f"\nCUDA/ROCm detected but error getting details: {e}")
        gpu_available = True

if not gpu_available and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
     print(f"\nApple Silicon GPU (MPS) available: True")
     mps_available = True

if not gpu_available and not mps_available:
    print("\nGPU not available or not detected by PyTorch.")
    print("Training will use CPU (expect significantly slower performance).")

print("-----------------------------------------")