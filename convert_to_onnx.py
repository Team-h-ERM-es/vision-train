import argparse
import pathlib
import sys

import torch
from ultralytics import YOLO


def select_device(prefer_gpu: bool = True) -> str:
    """Return the appropriate device string for Ultralytics YOLO export.

    Parameters
    ----------
    prefer_gpu : bool, optional
        If ``True`` and a CUDA-capable GPU is available, use the first GPU (``"0"``).
        Otherwise, fall back to ``"cpu"``. Defaults to ``True``.
    """
    if prefer_gpu and torch.cuda.is_available():
        return "0"  # first GPU
    return "cpu"


def convert_to_onnx(
    weights_path: pathlib.Path,
    imgsz: int = 640,
    dynamic: bool = False,
    simplify: bool = False,
    prefer_gpu: bool = True,
) -> pathlib.Path:
    """Convert a YOLOv8 PyTorch weight file (.pt) to ONNX.

    Parameters
    ----------
    weights_path : pathlib.Path
        Path to the trained ``.pt`` weights file.
    imgsz : int, optional
        Input image size (square). Defaults to ``640``.
    dynamic : bool, optional
        Export ONNX with dynamic input shapes. Defaults to ``False``.
    simplify : bool, optional
        Run ONNX simplification after export (requires onnx-simplifier). Defaults to ``False``.
    prefer_gpu : bool, optional
        Attempt to use GPU for export if available. Defaults to ``True``.

    Returns
    -------
    pathlib.Path
        Path to the exported ``.onnx`` model.
    """
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    device = select_device(prefer_gpu)
    print(f"Using device: {device}")

    model = YOLO(str(weights_path))
    print(f"Loaded model from {weights_path}")

    print("Starting ONNX export …")
    export_results = model.export(
        format="onnx",
        imgsz=imgsz,
        device=device,
        dynamic=dynamic,
        simplify=simplify,
    )

    if isinstance(export_results, (list, tuple)) and export_results:
        # ultralytics <= 8.1.x returns list/tuple of paths
        onnx_path = pathlib.Path(export_results[0])
    else:
        # ultralytics >= 8.2 returns an object with 'file' attribute
        onnx_path = pathlib.Path(getattr(export_results, "file", ""))

    if not onnx_path or not onnx_path.exists():
        raise RuntimeError("ONNX export failed – output file not found.")

    print(f"ONNX model saved to: {onnx_path}")
    return onnx_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a trained YOLOv8 .pt model to ONNX format."
    )
    parser.add_argument(
        "weights",
        type=pathlib.Path,
        help="Path to the YOLOv8 .pt weights file (e.g., runs/segment/exp/weights/best.pt)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size (square). Must match training size. Default: 640",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Enable dynamic input shapes in ONNX export.",
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Simplify the exported ONNX graph using onnx-simplifier.",
    )
    parser.add_argument(
        "--cpu",
        dest="prefer_gpu",
        action="store_false",
        help="Force export on CPU even if a GPU is available.",
    )

    args = parser.parse_args()

    try:
        convert_to_onnx(
            weights_path=args.weights,
            imgsz=args.imgsz,
            dynamic=args.dynamic,
            simplify=args.simplify,
            prefer_gpu=args.prefer_gpu,
        )
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1) 