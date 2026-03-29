"""
Export trained models to SnapML-compatible formats.

SnapML in Lens Studio supports ONNX models. This script:
1. Exports YOLOv8 to ONNX with optimizations for mobile
2. Exports LSTM classifier to ONNX
3. Validates exported models
"""

import os
import argparse

import torch
import numpy as np


def export_yolo_to_onnx(
    model_path: str = "runs/detect/repairlens_damage/weights/best.pt",
    output_path: str = "../lens-studio/models/damage_detector.onnx",
    img_size: int = 640,
):
    """Export YOLOv8 model to ONNX for SnapML."""
    from ultralytics import YOLO

    model = YOLO(model_path)

    model.export(
        format="onnx",
        imgsz=img_size,
        simplify=True,
        opset=12,  # SnapML compatible opset
        half=False,  # Full precision for compatibility
        dynamic=False,  # Fixed input size for SnapML
    )

    # Move to lens-studio models directory
    onnx_path = model_path.replace(".pt", ".onnx")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.rename(onnx_path, output_path)
    print(f"YOLOv8 exported to {output_path}")


def export_lstm_to_onnx(
    model_path: str = "../backend/weights/lstm_classifier.pt",
    output_path: str = "../lens-studio/models/lstm_classifier.onnx",
):
    """Export LSTM classifier to ONNX."""
    from train_lstm import LSTMClassifierModel

    model = LSTMClassifierModel(num_classes=3)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    dummy_input = torch.randn(1, 3, 128, 128)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input_image"],
        output_names=["class_logits"],
        dynamic_axes=None,
    )
    print(f"LSTM classifier exported to {output_path}")


def validate_onnx(model_path: str):
    """Validate an exported ONNX model."""
    import onnx
    import onnxruntime as ort

    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    print(f"ONNX model valid: {model_path}")

    session = ort.InferenceSession(model_path)
    input_info = session.get_inputs()[0]
    print(f"  Input: {input_info.name}, shape={input_info.shape}, type={input_info.type}")

    for output in session.get_outputs():
        print(f"  Output: {output.name}, shape={output.shape}, type={output.type}")

    # Test inference
    dummy = np.random.randn(*input_info.shape).astype(np.float32)
    results = session.run(None, {input_info.name: dummy})
    print(f"  Test inference OK, output shapes: {[r.shape for r in results]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export models for SnapML")
    parser.add_argument("--yolo", action="store_true", help="Export YOLOv8")
    parser.add_argument("--lstm", action="store_true", help="Export LSTM classifier")
    parser.add_argument("--validate", type=str, help="Validate an ONNX model file")
    parser.add_argument("--all", action="store_true", help="Export all models")
    args = parser.parse_args()

    if args.validate:
        validate_onnx(args.validate)
    elif args.all or (not args.yolo and not args.lstm):
        export_yolo_to_onnx()
        export_lstm_to_onnx()
    else:
        if args.yolo:
            export_yolo_to_onnx()
        if args.lstm:
            export_lstm_to_onnx()
