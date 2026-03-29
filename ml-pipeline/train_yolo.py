"""
YOLOv8 Training Script for Device Damage Detection

Trains a YOLOv8n (nano) model on device damage images to detect:
- cracked_screen (class 0)
- battery_swelling (class 1)
- charging_port_damage (class 2)

Dataset structure expected:
    datasets/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/

Labels in YOLO format: <class_id> <x_center> <y_center> <width> <height>
"""

import os
from pathlib import Path


def create_dataset_yaml(data_dir: str = "datasets") -> str:
    """Create the dataset configuration YAML for YOLOv8."""
    yaml_content = f"""
path: {os.path.abspath(data_dir)}
train: images/train
val: images/val

nc: 3
names:
  0: cracked_screen
  1: battery_swelling
  2: charging_port_damage
"""
    yaml_path = os.path.join(data_dir, "damage_dataset.yaml")
    os.makedirs(data_dir, exist_ok=True)
    with open(yaml_path, "w") as f:
        f.write(yaml_content.strip())
    print(f"Dataset YAML created at {yaml_path}")
    return yaml_path


def train(
    data_yaml: str = "datasets/damage_dataset.yaml",
    epochs: int = 50,
    img_size: int = 640,
    batch_size: int = 16,
    model_variant: str = "yolov8n.pt",
):
    """
    Train YOLOv8 on device damage dataset.

    Args:
        data_yaml: Path to dataset YAML configuration
        epochs: Number of training epochs
        img_size: Input image size
        batch_size: Training batch size
        model_variant: YOLOv8 pretrained model to fine-tune
    """
    from ultralytics import YOLO

    # Load pretrained YOLOv8 nano (smallest, fastest for mobile/AR)
    model = YOLO(model_variant)

    # Train
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        name="repairlens_damage",
        project="runs/detect",
        # Optimizations for mobile deployment
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        augment=True,
        mosaic=1.0,
        mixup=0.1,
        # Save best model
        save=True,
        save_period=10,
    )

    print(f"\nTraining complete. Results saved to {results.save_dir}")

    # Export to ONNX for SnapML compatibility
    best_model_path = os.path.join(results.save_dir, "weights", "best.pt")
    export_model = YOLO(best_model_path)
    export_model.export(format="onnx", imgsz=img_size, simplify=True)
    print("Model exported to ONNX format for SnapML integration.")

    return results


def validate(model_path: str = "runs/detect/repairlens_damage/weights/best.pt"):
    """Run validation on the trained model."""
    from ultralytics import YOLO

    model = YOLO(model_path)
    metrics = model.val()

    print(f"\nmAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

    for i, name in enumerate(["cracked_screen", "battery_swelling", "charging_port_damage"]):
        print(f"  {name}: AP50={metrics.box.ap50[i]:.4f}")

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train YOLOv8 damage detector")
    parser.add_argument("--data-dir", default="datasets", help="Dataset directory")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    if args.validate_only:
        validate()
    else:
        yaml_path = create_dataset_yaml(args.data_dir)
        train(
            data_yaml=yaml_path,
            epochs=args.epochs,
            img_size=args.img_size,
            batch_size=args.batch_size,
        )
