"""
Dataset Preparation Script

Downloads and prepares device damage datasets for training.
Supports:
- Kaggle datasets for mobile screen cracks
- Custom image directories
- Automatic YOLO-format label generation from annotations
"""

import os
import shutil
import random
from pathlib import Path

import numpy as np
from PIL import Image


CLASSES = ["cracked_screen", "battery_swelling", "charging_port_damage"]


def prepare_yolo_dataset(
    source_dir: str,
    output_dir: str = "datasets",
    train_ratio: float = 0.8,
):
    """
    Prepare dataset in YOLO format from classified image directories.

    Expected source structure:
        source_dir/
        ├── cracked_screen/
        │   ├── img1.jpg
        │   └── img1.txt  (optional YOLO label)
        ├── battery_swelling/
        └── charging_port_damage/

    If no .txt label files exist, generates full-image bounding boxes.
    """
    images_train = os.path.join(output_dir, "images", "train")
    images_val = os.path.join(output_dir, "images", "val")
    labels_train = os.path.join(output_dir, "labels", "train")
    labels_val = os.path.join(output_dir, "labels", "val")

    for d in [images_train, images_val, labels_train, labels_val]:
        os.makedirs(d, exist_ok=True)

    all_samples = []

    for class_idx, class_name in enumerate(CLASSES):
        class_dir = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"Warning: {class_dir} not found, skipping.")
            continue

        for fname in os.listdir(class_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(class_dir, fname)
            label_path = os.path.splitext(img_path)[0] + ".txt"

            # Check for existing YOLO label
            if os.path.exists(label_path):
                with open(label_path) as f:
                    label_content = f.read().strip()
            else:
                # Generate a full-image bounding box label
                label_content = f"{class_idx} 0.5 0.5 0.9 0.9"

            all_samples.append((img_path, label_content, class_name))

    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * train_ratio)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]

    for split_name, samples, img_dir, lbl_dir in [
        ("train", train_samples, images_train, labels_train),
        ("val", val_samples, images_val, labels_val),
    ]:
        for i, (img_path, label, class_name) in enumerate(samples):
            ext = Path(img_path).suffix
            new_name = f"{class_name}_{i:04d}"

            shutil.copy2(img_path, os.path.join(img_dir, new_name + ext))
            with open(os.path.join(lbl_dir, new_name + ".txt"), "w") as f:
                f.write(label)

        print(f"{split_name}: {len(samples)} samples")

    print(f"\nDataset prepared in {output_dir}/")
    print(f"Total: {len(all_samples)} images ({len(train_samples)} train, {len(val_samples)} val)")


def prepare_crop_dataset(
    source_dir: str,
    output_dir: str = "datasets/crops",
):
    """
    Prepare cropped dataset for LSTM classifier training.
    Simply organizes images into class subdirectories.
    """
    for class_name in CLASSES:
        src = os.path.join(source_dir, class_name)
        dst = os.path.join(output_dir, class_name)
        os.makedirs(dst, exist_ok=True)

        if not os.path.isdir(src):
            continue

        for fname in os.listdir(src):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                shutil.copy2(os.path.join(src, fname), os.path.join(dst, fname))

    print(f"Crop dataset prepared in {output_dir}/")


def generate_synthetic_data(output_dir: str = "datasets/synthetic", num_per_class: int = 100):
    """Generate synthetic placeholder images for testing the pipeline."""
    for class_idx, class_name in enumerate(CLASSES):
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        for i in range(num_per_class):
            # Create a simple synthetic image with class-specific color tint
            img = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)

            # Add class-specific visual pattern
            if class_name == "cracked_screen":
                # Draw random lines to simulate cracks
                for _ in range(5):
                    y = random.randint(0, 479)
                    img[max(0, y - 1) : y + 2, :, :] = [200, 50, 50]
            elif class_name == "battery_swelling":
                # Add a bulge-like bright region
                cy, cx = 240, 320
                for y in range(480):
                    for x in range(640):
                        dist = ((y - cy) ** 2 + (x - cx) ** 2) ** 0.5
                        if dist < 100:
                            img[y, x] = np.clip(img[y, x] + 80, 0, 255)
            elif class_name == "charging_port_damage":
                # Darken bottom region
                img[400:, 250:390, :] = np.clip(img[400:, 250:390, :] - 80, 0, 255)

            Image.fromarray(img).save(os.path.join(class_dir, f"synth_{i:04d}.jpg"))

            # YOLO label
            with open(os.path.join(class_dir, f"synth_{i:04d}.txt"), "w") as f:
                f.write(f"{class_idx} 0.5 0.5 0.8 0.8")

    print(f"Synthetic dataset generated in {output_dir}/ ({num_per_class} per class)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare datasets for RepairLens")
    parser.add_argument("--source", help="Source image directory")
    parser.add_argument("--output", default="datasets")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic data")
    parser.add_argument("--num-synthetic", type=int, default=100)
    parser.add_argument("--crops", action="store_true", help="Prepare crop dataset for LSTM")
    args = parser.parse_args()

    if args.synthetic:
        generate_synthetic_data(
            os.path.join(args.output, "synthetic"), args.num_synthetic
        )
    elif args.crops and args.source:
        prepare_crop_dataset(args.source, os.path.join(args.output, "crops"))
    elif args.source:
        prepare_yolo_dataset(args.source, args.output)
    else:
        print("Generating synthetic data for demo...")
        generate_synthetic_data(os.path.join(args.output, "synthetic"), 100)
        prepare_yolo_dataset(os.path.join(args.output, "synthetic"), args.output)
        prepare_crop_dataset(os.path.join(args.output, "synthetic"), os.path.join(args.output, "crops"))
