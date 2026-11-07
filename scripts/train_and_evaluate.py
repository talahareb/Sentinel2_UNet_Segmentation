#!/usr/bin/env python3
"""Train the original U-Net architecture, evaluate masks and optionally convert to TFLite."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentinel_unet.data import load_dataset
from sentinel_unet.metrics import evaluate_masks
from sentinel_unet.visualization import save_examples, save_history


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate Sentinel-2 U-Net binary segmentation")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "sentinel2.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--split-mode", choices=["three-way", "original"], default="three-way")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-tflite", action="store_true")
    args = parser.parse_args()

    import numpy as np
    import tensorflow as tf
    from sentinel_unet.model import build_unet
    from sentinel_unet.tflite_utils import convert_dynamic_range, get_tflite_tensor_types, predict_tflite

    tf.keras.utils.set_random_seed(args.seed)
    data = load_dataset(args.data, mode=args.split_mode)
    args.output.mkdir(parents=True, exist_ok=True)
    model = build_unet(tuple(data.x_train.shape[1:]))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    print(f"Train/validation/test: {len(data.x_train)}/{len(data.x_val)}/{len(data.x_test)}")
    if args.split_mode == "original":
        print("NOTE: The original lab reuses the same 10 images as validation AND test; not an independent test.")
    else:
        print("NOTE: The source dataset's geographic independence is not documented.")
    fit = model.fit(data.x_train, data.y_train, validation_data=(data.x_val, data.y_val), epochs=args.epochs, batch_size=args.batch_size)
    model.save(args.output / "unet.keras")
    with (args.output / "history.csv").open("w", newline="") as file:
        keys = list(fit.history)
        writer = csv.DictWriter(file, fieldnames=["epoch", *keys])
        writer.writeheader()
        for i in range(len(fit.history[keys[0]])):
            writer.writerow({"epoch": i+1, **{key: fit.history[key][i] for key in keys}})
    save_history(fit.history, args.output / "training_history.png")
    predictions = model.predict(data.x_test, verbose=0)
    metrics = {"experiment": {"split_mode": args.split_mode, "train_count": len(data.x_train), "val_count": len(data.x_val), "test_count": len(data.x_test), "seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size}, "keras_evaluation_without_padding": evaluate_masks(data.y_test, predictions, pad=data.pad), "keras_evaluation_with_padding": evaluate_masks(data.y_test, predictions, pad=0)}
    save_examples(data.x_test, data.y_test, predictions, args.output / "segmentation_examples.png", pad=data.pad)

    if not args.skip_tflite:
        path = convert_dynamic_range(model, args.output / "unet_dynamic_range.tflite")
        lite_predictions = predict_tflite(path, data.x_test)
        metrics["tflite_evaluation_without_padding"] = evaluate_masks(data.y_test, lite_predictions, pad=data.pad)
        metrics["tflite_tensor_details"] = get_tflite_tensor_types(path)
    (args.output / "evaluation.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
