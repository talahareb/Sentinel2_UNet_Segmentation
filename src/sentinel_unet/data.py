"""Dataset validation and split handling for the original Sentinel-2 lab."""

from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class DatasetSplit:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    pad: int
    mode: str


def _preprocess_images(x: np.ndarray, pad: int, train_max: float) -> np.ndarray:
    if pad:
        x = np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="constant")
    return (x / train_max).astype(np.float32, copy=False)


def _preprocess_masks(y: np.ndarray, pad: int) -> np.ndarray:
    if pad:
        y = np.pad(y, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="constant")
    return y.astype(np.float32, copy=False)


def load_dataset(path: str | Path, mode: str = "three-way", pad: int = 3) -> DatasetSplit:
    """Load X/Y from sentinel2.npz.

    Original: first 40 train, remaining 10 both validation and final evaluation.
    Three-way: first 35 train, next 5 validation, final 10 independent test.

    Records stay in source order, as in the lab. For real-world remote-sensing
    benchmarking, use region-disjoint train/validation/test partitions instead.
    """
    if mode not in {"original", "three-way"}:
        raise ValueError("mode must be 'original' or 'three-way'")
    if not isinstance(pad, int) or pad < 0:
        raise ValueError("pad must be a nonnegative integer")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}. See data/README.md.")
    with np.load(path, allow_pickle=False) as archive:
        if "X" not in archive.files or "Y" not in archive.files:
            raise KeyError("The .npz archive must contain both X and Y arrays")
        x = archive["X"].astype(np.float32, copy=False)
        y = archive["Y"].astype(np.float32, copy=False)

    if x.ndim != 4 or x.shape[-1] != 12:
        raise ValueError(f"Expected X shape (N, H, W, 12); received {x.shape}")
    if y.ndim == 3:
        y = y[..., np.newaxis]
    if y.ndim != 4 or y.shape[-1] != 1 or y.shape[:3] != x.shape[:3]:
        raise ValueError(f"Expected Y shape (N, H, W, 1) matching X; received {y.shape}")
    if len(x) < 50:
        raise ValueError("The original 40/10 lab split requires at least 50 images")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("Non-finite values found in dataset")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("Binary segmentation expects masks coded as 0 or 1")
    if (x.shape[1] + 2 * pad) % 4 or (x.shape[2] + 2 * pad) % 4:
        raise ValueError("Padded image height/width must be divisible by 4")

    if mode == "original":
        train_end = 40
        val_slice = slice(40, None)
        test_slice = slice(40, None)  # same held-out images used for validation and evaluation
    else:
        train_end = 35
        val_slice = slice(35, 40)
        test_slice = slice(40, None)
    train_max = float(np.max(x[:train_end]))
    if train_max <= 0:
        raise ValueError("Non-positive training maximum: cannot apply original normalization")

    x = _preprocess_images(x, pad, train_max)
    y = _preprocess_masks(y, pad)
    return DatasetSplit(x[:train_end], y[:train_end], x[val_slice], y[val_slice], x[test_slice], y[test_slice], pad, mode)
