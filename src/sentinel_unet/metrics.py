"""Binary segmentation evaluation beyond overall pixel accuracy."""
import numpy as np


def evaluate_masks(y_true: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5, pad: int = 0) -> dict:
    """Return per-pixel accuracy, foreground IoU/Dice, precision, recall.

    Removes any artificial pad region. Empty-positive masks use 1.0 for
    undefined precision/IoU/Dice only if both arrays contain no positives.
    """
    true = np.asarray(y_true)
    pred = np.asarray(probabilities)
    if true.shape != pred.shape:
        raise ValueError(f"Mask shape mismatch: {true.shape} vs {pred.shape}")
    if true.ndim != 4 or true.shape[-1] != 1:
        raise ValueError("Expected masks of shape (N,H,W,1)")
    if not (0 <= threshold <= 1):
        raise ValueError("threshold must be in [0,1]")
    if pad < 0 or (pad and (true.shape[1] <= 2 * pad or true.shape[2] <= 2 * pad)):
        raise ValueError("Invalid pad")
    if pad:
        true = true[:, pad:-pad, pad:-pad, :]
        pred = pred[:, pad:-pad, pad:-pad, :]
    tr = (true > 0.5)
    pr = (pred > threshold)
    tp = int(np.sum(tr & pr))
    fp = int(np.sum(~tr & pr))
    fn = int(np.sum(tr & ~pr))
    tn = int(np.sum(~tr & ~pr))
    n = tp + fp + fn + tn
    return {
        "pixel_accuracy": float((tp + tn) / n),
        "foreground_iou": float(tp / (tp + fp + fn)) if tp + fp + fn else 1.0,
        "foreground_dice": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 1.0,
        "precision": float(tp / (tp + fp)) if tp + fp else (1.0 if not tr.any() else 0.0),
        "recall": float(tp / (tp + fn)) if tp + fn else 1.0,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "evaluated_pixels": n,
    }
