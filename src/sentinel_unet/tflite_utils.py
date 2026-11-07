"""TFLite conversion, per-sample inference and tensor-type inspection."""
from pathlib import Path
import numpy as np


def convert_dynamic_range(model, destination: str | Path):
    """Request TFLite dynamic-range optimization, typically 8-bit stored weights.

    Inspect resulting tensors before making claims about specific operator/weight types.
    """
    import tensorflow as tf
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converted = converter.convert()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(converted)
    return destination


def predict_tflite(model_path: str | Path, images: np.ndarray) -> np.ndarray:
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    predictions = []
    for image in images:
        if interpreter.get_input_details()[0]["shape"][0] != 1:
            new_shape = np.array([1, *image.shape], dtype=np.int32)
            interpreter.resize_tensor_input(input_detail["index"], new_shape)
            interpreter.allocate_tensors()
            input_detail = interpreter.get_input_details()[0]
            output_detail = interpreter.get_output_details()[0]
        sample = image[np.newaxis, ...]
        scale, zero_point = input_detail["quantization"]
        if np.issubdtype(input_detail["dtype"], np.integer) and scale:
            sample = np.clip(np.round(sample / scale + zero_point), np.iinfo(input_detail["dtype"]).min, np.iinfo(input_detail["dtype"]).max)
        sample = sample.astype(input_detail["dtype"])
        interpreter.set_tensor(input_detail["index"], sample)
        interpreter.invoke()
        output = interpreter.get_tensor(output_detail["index"])
        out_scale, out_zero_point = output_detail["quantization"]
        if np.issubdtype(output_detail["dtype"], np.integer) and out_scale:
            output = (output.astype(np.float32) - out_zero_point) * out_scale
        predictions.append(output[0].astype(np.float32))
    return np.stack(predictions, axis=0)


def get_tflite_tensor_types(model_path: str | Path) -> dict:
    import tensorflow as tf
    itp = tf.lite.Interpreter(model_path=str(model_path))
    itp.allocate_tensors()
    counts = {}
    for t in itp.get_tensor_details():
        name = np.dtype(t["dtype"]).name
        counts[name] = counts.get(name, 0) + 1
    return {"model_size_bytes": Path(model_path).stat().st_size, "tensor_type_counts": counts}
