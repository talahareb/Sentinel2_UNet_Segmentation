# Experiment and reproduction notes

## Original lab

- Source: `notebooks/original_lab3.ipynb`, retained unchanged, including notebook outputs and 100-epoch training log.
- Task: binary cultivated-land segmentation from 12-channel Sentinel-2 image patches.
- Original preprocessing: pad 3 pixels on each side; divide training and evaluation image values by **training-set maximum**.
- Partition: first **40** image patches for training and the remaining **10** for both per-epoch validation and final evaluation.
- Architecture: two Conv2D/BN/ReLU downsampling levels (32, 64 filters), 128-filter bottleneck, two upsampling/concatenation levels (64, 32 filters), one-channel sigmoid output.
- Fitting: Adam, binary cross-entropy, accuracy, 100 epochs, batch size 2.
- Notebook output: Keras accuracy **0.9178375005722046**, manually thresholded pixel accuracy **0.9178375244140625**, TFLite reported accuracy **0.9178375244140625**.

## Important interpretation

- Original notebook calls the final partition `X_test` but uses it as the validation partition during training. **There is no independent third test split** in that run.
- Pixel accuracy includes padded zero-valued borders, so it may be higher than accuracy measured on just the original 250×250 pixels. There are no recorded foreground IoU, Dice, precision or recall values.
- The notebook reports TFLite conversion with a deprecated `OPTIMIZE_FOR_SIZE` configuration and `supported_types=[tf.float32]`. That output is not sufficient to independently verify 8-bit weight storage. Neither the `.tflite` artifact nor original dataset was provided. The companion pipeline uses `Optimize.DEFAULT` and provides tensor-type counts so users can inspect the produced file.
- Training console values are from **one recorded run** and are not independently re-executed here.
- The Sentinel-2 patches' geographic origin and train/test geographic independence are unknown. Sample-based splitting may overstate geographic generalization.
- Any future evaluation must report metrics on **unpadded** pixels and ideally reserve region-disjoint, held-out imagery.

## Companion pipeline

The newly organized `src/` and `scripts/` code is a reproducibility aid based on the lab model. It is not being represented as the untouched original coursework. The default three-way split is **35 train / 5 validation / 10 test**. This makes an independently held-out test partition by index, but geographic independence cannot be confirmed without scene metadata. Use `--split-mode original` to follow the notebook's 40/10 protocol (including validation reuse).
