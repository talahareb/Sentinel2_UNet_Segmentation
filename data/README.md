# Dataset setup

The original lab loads a local archive named **`sentinel2.npz`** containing:

- `X`: Sentinel-2 multispectral image patches, shape `(N, H, W, 12)`
- `Y`: binary cultivated-land segmentation masks, shape `(N, H, W, 1)` (or `(N,H,W)`), values `0` and `1`

The notebook shows **50 patches** with 40 used for training and 10 for evaluation. Its padding of three pixels on each side converts 250×250 patches to 256×256. Channels/band order, geographic origin, dates, acquisition conditions, class labeling method, source URL and dataset redistribution license **were not provided**. Therefore the data is not bundled, and we do not suggest a potentially incorrect public download URL.

To run:

1. Obtain the **same `sentinel2.npz` used in the course** from your original course materials, where you have permission to use it.
2. Place it as `data/sentinel2.npz` (Git ignores this file).
3. Run `python scripts/train_and_evaluate.py --data data/sentinel2.npz` from repository root.

For any other dataset, adapt band ordering, mask semantics, normalization and split logic after confirming equivalence. The repository intentionally does not substitute fabricated or generic satellite data for the original source.
