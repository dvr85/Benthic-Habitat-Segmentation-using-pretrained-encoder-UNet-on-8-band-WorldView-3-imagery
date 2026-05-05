# Data layout

Raw imagery is **not** committed to the repository — the labeled set is ~1.1 GB of GeoTIFFs and PNGs. Populate this directory locally before running training or evaluation.

## Expected structure

```
data/
├── raw_labeled_data/                 # Training set (9 images)
│   ├── images/
│   │   ├── <stem>.tif                # 8-band uint16 WV3 GeoTIFF
│   │   └── ...
│   ├── annotations/
│   │   ├── <stem>.tif                # single-band uint8, values in {0..6}
│   │   └── ...
│   └── colored/annotations/
│       ├── <stem>.png                # RGB visualization of class indices
│       └── ...
│
└── test_data/
    ├── Agana/                        # 15 in-distribution clips
    │   ├── images/<stem>.tif         # 8-band uint16 WV3 GeoTIFF
    │   └── annotations/<stem>.png    # 3-channel RGB mask using CLASS_COLORS palette
    │
    └── Manell/                       # 6 out-of-distribution clips
        ├── images/<stem>.tif
        └── annotations/<stem>.png
```

## File-format notes

- **Image TIFFs**: 8 bands, uint16 digital numbers (DN), no normalization required at write time — the dataset class handles per-band z-scoring at load time.
- **Training annotations**: single-band uint8, values directly in `{0, 1, 2, 3, 4, 5, 6}` (Background, Seagrass, Coral, Macroalgae, Sand, Land, Ocean).
- **Test annotations**: 3-channel RGB PNGs whose colors map to class indices via the `CLASS_COLORS` palette in [`src/utils.py`](../src/utils.py). They are decoded back to class indices by `encode_rgb_to_class()` inside `src/evaluate.py`. **Do not pre-convert them** — the pipeline expects RGB.
- **Stems**: each image's `<stem>.tif` must have a matching annotation by the same stem in the corresponding annotations directory.

## Class palette (RGB)

| Class | Color |
|---|---|
| 0 — Background | `(0, 0, 0)` |
| 1 — Seagrass | `(134, 164, 117)` |
| 2 — Coral | `(255, 127, 80)` |
| 3 — Macroalgae | `(101, 138, 42)` |
| 4 — Sand | `(203, 189, 147)` |
| 5 — Land | `(139, 98, 76)` |
| 6 — Ocean | `(127, 205, 255)` |

## Sourcing the data

This project was built from a CISC 684 course dataset of Guam reef imagery. If you don't have access to that dataset, you can adapt the pipeline to any 8-band multispectral source by:

1. Replacing the imagery in `raw_labeled_data/` and `test_data/Agana/` and `test_data/Manell/`.
2. Re-running `uv run python scripts/compute_band_stats.py` to recompute the per-band normalization stats for your sensor.
3. Adjusting `cfg.model.in_channels` if your imagery has a different band count.
4. Updating `CLASS_NAMES` and `CLASS_COLORS` in `src/utils.py` if your label schema differs.
