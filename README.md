# Super Model v1: Sticker-Centric Book Spine Detection

**A lightweight YOLOv11 pipeline that reads library call numbers off a bookshelf photo, in shelf order, for automated inventory.**

📄 **Paper:** [Computer Vision-Based Automatic Bookshelf Organization](https://ieeexplore.ieee.org/document/11636493) (IEEE, 2026)
📦 **Model weights, notebooks, results and test images:** [Google Drive folder](https://drive.google.com/drive/folders/1uv3XJsbOh32Au6SEQLGAVyFsrFkcjifY?usp=sharing)

## Overview

Library inventory is a *high-noise, low-signal* problem. The only thing needed to check shelf order is the call number, and it sits on a small, standardised white sticker. The rest of the spine is artwork, logos and stylised type, which confuses general-purpose models.

Most published approaches segment or detect the **whole spine**. This project takes a different approach: it detects **only the call-number sticker**. Narrowing the region of interest to a high-contrast, regular shape makes detection more accurate and much cheaper, which suits mobile and edge deployment.

## Pipeline (V7)

```
Bookshelf photo
  → YOLOv11m sticker detection (super_model_v1.pt)
  → Spatial sort: shelf rows top-to-bottom, then left-to-right
  → 15 px adaptive padding around each box
  → 3× bicubic upscale → CLAHE → 3×3 sharpening
  → EasyOCR with a strict allowlist (A-Z, 0-9, '.', '-')
  → Call numbers in shelf order
```

- **Adaptive padding** fixes "Z-clipping", where tight boxes cut characters at the sticker edge (a "7" or "Z" sliced in half).
- **Super-resolution preprocessing** makes millimetre-scale characters and decimal points legible. CLAHE evens out lighting on curved spines.
- **Strict allowlist OCR** stops the OCR engine from producing symbols that can't appear in a call number, which rejects noise from scuffs, barcodes and textures.

"V7" is the seventh iteration. Earlier versions using zero-shot segmentation (Segment Anything) failed on dense shelves, merging neighbouring books or splitting single ones.

## Results

The model was evaluated on a held-out validation set of **72 unseen images**:

| Metric | Score |
|---|---|
| mAP@50 | **0.995** |
| Precision | **0.997** |
| Recall | **0.999** |
| Box loss | 0.6077 |
| Confusion matrix | 697 true positives, 1 false negative, 0 false positives |
| F1 | 1.00 at confidence threshold 0.312 |

### Ablation

| Iteration | Architecture | Target | Preprocessing | Strict OCR allowlist | mAP@50 | Extraction success |
|---|---|---|---|---|---|---|
| V1 (baseline) | Segment Anything (SAM) | Full spine | None | No | 45.2% | 32.5% |
| V4 | YOLOv8 | Sticker | None | No | 88.4% | 71.0% |
| V6 | YOLOv11m | Sticker | 15 px padding + upscaling | No | 99.5% | 84.3% |
| **V7** | **YOLOv11m** | **Sticker** | **15 px padding + upscaling** | **Yes** | **99.5%** | **98.9%** |

### Related work

| Reference | Method | Target | mAP@50 |
|---|---|---|---|
| Ma et al., *Sensors* (2024) | Oriented R-CNN | Full spine | 90.22% |
| Li et al., *Electronics* (2025) | YOLOv11 + CBAM + PaddleOCR | Full spine | 97.40% |
| This work (V7) | YOLOv11m + EasyOCR | Sticker | 99.50% |

These figures come from different datasets and detection targets, so they are reference points rather than a like-for-like benchmark.

### Example output

This is a real run of the notebook on one shelf photo. It detected 24 stickers and printed them in shelf order, with each sticker's text lines joined by `|`:

```
Sticker 01: BL | 2747.3 | T67 | 2023
Sticker 02: 2 | BL | 2747.2 | A75 | 2003
Sticker 03: BL | 2747.8 | B35 | 2012
...
Sticker 16: BL | 2775.3 | D39 | 2006
Sticker 17: BL | 2775.3 | D46 | 2006
...
Sticker 23: BM | 199 | S3 | S7 | 2016
Sticker 24: BM | 199 | 8 | S719 | 2016
```

The full output is in [`notebooks/V7_pipeline.ipynb`](notebooks/V7_pipeline.ipynb). It also shows the kind of OCR errors that remain, such as `82747.8` for `2747.8` and stray fragments from neighbouring labels.

## How to run

1. Download `super_model_v1.pt` from the [Drive folder](https://drive.google.com/drive/folders/1uv3XJsbOh32Au6SEQLGAVyFsrFkcjifY?usp=sharing). Test images are in the same folder.
2. Install the dependencies (a GPU is optional; EasyOCR and YOLO both run on CPU, just more slowly):

   ```bash
   git clone https://github.com/humzagk/Book_Spine_Detection_Model.git
   cd Book_Spine_Detection_Model
   pip install -r requirements.txt
   ```

3. Run the pipeline on a shelf photo:

   ```bash
   python pipeline/v7_pipeline.py shelf.jpg --model super_model_v1.pt --out v7_results
   ```

   This prints the call numbers in shelf order. It also writes `v7_results/final_report.txt` and the preprocessed crop of every sticker to `v7_results/crops/`. Options: `--conf` (detection threshold, default 0.4) and `--padding` (default 15 px).

Alternatively, open [`notebooks/V7_pipeline.ipynb`](notebooks/V7_pipeline.ipynb) in Google Colab, upload the weights and an image, and run both cells.

## Training

| | |
|---|---|
| Base model | YOLOv11 Medium (`yolo11m.pt`), Ultralytics 8.3.20 |
| Environment | Python 3.9.16, PyTorch 2.2.2 + CUDA 11.8 |
| Hardware | NVIDIA Tesla V100 on the TRUBA HPC cluster |
| Schedule | 100 epochs, batch 16, image size 640, seed 0 (deterministic) |
| Data | "Virtual merge" of 9 labelled datasets (2,000+ images), one class: `Book-Label` |

The exact configurations are in [`configs/`](configs). `args.yaml` holds the full Ultralytics training arguments, and `master_data.yaml` lists the merged train and validation splits. To retrain, place the datasets under the `path:` root in `master_data.yaml` and run:

```bash
yolo cfg=configs/args.yaml
```

### Source datasets

The training data was drawn from these public [Roboflow Universe](https://universe.roboflow.com) datasets. Each dataset has its own licence.

- [book-spine-detector-model](https://universe.roboflow.com/test-awuby/book-spine-detector-model)
- [booklabels](https://universe.roboflow.com/zihaowang999-gmail-com/booklabels)
- [import-labels](https://universe.roboflow.com/will-jones/import-labels)
- [test-8m60q](https://universe.roboflow.com/cam-dyas/test-8m60q)
- [clara-book-label-image-detection](https://universe.roboflow.com/brian-prais/clara-book-label-image-detection)
- [np-ann](https://universe.roboflow.com/genebaybay-yahoo-com/np-ann)
- [a.i.](https://universe.roboflow.com/new-workspace-qnjlz/a.i.)
- [clara](https://universe.roboflow.com/will-jones/clara)
- [books-ponfj](https://universe.roboflow.com/zihaowang999-gmail-com/books-ponfj)
- [booklabels-c9zqr](https://universe.roboflow.com/classtest/booklabels-c9zqr)
- [books2](https://universe.roboflow.com/zihaowang999-gmail-com/books2)

## Repository structure

```
pipeline/v7_pipeline.py     # detection → sort → preprocess → OCR, as a CLI
notebooks/V7_pipeline.ipynb # original Colab notebook with a real run's output
configs/args.yaml           # Ultralytics training arguments for super_model_v1
configs/master_data.yaml    # merged dataset definition (9 datasets, 1 class)
tests/                      # unit tests for shelf sorting and cropping
requirements.txt
```

## Future work

- **Order checking**: compare the extracted call numbers against Dewey or Library of Congress order and flag misplaced volumes with their position on the shelf.
- **Edge deployment**: quantise the weights (FP32 → INT8) to run on-device on iOS and Android.
- **Wider validation**: more libraries, shelf materials (wood, reflective metal) and sticker standards.

## Acknowledgement

Numerical calculations were performed at TUBITAK ULAKBIM, High Performance and Grid Computing Center (TRUBA resources).

## License

Code is released under the [MIT License](LICENSE). The trained weights were produced from the third-party datasets above, so check their licences before using the weights commercially.
