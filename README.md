# Super Model v1: Sticker-Centric Book Spine Detection

**A high-precision, lightweight YOLOv11 model for automated library inventory management.**

## 📌 Overview 

**Super Model v1** (`super_model_v1.pt`) represents a paradigm shift in automated library inventory. Unlike traditional approaches that attempt to segment entire book spines—struggling with varied artwork, reflections, and complex backgrounds—this model focuses exclusively on the **Call Number Sticker** as the primary signal.

By narrowing the region of interest (ROI) to this high-contrast, standardized geometric target, the model achieves near-perfect detection rates with significantly lower computational overhead, making it ideal for edge deployment.

> **Note:** The full methodology, including the "Forensic Analysis of Iterative Failures" and "Spatial Sorting Logic," will be detailed in my upcoming research paper. *Stay tuned for the publication release.*

---

## 📥 Model Download

Due to GitHub's file size restrictions, the pre-trained model weights (`super_model_v1.pt`) and high-resolution evaluation metrics are hosted externally.

**You can access the model and results via this public Google Drive link:**
👉 **[Download Super Model v1 & Results](https://drive.google.com/drive/folders/15RD4xqWAx8WoOGeiW56Ca_HNutzfSKvs?usp=share_link)**

---

## 🚀 The Paradigm Shift

Current state-of-the-art research often treats library inventory as a **segmentation** problem, attempting to outline every book spine. This approach is computationally expensive and prone to "hallucination" when spines lack clear boundaries or feature complex art.

**My Approach (V7 Architecture):**

* **Target:** Detects the *sticker*, not the book.


* **Philosophy:** "High-Signal, Low-Noise." The sticker is a standardized white rectangle; the rest of the spine is noise.


* **Result:** A simpler, faster, and more robust pipeline that filters out cover art and focuses strictly on the data that matters for inventory: the Call Number.



---

## 📊 Performance Benchmarks

`Super Model v1` was trained on a custom dataset using a "Virtual Merge" strategy on a High-Performance Computing (HPC) cluster. It outperforms current SOTA benchmarks in precision and recall for this specific task.

| Metric | Score | Interpretation |
| --- | --- | --- |
| **mAP@50** | **0.995** | 99.5% accuracy at standard thresholds. |
| **Precision** | **0.997** | Near-zero false positives; extreme noise rejection. |
| **Recall** | **0.999** | Virtually no missed volumes in dense shelf environments.|
| **Box Loss** | **0.6077** | Minimal localization error for tight bounding boxes. |

### Comparison to Related Work

* **My Model (V7):** **99.5% mAP** (Sticker-Centric YOLOv11).


* *Reference A (Sensors Journal):* 90.22% mAP (Oriented R-CNN on full spines).


* *Reference B (Electronics Journal):* 97.4% mAP (YOLOv11 + CBAM on full spines).



---

## 🛠️ Model Architecture

* **Base Model:** YOLOv11 Medium (`yolo11m.pt`).


* **Training Hardware:** Tesla V100 GPU (TRUBA HPC Cluster).


* **Key Optimizations:**
* **Adaptive Padding (+15px):** Prevents "Z-Clipping" where characters at the edge of the sticker (like 'Z' or '7') are cut off.


* **Strict Allowlist OCR:** Post-processing filter to reject hallucinated symbols.





---

## 🔜 Coming Soon: Research Paper

A comprehensive technical report detailing the full pipeline is currently in preparation. It will cover:

1. **Forensic Analysis:** Why segmentation models (SAM) failed in high-density environments.


2. **Spatial Logic:** The algorithm used to sort detected stickers into correct shelf order.


3. **OCR Pipeline:** The "Sharpening + Bicubic Upscaling" preprocessing steps that enable reading decimal points on stickers.



---

*Created by Humza Gohar Kabir. For inquiries regarding the dataset or collaboration, please open an issue.*
