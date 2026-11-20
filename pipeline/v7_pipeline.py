"""
V7 sticker-centric pipeline: detect call-number stickers on a bookshelf photo,
put them in shelf order, and read each call number.

    raw image -> YOLOv11m sticker detection (super_model_v1.pt)
              -> spatial sort (shelf rows top-to-bottom, then left-to-right)
              -> 15 px padded crop -> 3x bicubic upscale, CLAHE, sharpening
              -> EasyOCR restricted to A-Z, 0-9, '.', '-'

Usage:
    python pipeline/v7_pipeline.py shelf.jpg --model super_model_v1.pt --out results/
"""

import argparse
import os

import cv2
import numpy as np

ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.- "


def preprocess_for_ocr(image):
    """Standardise a sticker crop for OCR: 3x upscale, greyscale, CLAHE, sharpen."""
    # 1. Upscale (3x): critical for small decimals such as ".3"
    zoom = cv2.resize(image, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    # 2. Greyscale
    gray = cv2.cvtColor(zoom, cv2.COLOR_BGR2GRAY)
    # 3. CLAHE: evens out lighting across curved spines
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    # 4. Sharpening: crisps the edges of characters
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(enhanced, -1, kernel)


def group_rows(boxes):
    """
    Group detections into shelf rows by their vertical centres.

    Boxes are sorted by centre y; a new row starts whenever the gap to the
    previous box's centre exceeds half the median box height. This adapts to
    image resolution and shelf spacing, unlike fixed pixel bands.
    """
    if not boxes:
        return []
    heights = sorted(b[3] - b[1] for b in boxes)
    threshold = max(1, heights[len(heights) // 2]) / 2
    by_y = sorted(boxes, key=lambda b: (b[1] + b[3]) / 2)
    rows, prev_cy = [[by_y[0]]], (by_y[0][1] + by_y[0][3]) / 2
    for b in by_y[1:]:
        cy = (b[1] + b[3]) / 2
        if cy - prev_cy > threshold:
            rows.append([])
        rows[-1].append(b)
        prev_cy = cy
    return rows


def sort_boxes(boxes):
    """
    Sort detections in reading order: shelf rows top-to-bottom, then left-to-right.
    Boxes are [x1, y1, x2, y2, conf].
    """
    return [b for row in group_rows(boxes) for b in sorted(row, key=lambda b: b[0])]


def padded_crop(img, box, padding):
    """Crop with `padding` pixels on each side (prevents clipping edge characters), clamped to the image."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box[:4]
    return img[max(0, y1 - padding):min(h, y2 + padding), max(0, x1 - padding):min(w, x2 + padding)]


def read_sticker(reader, crop, min_conf=0.2):
    """OCR a preprocessed crop and join its text lines top-to-bottom with ' | '."""
    lines = []
    for bbox, text, conf in reader.readtext(crop, detail=1, allowlist=ALLOWLIST):
        if conf > min_conf:
            y_center = (bbox[0][1] + bbox[2][1]) / 2
            lines.append((y_center, text.upper()))
    lines.sort(key=lambda x: x[0])
    return " | ".join(text for _, text in lines)


def run(image_path, model_path, out_dir, conf=0.4, padding=15):
    import torch
    import easyocr
    from ultralytics import YOLO

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    os.makedirs(os.path.join(out_dir, "crops"), exist_ok=True)

    model = YOLO(model_path)
    reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())

    detections = []
    for r in model.predict(image_path, conf=conf, verbose=False):
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            detections.append([int(x1), int(y1), int(x2), int(y2), float(box.conf)])
    detections = sort_boxes(detections)

    results = []
    for i, box in enumerate(detections, start=1):
        clean = preprocess_for_ocr(padded_crop(img, box, padding))
        cv2.imwrite(os.path.join(out_dir, "crops", f"sticker_{i:03d}.jpg"), clean)
        results.append(read_sticker(reader, clean))

    with open(os.path.join(out_dir, "final_report.txt"), "w") as f:
        f.write(f"V7 RESULTS ({len(results)} Stickers)\n================================\n")
        for i, text in enumerate(results, start=1):
            f.write(f"Sticker {i:02d}: {text}\n")
    return results


def main():
    p = argparse.ArgumentParser(description="Detect and read library call-number stickers in shelf order.")
    p.add_argument("image", help="bookshelf photo")
    p.add_argument("--model", default="super_model_v1.pt", help="path to the trained YOLO weights")
    p.add_argument("--out", default="v7_results", help="output directory (crops + final_report.txt)")
    p.add_argument("--conf", type=float, default=0.4, help="detection confidence threshold (default 0.4)")
    p.add_argument("--padding", type=int, default=15, help="pixels added around each box before cropping")
    args = p.parse_args()

    results = run(args.image, args.model, args.out, args.conf, args.padding)
    print(f"Found {len(results)} stickers.\n")
    for i, text in enumerate(results, start=1):
        print(f"Sticker {i:02d}: {text}")


if __name__ == "__main__":
    main()
