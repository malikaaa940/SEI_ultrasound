#!/usr/bin/env python3
"""
Build side-by-side comparisons: model's fitted-AC overlay next to the
EXPERT 'correct outline' image, paired by image NUMBER.

The expert folder contains three files per image number:
    Image N blank.jpg
    Image N correct outline.jpg      <- the one we want
    Image N incorrect outline.jpg
So we match on number AND filter to the file whose name contains
EXPERT_KEYWORD ('correct') while excluding 'incorrect' (which also
contains the substring 'correct').
"""

import re
from pathlib import Path
import cv2
import numpy as np

# --- Config -----------------------------------------------------------------
MODEL_DIR = "/Users/malikaeryomina/SEI_AI_ultrasound/predictions/abdomen_ellipse"
EXPERT_DIR = "/Users/malikaeryomina/SEI_AI_ultrasound/Triplicate images"  # <- set to your expert folder
OUTPUT_DIR = "/Users/malikaeryomina/SEI_AI_ultrasound/predictions/side_by_side_incorrect"

EXPERT_KEYWORD = "incorrect"    # pick the expert file containing this...

TARGET_HEIGHT = 500
MODEL_LABEL = "Model (fitted AC)"
EXPERT_LABEL = "Expert (correct outline)"
MAKE_COMBINED = True
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
# ----------------------------------------------------------------------------


def image_id(name):
    """First integer in a filename, e.g. 'Image 11 blank.jpg' -> '11'."""
    m = re.search(r"\d+", name)
    return m.group() if m else None


def index_model(d):
    """Map image-number -> model overlay path."""
    out = {}
    folder = Path(d)
    if not folder.is_dir():
        return out
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in IMAGE_EXTS:
            k = image_id(p.name)
            if k is not None and k not in out:
                out[k] = p
    return out


def index_expert(d):
    """Map image-number -> the 'correct outline' expert path only."""
    out = {}
    folder = Path(d)
    if not folder.is_dir():
        return out
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        name = p.name.lower()
        if EXPERT_KEYWORD in name:
            k = image_id(p.name)
            if k is not None and k not in out:
                out[k] = p
    return out


def resize_to_height(img, h):
    scale = h / img.shape[0]
    return cv2.resize(img, (max(1, int(img.shape[1] * scale)), h))


def add_label_bar(img, text):
    bar = np.full((40, img.shape[1], 3), 40, dtype=np.uint8)
    cv2.putText(bar, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([bar, img])


def make_pair(model_img, expert_img):
    m = add_label_bar(resize_to_height(model_img, TARGET_HEIGHT), MODEL_LABEL)
    e = add_label_bar(resize_to_height(expert_img, TARGET_HEIGHT), EXPERT_LABEL)
    divider = np.full((m.shape[0], 4, 3), 90, dtype=np.uint8)
    return np.hstack([m, divider, e])


def pad_to_width(img, width):
    if img.shape[1] >= width:
        return img
    pad = np.full((img.shape[0], width - img.shape[1], 3), 20, dtype=np.uint8)
    return np.hstack([img, pad])


def main():
    model_idx = index_model(MODEL_DIR)
    expert_idx = index_expert(EXPERT_DIR)

    if not model_idx:
        print(f"No model images found in {Path(MODEL_DIR).resolve()}")
        return
    if not expert_idx:
        print(f"No '{EXPERT_KEYWORD}' expert images found in {Path(EXPERT_DIR).resolve()}")
        return

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    shared = sorted(set(model_idx) & set(expert_idx), key=lambda x: int(x))
    only_model = sorted(set(model_idx) - set(expert_idx), key=lambda x: int(x))
    only_expert = sorted(set(expert_idx) - set(model_idx), key=lambda x: int(x))

    if only_model:
        print(f"No expert 'correct' match for image(s): {', '.join(only_model)}")
    if only_expert:
        print(f"No model match for image(s): {', '.join(only_expert)}")
    if not shared:
        print("No overlapping image numbers between the two folders.")
        return

    print(f"\nBuilding {len(shared)} side-by-side comparison(s)...\n")

    pairs = []
    for k in shared:
        model_img = cv2.imread(str(model_idx[k]))
        expert_img = cv2.imread(str(expert_idx[k]))
        if model_img is None or expert_img is None:
            print(f"  Image {k}: could not read one of the files, skipping")
            continue

        composite = make_pair(model_img, expert_img)

        title = np.full((36, composite.shape[1], 3), 20, dtype=np.uint8)
        cv2.putText(title, f"Image {k}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 200, 255), 2, cv2.LINE_AA)
        composite = np.vstack([title, composite])

        out_path = out_dir / f"compare_image_{k}.jpg"
        cv2.imwrite(str(out_path), composite)
        pairs.append(composite)
        print(f"  Image {k}: {model_idx[k].name}  |  {expert_idx[k].name}")

    if MAKE_COMBINED and pairs:
        max_w = max(p.shape[1] for p in pairs)
        gap = np.full((10, max_w, 3), 20, dtype=np.uint8)
        stacked = []
        for p in pairs:
            stacked.append(pad_to_width(p, max_w))
            stacked.append(gap)
        combined = np.vstack(stacked[:-1])
        combined_path = out_dir / "compare_all.jpg"
        cv2.imwrite(str(combined_path), combined)
        print(f"\n  Combined montage: {combined_path.name}")

    print(f"\nDone. Comparisons saved in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()