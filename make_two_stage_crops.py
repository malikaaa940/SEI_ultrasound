"""
CHANGE 3: Two-stage crop-then-segment (dataset builder for stage 2).

Idea: the coarse cone crop keeps a lot of non-target tissue, so thin structures
(UV, ribs) occupy few pixels. Stage 2 fixes this by cropping each image tightly
to the abdomen and segmenting inside that crop, where the target fills the frame.

This script builds the stage-2 TRAINING data. For each image it:
  1. reads the YOLO-seg label,
  2. finds the abdomen polygon (crop class), takes its bounding box + margin,
  3. crops the image to that box,
  4. transforms EVERY polygon (all classes) into the crop's coordinate frame and
     re-normalises, so no re-annotation is needed,
  5. writes the cropped image + new label into a parallel dataset.

HONEST CAVEATS (state these to the examiner):
  * Training crops come from the GROUND-TRUTH abdomen box; at inference the box
    must come from a stage-1 detector. Boxes will differ -> train/inference
    distribution shift. Mitigate with a generous margin and, later, by cropping
    the training set from *predicted* stage-1 boxes too.
  * Polygon clipping here is vertex clamping to the crop edges, not true
    polygon-rectangle intersection. A margin keeps most structures fully inside,
    so distortion is minor; set USE_SHAPELY=True for exact clipping if installed.
  * Images with no abdomen label are skipped and logged (nothing to crop around).
"""

import os
import cv2
import yaml
import numpy as np

# ---- config ----
DATASET_DIR   = "/Users/malikaeryomina/SEI_AI_ultrasound/dataset_yolo"
YAML_PATH     = "/Users/malikaeryomina/SEI_AI_ultrasound/dataset.yaml"
OUT_DIR       = "/Users/malikaeryomina/SEI_AI_ultrasound/dataset_yolo_cropped"
CROP_CLASS_NAME = "abdomen"     # resolves to index 0 in your yaml
CROP_CLASS_INDEX_FALLBACK = 0
MARGIN_FRAC   = 0.20            # wider margin so spine (often outside abdomen) is less clipped
MIN_CROP_PX   = 32             # skip degenerate tiny crops
USE_SHAPELY   = False          # exact polygon clipping if shapely is installed
SPLITS        = ["train", "val", "test"]


def resolve_crop_class():
    try:
        with open(YAML_PATH) as f:
            names = yaml.safe_load(f).get("names", {})
        for idx, nm in names.items():
            if str(nm).strip().lower() == CROP_CLASS_NAME.lower():
                return int(idx), names
        print(f"  -> '{CROP_CLASS_NAME}' not found in yaml names; "
              f"using fallback index {CROP_CLASS_INDEX_FALLBACK}.")
        return CROP_CLASS_INDEX_FALLBACK, names
    except Exception as e:
        print(f"  -> could not read {YAML_PATH} ({e}); "
              f"using fallback index {CROP_CLASS_INDEX_FALLBACK}.")
        return CROP_CLASS_INDEX_FALLBACK, {}


def read_label(txt_path):
    """Return list of (cls, poly_xy_normalised Nx2 float)."""
    rows = []
    if not os.path.exists(txt_path):
        return rows
    with open(txt_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 7:            # cls + at least 3 (x,y) pairs
                continue
            cls = int(float(parts[0]))
            coords = np.array(parts[1:], dtype=float)
            coords = coords[: (len(coords) // 2) * 2].reshape(-1, 2)
            rows.append((cls, coords))
    return rows


def abdomen_bbox_px(rows, crop_cls, W, H):
    """Bounding box (x1,y1,x2,y2) in pixels of the largest crop-class polygon."""
    best = None
    best_area = -1.0
    for cls, poly in rows:
        if cls != crop_cls:
            continue
        px = poly.copy()
        px[:, 0] *= W
        px[:, 1] *= H
        x1, y1 = px[:, 0].min(), px[:, 1].min()
        x2, y2 = px[:, 0].max(), px[:, 1].max()
        area = (x2 - x1) * (y2 - y1)
        if area > best_area:
            best_area, best = area, (x1, y1, x2, y2)
    return best


def pad_and_clip_box(box, W, H):
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1 -= MARGIN_FRAC * bw
    x2 += MARGIN_FRAC * bw
    y1 -= MARGIN_FRAC * bh
    y2 += MARGIN_FRAC * bh
    x1 = int(max(0, round(x1)));  y1 = int(max(0, round(y1)))
    x2 = int(min(W, round(x2)));  y2 = int(min(H, round(y2)))
    return x1, y1, x2, y2


def transform_polys(rows, box, W, H):
    """Map every polygon into the crop frame, re-normalise, clip. Returns new rows."""
    x1, y1, x2, y2 = box
    cw, ch = x2 - x1, y2 - y1
    out = []
    for cls, poly in rows:
        px = poly.copy()
        px[:, 0] = px[:, 0] * W - x1
        px[:, 1] = px[:, 1] * H - y1

        if USE_SHAPELY:
            try:
                from shapely.geometry import Polygon, box as shp_box
                inter = Polygon(px).buffer(0).intersection(shp_box(0, 0, cw, ch))
                if inter.is_empty:
                    continue
                geoms = list(inter.geoms) if inter.geom_type == "MultiPolygon" else [inter]
                for g in geoms:
                    xy = np.array(g.exterior.coords)[:-1]
                    norm = np.stack([xy[:, 0] / cw, xy[:, 1] / ch], 1)
                    out.append((cls, np.clip(norm, 0, 1)))
                continue
            except Exception:
                pass  # fall through to clamp

        # Drop polygons whose bbox lies entirely outside the crop.
        if px[:, 0].max() < 0 or px[:, 1].max() < 0 or px[:, 0].min() > cw or px[:, 1].min() > ch:
            continue
        norm = np.stack([px[:, 0] / cw, px[:, 1] / ch], 1)
        out.append((cls, np.clip(norm, 0.0, 1.0)))   # clamp vertices to crop edges
    return out


def write_label(txt_path, rows):
    with open(txt_path, "w") as f:
        for cls, poly in rows:
            flat = " ".join(f"{v:.6f}" for v in poly.reshape(-1))
            f.write(f"{cls} {flat}\n")


def main():
    crop_cls, names = resolve_crop_class()
    print(f"Cropping around class index {crop_cls} ('{CROP_CLASS_NAME}'), "
          f"margin={MARGIN_FRAC}, shapely={USE_SHAPELY}")

    stats = {"written": 0, "no_abdomen": 0, "unreadable": 0, "too_small": 0}
    skipped = []

    for split in SPLITS:
        img_dir = os.path.join(DATASET_DIR, "images", split)
        lbl_dir = os.path.join(DATASET_DIR, "labels", split)
        out_img = os.path.join(OUT_DIR, "images", split)
        out_lbl = os.path.join(OUT_DIR, "labels", split)
        os.makedirs(out_img, exist_ok=True)
        os.makedirs(out_lbl, exist_ok=True)
        if not os.path.isdir(img_dir):
            continue

        for fn in os.listdir(img_dir):
            if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img = cv2.imread(os.path.join(img_dir, fn), cv2.IMREAD_UNCHANGED)
            if img is None:
                stats["unreadable"] += 1;  skipped.append((fn, "unreadable"));  continue
            H, W = img.shape[:2]

            rows = read_label(os.path.join(lbl_dir, os.path.splitext(fn)[0] + ".txt"))
            box = abdomen_bbox_px(rows, crop_cls, W, H)
            if box is None:
                stats["no_abdomen"] += 1;  skipped.append((fn, "no_abdomen"));  continue

            x1, y1, x2, y2 = pad_and_clip_box(box, W, H)
            if (x2 - x1) < MIN_CROP_PX or (y2 - y1) < MIN_CROP_PX:
                stats["too_small"] += 1;  skipped.append((fn, "too_small"));  continue

            crop = img[y1:y2, x1:x2]
            new_rows = transform_polys(rows, (x1, y1, x2, y2), W, H)

            cv2.imwrite(os.path.join(out_img, fn), crop)
            write_label(os.path.join(out_lbl, os.path.splitext(fn)[0] + ".txt"), new_rows)
            stats["written"] += 1

    # stage-2 yaml
    yaml_out = os.path.join(OUT_DIR, "dataset.yaml")
    with open(yaml_out, "w") as f:
        f.write(f"path: {os.path.abspath(OUT_DIR)}\n")
        f.write("train: images/train\nval: images/val\ntest: images/test\n\nnames:\n")
        if names:
            for idx, nm in names.items():
                f.write(f"  {idx}: {nm}\n")

    print("\n--- two-stage crop summary ---")
    for k, v in stats.items():
        print(f"  {k:12s}: {v}")
    if skipped:
        print(f"  (skipped list, first 10): {skipped[:10]}")
    print(f"Stage-2 dataset -> {OUT_DIR}\nStage-2 yaml    -> {yaml_out}")


if __name__ == "__main__":
    main()