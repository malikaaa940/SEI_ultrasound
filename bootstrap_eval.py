"""
CHANGE 2b: Bootstrap confidence intervals for per-class metrics on the frozen test set.

model.val() gives one aggregate number per class with no uncertainty. On 19 test
images that number is fragile. Bootstrapping resamples the test IMAGES with
replacement B times and recomputes the per-class metric each time; the 2.5th and
97.5th percentiles give a 95% CI that shows how much the estimate could swing.

Metric used here is class-level (semantic) overlap, not COCO mAP:
  for each image and class, we union all predicted masks of that class into one
  binary mask, do the same for ground truth, and compute Dice (DSC) and IoU.
  This sidesteps fragile instance matching and answers the practical question --
  "when this structure is present, how well is it covered?" -- which is what the
  downstream quality score actually consumes. It is NOT the same number as
  mAP50-95; report it as a complementary, interpretable per-class measure.

Also reported per class:
  detection_rate = fraction of GT-present images where the model predicted any
                   pixels of that class (a recall-like signal).
"""

import os
import glob
import numpy as np
import cv2
from ultralytics import YOLO

# ---- config ----
MODEL_PATH  = "/Users/malikaeryomina/SEI_AI_ultrasound/best (1).pt"  # the run you just trained -- CONFIRM the exact folder from the training output
DATASET_DIR = "/Users/malikaeryomina/SEI_AI_ultrasound/dataset_yolo"
SPLIT       = "test"
DEVICE      = "mps"
CONF        = 0.25
N_BOOT      = 2000
SEED        = 42
CLASS_NAMES = None      # {0:"abdomen",...}; None -> pulled from model.names


def poly_to_mask(polys_norm_or_px, H, W, normalised):
    """Rasterise a list of polygons into one binary mask."""
    m = np.zeros((H, W), np.uint8)
    for p in polys_norm_or_px:
        p = np.asarray(p, dtype=float).reshape(-1, 2)
        if p.shape[0] < 3:
            continue
        q = p.copy()
        if normalised:
            q[:, 0] *= W
            q[:, 1] *= H
        cv2.fillPoly(m, [q.astype(np.int32)], 1)
    return m


def read_gt(txt_path):
    """class_idx -> list of normalised polygons."""
    out = {}
    if not os.path.exists(txt_path):
        return out
    with open(txt_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 7:
                continue
            cls = int(float(parts[0]))
            coords = np.array(parts[1:], dtype=float)
            coords = coords[: (len(coords) // 2) * 2].reshape(-1, 2)
            out.setdefault(cls, []).append(coords)
    return out


def dsc_iou(a, b):
    a = a.astype(bool); b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    sa, sb = a.sum(), b.sum()
    union = np.logical_or(a, b).sum()
    dsc = 2 * inter / (sa + sb) if (sa + sb) > 0 else np.nan
    iou = inter / union if union > 0 else np.nan
    return dsc, iou


def collect_records(model, class_names):
    img_dir = os.path.join(DATASET_DIR, "images", SPLIT)
    lbl_dir = os.path.join(DATASET_DIR, "labels", SPLIT)
    paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) +
                   glob.glob(os.path.join(img_dir, "*.jpeg")) +
                   glob.glob(os.path.join(img_dir, "*.png")))
    records = []   # one dict per image: {cls: {"dsc":..,"iou":..,"detected":..,"gt_present":..}}
    for pth in paths:
        img = cv2.imread(pth)
        if img is None:
            continue
        H, W = img.shape[:2]
        gt = read_gt(os.path.join(lbl_dir, os.path.splitext(os.path.basename(pth))[0] + ".txt"))

        res = model.predict(pth, device=DEVICE, conf=CONF, verbose=False)[0]
        pred_polys = {}   # cls -> list of pixel polygons
        if res.masks is not None:
            cls_ids = res.boxes.cls.cpu().numpy().astype(int)
            for poly, c in zip(res.masks.xy, cls_ids):
                pred_polys.setdefault(int(c), []).append(poly)

        rec = {}
        for c in class_names:
            gt_present = c in gt and len(gt[c]) > 0
            gt_mask = poly_to_mask(gt.get(c, []), H, W, normalised=True)
            pr_mask = poly_to_mask(pred_polys.get(c, []), H, W, normalised=False)
            detected = pr_mask.sum() > 0
            if gt_present:
                d, i = dsc_iou(pr_mask, gt_mask)
            else:
                d, i = np.nan, np.nan
            rec[c] = {"dsc": d, "iou": i, "detected": detected, "gt_present": gt_present}
        records.append(rec)
    return records


def summarise(records, class_names, class_labels):
    rng = np.random.default_rng(SEED)
    n = len(records)
    idx_all = np.arange(n)

    print(f"\nBootstrapping over {n} test images x {N_BOOT} resamples "
          f"(conf={CONF}, seed={SEED}).")
    print(f"{'class':<16}{'Dice mean':>11}{'95% CI (Dice)':>22}"
          f"{'det.rate':>10}{'95% CI (det)':>20}{'GT imgs':>9}")

    for c in class_names:
        dsc_vals = np.array([r[c]["dsc"] for r in records if r[c]["gt_present"]], dtype=float)
        det_vals = np.array([1.0 if r[c]["detected"] else 0.0
                             for r in records if r[c]["gt_present"]], dtype=float)
        gt_imgs = len(dsc_vals)
        if gt_imgs == 0:
            print(f"{str(class_labels.get(c,c)):<16}{'--':>11}{'(no GT in test)':>22}"
                  f"{'--':>10}{'--':>20}{0:>9}")
            continue

        # bootstrap over the GT-present images for this class
        present_idx = np.array([k for k, r in enumerate(records) if r[c]["gt_present"]])
        boot_dsc, boot_det = [], []
        for _ in range(N_BOOT):
            samp = rng.choice(present_idx, size=len(present_idx), replace=True)
            dd = np.array([records[s][c]["dsc"] for s in samp], dtype=float)
            ee = np.array([1.0 if records[s][c]["detected"] else 0.0 for s in samp])
            boot_dsc.append(np.nanmean(dd))
            boot_det.append(ee.mean())
        d_lo, d_hi = np.percentile(boot_dsc, [2.5, 97.5])
        e_lo, e_hi = np.percentile(boot_det, [2.5, 97.5])

        print(f"{str(class_labels.get(c,c)):<16}"
              f"{np.nanmean(dsc_vals):>11.3f}"
              f"{f'[{d_lo:.3f}, {d_hi:.3f}]':>22}"
              f"{det_vals.mean():>10.3f}"
              f"{f'[{e_lo:.3f}, {e_hi:.3f}]':>20}"
              f"{gt_imgs:>9}")
    print("\nA class with few GT images and/or a very wide Dice CI is NOT reliably estimated.")


def main():
    model = YOLO(MODEL_PATH)
    class_labels = CLASS_NAMES or model.names
    class_names = sorted(class_labels.keys())
    records = collect_records(model, class_names)
    if not records:
        raise SystemExit("No test images found -- check DATASET_DIR / SPLIT.")
    summarise(records, class_names, class_labels)


if __name__ == "__main__":
    main()