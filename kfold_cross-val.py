"""
CHANGE 2a: K-fold cross-validation for reliable per-class metrics.

With ~100 labelled images a single val/test split gives noisy per-class numbers.
K-fold pools ALL labelled images, trains k models each holding out a different
1/k as validation, and reports per-class mAP as mean +/- 95% CI ACROSS folds.
That variance is the honest signal of how reliable each per-class number is.

This is complementary to your frozen single test set, not a replacement:
  - frozen test  -> one clean, never-touched estimate for the final write-up;
  - k-fold       -> variance / reliability of the per-class estimates.

Cost is k x training. See the speed notes at the bottom.
"""

import os
import glob
import numpy as np
from ultralytics import YOLO
from sklearn.model_selection import KFold

try:
    from scipy import stats as _sps
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

# ---- config ----
DATASET_DIR = "/Users/malikaeryomina/SEI_AI_ultrasound/dataset_yolo"
YAML_NAMES  = {0: "abdomen", 1: "stomach", 2: "umbilical_vein", 3: "rib", 4: "spine"}
BASE_MODEL  = "yolov8n-seg.pt"
K           = 5
SEED        = 42
EPOCHS      = 100                      # MATCH train_segment_model.py; lower only for quick dev runs
IMGSZ       = 640
BATCH       = 8
DEVICE      = "mps"
POOL_SPLITS = ["train", "val"]         # keep the frozen TEST set OUT of CV
WORK_DIR    = "kfold_runs"


def gather_images():
    imgs = []
    for sp in POOL_SPLITS:
        d = os.path.join(DATASET_DIR, "images", sp)
        imgs += glob.glob(os.path.join(d, "*.jpg")) + \
                glob.glob(os.path.join(d, "*.jpeg")) + \
                glob.glob(os.path.join(d, "*.png"))
    imgs = sorted(set(imgs))            # deterministic order before the seeded split
    return imgs


def write_fold_yaml(fold_dir, train_list, val_list, names):
    os.makedirs(fold_dir, exist_ok=True)
    tr = os.path.join(fold_dir, "train.txt")
    va = os.path.join(fold_dir, "val.txt")
    with open(tr, "w") as f: f.write("\n".join(train_list) + "\n")
    with open(va, "w") as f: f.write("\n".join(val_list) + "\n")
    yml = os.path.join(fold_dir, "fold.yaml")
    with open(yml, "w") as f:
        f.write(f"path: {os.path.abspath(fold_dir)}\n")
        f.write(f"train: {os.path.abspath(tr)}\n")
        f.write(f"val: {os.path.abspath(va)}\n\nnames:\n")
        if names:
            for idx, nm in names.items():
                f.write(f"  {idx}: {nm}\n")
    return yml


def ci95(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    n = len(vals)
    if n == 0:
        return (np.nan, np.nan, np.nan, 0)
    m = vals.mean()
    if n == 1:
        return (m, np.nan, np.nan, 1)
    sd = vals.std(ddof=1)
    se = sd / np.sqrt(n)
    tcrit = _sps.t.ppf(0.975, n - 1) if _HAVE_SCIPY else 1.96
    return (m, m - tcrit * se, m + tcrit * se, n)


def main():
    images = gather_images()
    if len(images) < K:
        raise SystemExit(f"Only {len(images)} images pooled; need >= K={K}.")
    print(f"Pooled {len(images)} images from splits {POOL_SPLITS}; running {K}-fold CV.")

    kf = KFold(n_splits=K, shuffle=True, random_state=SEED)
    per_class = {}     # class_idx -> list of per-fold mAP50-95
    names_seen = YAML_NAMES

    for fold, (tr_idx, va_idx) in enumerate(kf.split(images)):
        fold_dir = os.path.join(WORK_DIR, f"fold_{fold}")
        train_list = [images[i] for i in tr_idx]
        val_list   = [images[i] for i in va_idx]
        yml = write_fold_yaml(fold_dir, train_list, val_list, names_seen)
        print(f"\n=== fold {fold}: {len(train_list)} train / {len(val_list)} val ===")

        model = YOLO(BASE_MODEL)
        # Augmentation recipe MIRRORS train_segment_model.py so the CV numbers
        # reflect your real model. If you change one, change both.
        model.train(data=yml, epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH,
                    device=DEVICE, seed=SEED, deterministic=True,
                    cache=True, patience=50,
                    copy_paste=0.4, mosaic=1.0, scale=0.5, fliplr=0.5,
                    degrees=10, translate=0.1,
                    hsv_h=0.0, hsv_s=0.0,               # hsv_v kept at default (brightness is real)
                    project=WORK_DIR, name=f"fold_{fold}_train", verbose=False)
        metrics = model.val(project=WORK_DIR, name=f"fold_{fold}_val", verbose=False)

        # Per-class mAP50-95 (segmentation). Field names can vary by ultralytics
        # version; print what is available if extraction fails.
        try:
            names_seen = names_seen or metrics.names
            maps = metrics.seg.maps                      # array indexed by class id
            cls_ids = getattr(metrics.seg, "ap_class_index", range(len(maps)))
            for c in cls_ids:
                per_class.setdefault(int(c), []).append(float(maps[int(c)]))
        except Exception as e:
            print(f"  !! could not read per-class seg maps ({e}). "
                  f"Inspect metrics.seg attributes for your ultralytics version.")

    print("\n================ K-FOLD PER-CLASS (mask mAP50-95) ================")
    print(f"{'class':<16}{'mean':>8}{'95% CI':>22}{'folds':>7}")
    for c in sorted(per_class):
        m, lo, hi, n = ci95(per_class[c])
        nm = (names_seen or {}).get(c, str(c)) if names_seen else str(c)
        ci = f"[{lo:.3f}, {hi:.3f}]" if not np.isnan(lo) else "  (n<2)"
        print(f"{str(nm):<16}{m:>8.3f}{ci:>22}{n:>7}")
    print("=================================================================")
    print("A wide CI or missing folds for a class == that per-class number is NOT reliable.")


if __name__ == "__main__":
    main()