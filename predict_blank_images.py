#!/usr/bin/env python3
"""
Run the model on the blank images, keep only the real abdomen, and draw BOTH:
  - the raw abdomen segmentation mask (translucent fill + edge), and
  - the smooth fitted AC ellipse on top of it.

This lets you see what the model actually segmented (jagged mask) alongside
the clinical AC ellipse fitted to it.

Fixes two problems with the raw predict output:
  1. Jagged outline  -> fit an ellipse to the abdomen mask (cv2.fitEllipse).
                        The AC is clinically an ellipse anyway, so this is
                        both smoother AND the correct thing to measure.
  2. Extra abdomen on the small panel / thumbnail:
        - CONF threshold drops low-confidence detections, and
        - we keep only the LARGEST abdomen mask per image (the real one is
          bigger than the small-panel duplicate).
"""

from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

# --- Config -----------------------------------------------------------------
MODEL_PATH = "/Users/malikaeryomina/runs/segment/train-14/weights/best.pt"
SOURCE_DIR = "/Users/malikaeryomina/SEI_AI_ultrasound/blank_images"
OUTPUT_DIR = "/Users/malikaeryomina/SEI_AI_ultrasound/predictions/abdomen_ellipse"
DEVICE = "mps"
IMGSZ = 640
CONF = 0.70                # don't segment abdomen below this confidence
ABDOMEN_CLASS = 0          # index of the abdomen class (CHECK with model.names!)
KEEP_LARGEST_ONLY = True   # keep only the biggest abdomen (removes small-panel dupes)

MASK_COLOR = (0, 255, 0)   # BGR -> green: the raw segmentation mask
MASK_ALPHA = 0.35          # transparency of the mask fill (0=invisible, 1=solid)
MASK_EDGE_THICKNESS = 1    # outline of the raw mask (0 to hide the jagged edge)
ELLIPSE_COLOR = (0, 0, 255)  # BGR -> red: the fitted AC ellipse
ELLIPSE_THICKNESS = 2
# ----------------------------------------------------------------------------


def polygon_area(pts):
    return cv2.contourArea(pts.astype(np.float32))


def main():
    model = YOLO(MODEL_PATH)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = [p for p in Path(SOURCE_DIR).iterdir()
              if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if not images:
        print(f"No images found in {Path(SOURCE_DIR).resolve()}")
        return

    print(f"Processing {len(images)} image(s)...\n")

    for img_path in sorted(images):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  {img_path.name}: could not read, skipping")
            continue

        results = model.predict(
            source=str(img_path),
            device=DEVICE,
            imgsz=IMGSZ,
            conf=CONF,
            classes=[ABDOMEN_CLASS],
            retina_masks=True,
            verbose=False,
        )
        r = results[0]

        if r.masks is None or len(r.masks.xy) == 0:
            print(f"  {img_path.name}: no abdomen detected at conf>={CONF}")
            cv2.imwrite(str(out_dir / img_path.name), img)
            continue

        # candidate abdomen polygons (in original-image pixel coords)
        polys = [p.astype(np.int32) for p in r.masks.xy if len(p) >= 5]
        confs = [float(c) for c in r.boxes.conf]

        if not polys:
            print(f"  {img_path.name}: mask too small to fit an ellipse")
            cv2.imwrite(str(out_dir / img_path.name), img)
            continue

        if KEEP_LARGEST_ONLY:
            idx = int(np.argmax([polygon_area(p) for p in polys]))
            chosen = [polys[idx]]
            chosen_conf = [confs[idx]] if idx < len(confs) else [None]
        else:
            chosen = polys
            chosen_conf = confs

        # 1) draw the raw segmentation mask as a translucent fill
        overlay = img.copy()
        for poly in chosen:
            cv2.fillPoly(overlay, [poly], MASK_COLOR)
        cv2.addWeighted(overlay, MASK_ALPHA, img, 1 - MASK_ALPHA, 0, img)

        # 2) draw the mask edge (the actual jagged boundary) and the fitted ellipse
        for poly, cf in zip(chosen, chosen_conf):
            if MASK_EDGE_THICKNESS > 0:
                cv2.polylines(img, [poly], True, MASK_COLOR, MASK_EDGE_THICKNESS)
            ellipse = cv2.fitEllipse(poly)          # ((cx,cy),(MA,ma),angle)
            cv2.ellipse(img, ellipse, ELLIPSE_COLOR, ELLIPSE_THICKNESS)

        tag = f"{chosen_conf[0]:.2f}" if chosen_conf[0] is not None else "n/a"
        print(f"  {img_path.name}: mask + ellipse drawn (conf {tag})")
        cv2.imwrite(str(out_dir / img_path.name), img)

    print(f"\nDone. Mask + ellipse overlays saved in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()