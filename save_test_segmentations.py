from ultralytics import YOLO
from pathlib import Path
import cv2


MODEL_PATH = "/Users/malikaeryomina/runs/segment/train-14/weights/best.pt"

TEST_DIR   = Path("/Users/malikaeryomina/SEI_AI_ultrasound/dataset_yolo/images/test")
OUTPUT_DIR = Path("/Users/malikaeryomina/SEI_AI_ultrasound/test_segmentations")

DEVICE = "mps"
# ------------------------------------------------------------------------


def save_segmentations():
    if not TEST_DIR.exists():
        print(f"Error: test folder not found at {TEST_DIR}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(MODEL_PATH)

    image_files = sorted(
        p for p in TEST_DIR.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    if not image_files:
        print(f"No images found in {TEST_DIR}")
        return

    print(f"Found {len(image_files)} test images. Running segmentation...\n")

    for img_path in image_files:
        # run the model on this single image
        results = model.predict(
            source=str(img_path),
            device=DEVICE,
            save=False,      # we handle saving ourselves so we control the folder
            verbose=False,
        )
        result = results[0]

        # result.plot() returns the image with masks + boxes drawn on (BGR array)
        annotated = result.plot()

        out_path = OUTPUT_DIR / f"{img_path.stem}_segmented.jpg"
        cv2.imwrite(str(out_path), annotated)

        n = 0 if result.boxes is None else len(result.boxes)
        print(f"  {img_path.name}: {n} structures")

    print(f"\nDone. All {len(image_files)} segmentations are in: {OUTPUT_DIR}")


if __name__ == "__main__":
    save_segmentations()