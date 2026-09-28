import cv2
import numpy as np
import os
from ultralytics import YOLO

# --- Configuration Paths ---
# Update these paths to match your local Apple M2 directories
MODEL_WEIGHTS = "/Users/malikaeryomina/runs/segment/train-12/weights/best.pt"
IMAGES_DIR = "/Users/malikaeryomina/SEI_AI_ultrasound/dataset_yolo/images/val"
OUTPUT_DIR = "/Users/malikaeryomina/SEI_AI_ultrasound/raw_vs_predict_outputs"

# Define distinct colors for your specific classes (BGR format for OpenCV)
CLASS_COLORS = {
    0: (255, 100, 100),  # Abdomen (Blue-ish)
    1: (100, 255, 100),  # Stomach (Green-ish)
    2: (100, 100, 255),  # Umbilical Vein (Red-ish)
    3: (0, 255, 255)  # Ribs (Yellow)
}


def draw_transparent_polygons(image, polygons, alpha=0.4):
    """Draws colored, semi-transparent prediction polygons on an image."""
    overlay = image.copy()

    for poly in polygons:
        class_id = poly['class_id']
        points = poly['points']
        color = CLASS_COLORS.get(class_id, (255, 255, 255))

        # Draw filled polygon on the overlay
        cv2.fillPoly(overlay, [points], color)
        # Draw a solid border
        cv2.polylines(overlay, [points], isClosed=True, color=color, thickness=2)

    # Blend the overlay with the original image
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

def main():
    # 1. Setup output directory and load the trained model
    print(f"Loading model weights from: {MODEL_WEIGHTS}")
    model = YOLO(MODEL_WEIGHTS)

    # NEW: Automatically extract the run name from your weights path
    # For example: extracts 'train' from '.../runs/segment/train/weights/best.pt'
    try:
        run_name = os.path.basename(os.path.dirname(os.path.dirname(MODEL_WEIGHTS)))
    except Exception:
        run_name = "custom_run"

    # Append the run name to your output directory path so it creates a unique subfolder
    current_run_output_dir = os.path.join(OUTPUT_DIR, run_name)
    os.makedirs(current_run_output_dir, exist_ok=True)
    print(f"Images from this run will be safely saved to: {current_run_output_dir}")

    # 2. Grab all ultrasound images from the test folder
    image_files = [f for f in os.listdir(IMAGES_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]
    print(f"Found {len(image_files)} test images. Starting side-by-side generation...")

    for img_name in image_files:
        img_path = os.path.join(IMAGES_DIR, img_name)

        # Load the base raw image
        raw_img = cv2.imread(img_path)
        if raw_img is None:
            print(f"Warning: Could not read {img_name}. Skipping.")
            continue

        # --- LEFT PANEL: COMPLETELY RAW IMAGE ---
        img_left = raw_img.copy()
        cv2.putText(img_left, "Raw Scan", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # --- RIGHT PANEL: AI PREDICTION IMAGE ---
        results = model(img_path, verbose=False)[0]
        pred_polygons = []

        if results.masks is not None:
            for box, mask in zip(results.boxes, results.masks.xy):
                class_id = int(box.cls[0])
                points = np.array(mask, dtype=np.int32)
                pred_polygons.append({'class_id': class_id, 'points': points})

        img_right = draw_transparent_polygons(raw_img.copy(), pred_polygons)
        cv2.putText(img_right, "AI Prediction", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)

        # --- HORIZONTAL STACKING ---
        combined_comparison = np.hstack((img_left, img_right))

        # MODIFIED: Save into the unique run subfolder instead of the main directory
        output_file_path = os.path.join(current_run_output_dir, f"split_{img_name}")
        cv2.imwrite(output_file_path, combined_comparison)
        print(f"Generated comparison for: {img_name}")

    print(f"\nSuccess! Images are safely stored without overwriting in:\n{current_run_output_dir}")




print(f"\nSuccess! All side-by-side presentation images are saved in:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()