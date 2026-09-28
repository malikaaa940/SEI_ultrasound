import cv2
import numpy as np
from ultralytics import YOLO
import math

# Import our custom logic engine
from metrics_types import SegmentationMetrics, ScoringResult
from quality_evaluator import calculate_basic_quality_score
import scoring_exceptions

# --- CONFIGURATION ---
CLASS_MAP = {
    "AC": 0,
    "Stomach": 1,
    "Vein": 2,
    "Rib": 3,
    "Spine": 4
}


def analyze_ultrasound(image_path: str, model_path: str) -> ScoringResult:
    print(f"Loading image: {image_path}")

    # Load the image to get total image area (Height x Width)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read the image file.")

    img_height, img_width = img.shape[:2]
    total_area = img_height * img_width

    # 1. INFERENCE
    print("Running YOLO AI...")
    model = YOLO(model_path)
    results = model.predict(source=image_path, conf=0.25, iou=0.1, save=True)
    result = results[0]

    # Initialize our tracking variables
    ac_area = 0
    rib_count = 0
    stomach_present = False
    vein_present = False
    ac_ellipse = None
    vein_ellipse = None

    # 2. EXTRACTION
    if result.masks is not None:
        # Loop through each detected object
        for i, box in enumerate(result.boxes):
            class_id = int(box.cls[0].item())

            # Retrieve the mask tensor for pixel-area calculations
            mask_tensor = result.masks.data[i].cpu().numpy()

            # Retrieve the pre-calculated contour coordinates from YOLO
            mask_polygon = result.masks.xy[i]

            # Map the detection to our metrics
            if class_id == CLASS_MAP["AC"]:
                # Count the number of 'True' or '1' pixels in the mask
                ac_area = int(np.sum(mask_tensor > 0.5))

                # Fit Ellipse directly to the YOLO polygon
                if len(mask_polygon) >= 5:
                    pts = np.array(mask_polygon, dtype=np.float32)
                    ac_ellipse = cv2.fitEllipse(pts)

            elif class_id == CLASS_MAP["Vein"]:
                vein_present = True

                # Fit Ellipse to the Vein
                if len(mask_polygon) >= 5:
                    pts = np.array(mask_polygon, dtype=np.float32)
                    vein_ellipse = cv2.fitEllipse(pts)

            elif class_id == CLASS_MAP["Rib"]:
                rib_count += 1

            elif class_id == CLASS_MAP["Stomach"]:
                stomach_present = True

    # --- DEBUG VISUALIZATION ---
    # Overlay the ideal ellipses onto a copy of the original image
    debug_img = img.copy()

    if ac_ellipse is not None:
        # Draw AC Ellipse in Green, 2px thickness
        cv2.ellipse(debug_img, ac_ellipse, (0, 255, 0), 2)

    if vein_ellipse is not None:
        # Draw Vein Ellipse in Blue, 2px thickness
        cv2.ellipse(debug_img, vein_ellipse, (255, 0, 0), 2)

    # Save the debug visualization
    debug_filename = f"fitted_{image_path.split('/')[-1]}"
    cv2.imwrite(debug_filename, debug_img)
    print(f"Debug visualization saved as: {debug_filename}")

    # 3. TRANSLATION
    print(
        f"Extraction Complete -> AC Area: {ac_area}, Ribs: {rib_count}, Vein: {vein_present}, Stomach: {stomach_present}")

    metrics = SegmentationMetrics(
        total_image_area=total_area,
        ac_area=ac_area,
        rib_count=rib_count,
        vein_present=vein_present,
        stomach_present=stomach_present,
        ac_ellipse=ac_ellipse,
        vein_ellipse=vein_ellipse
    )

    # 4. EVALUATION
    try:
        score_result = calculate_basic_quality_score(metrics)
        print("\n--- FINAL SCORING ---")
        print(f"Score: {score_result.final_score}/100")
        print(f"Quality Grade: {score_result.quality_label}")
        return score_result

    except Exception as e:
        print(f"\nPipeline Error: {str(e)}")
        return None


if __name__ == "__main__":
    # Example usage:
    TEST_IMAGE = "/Users/malikaeryomina/SEI_AI_ultrasound/Ultrasound_AI_Project/dataset/image_grayscale/Suboptimal_044.jpg"
    MODEL_WEIGHTS = "/Users/malikaeryomina/runs/segment/train-13/weights/best.pt"

    analyze_ultrasound(TEST_IMAGE, MODEL_WEIGHTS)