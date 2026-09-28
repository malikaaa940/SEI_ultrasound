import cv2
import numpy as np
import os

# --- Configuration Paths ---
# Point these to the specific folder you want to audit (train, val, or test)
IMAGES_DIR = "/Users/malikaeryomina/SEI_AI_ultrasound/dataset_yolo/images/train"
LABELS_DIR = "/Users/malikaeryomina/SEI_AI_ultrasound/dataset_yolo/labels/train"
OUTPUT_DIR = "/Users/malikaeryomina/SEI_AI_ultrasound/ground_truth_review"

# Define distinct colors for your specific classes (BGR format for OpenCV)
CLASS_COLORS = {
    0: (255, 100, 100),  # Abdomen (Blue-ish)
    1: (100, 255, 100),  # Stomach (Green-ish)
    2: (100, 100, 255),  # Umbilical Vein (Red-ish)
    3: (0, 255, 255)  # Ribs (Yellow)
}

CLASS_NAMES = {
    0: "Abdomen",
    1: "Stomach",
    2: "Umbilical Vein",
    3: "Ribs"
}


def load_ground_truth(label_path, img_width, img_height):
    """Parses YOLO text labels and converts normalized coordinates to pixel polygons."""
    polygons = []
    if not os.path.exists(label_path):
        return polygons  # Return empty if it's a background image (no labels)

    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 3: continue  # Skip malformed lines

            class_id = int(parts[0])
            # YOLO polygon format: class_id x1 y1 x2 y2 ... xn yn (normalized 0.0 - 1.0)
            coords = [float(x) for x in parts[1:]]
            points = []
            for i in range(0, len(coords), 2):
                x = int(coords[i] * img_width)
                y = int(coords[i + 1] * img_height)
                points.append([x, y])

            polygons.append({
                'class_id': class_id,
                'points': np.array(points, dtype=np.int32)
            })
    return polygons


def draw_transparent_polygons(image, polygons, alpha=0.4):
    """Draws colored, semi-transparent polygons on an image."""
    overlay = image.copy()

    for poly in polygons:
        class_id = poly['class_id']
        points = poly['points']
        color = CLASS_COLORS.get(class_id, (255, 255, 255))

        # Draw filled polygon on the overlay
        cv2.fillPoly(overlay, [points], color)
        # Draw a solid border
        cv2.polylines(overlay, [points], isClosed=True, color=color, thickness=2)

        # Calculate a rough center to place the text label
        M = cv2.moments(points)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            label_text = CLASS_NAMES.get(class_id, str(class_id))
            # Draw text with a black outline for readability
            cv2.putText(overlay, label_text, (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(overlay, label_text, (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Blend the overlay with the original image
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


def main():
    # 1. Setup output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. Grab all ultrasound images
    image_files = [f for f in os.listdir(IMAGES_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]
    print(f"Found {len(image_files)} images. Starting visual audit generation...")

    for img_name in image_files:
        img_path = os.path.join(IMAGES_DIR, img_name)
        label_path = os.path.join(LABELS_DIR, img_name.rsplit('.', 1)[0] + '.txt')

        # Load the base raw image
        raw_img = cv2.imread(img_path)
        if raw_img is None:
            continue
        height, width = raw_img.shape[:2]

        # --- LEFT PANEL: RAW IMAGE ---
        img_left = raw_img.copy()
        cv2.putText(img_left, "Raw Image", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # --- RIGHT PANEL: MANUAL ANNOTATIONS ---
        img_right = raw_img.copy()
        gt_polygons = load_ground_truth(label_path, width, height)

        if len(gt_polygons) > 0:
            img_right = draw_transparent_polygons(img_right, gt_polygons)
            cv2.putText(img_right, "MakeSense Annotations", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)
        else:
            cv2.putText(img_right, "MakeSense: NO LABELS (Background)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2, cv2.LINE_AA)

        # --- STACK AND SAVE ---
        combined_comparison = np.hstack((img_left, img_right))
        output_file_path = os.path.join(OUTPUT_DIR, f"audit_{img_name}")
        cv2.imwrite(output_file_path, combined_comparison)

    print(f"\nAudit complete! Review your annotations in:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()