import json
import os
import cv2
import numpy as np

# --- Configuration Paths ---
JSON_PATHS = [
    "labels_my-project-name_2026-06-16-08-03-48.json",
    "labels_my-project-name_2026-06-16-06-25-35.json",
    "labels_my-project-name_2026-06-17-04-13-07.json",
    "labels_my-project-name_2026-06-18-01-31-23.json",
    "labels_my-project-name_2026-06-18-02-54-11.json",
    "labels_my-project-name_2026-06-18-05-47-35.json",
    "labels_my-project-name_2026-06-19-03-15-02.json"
]
# Your master source folder containing all raw images
IMG_SOURCE_DIR = "/Users/malikaeryomina/SEI_AI_ultrasound/Ultrasound_AI_Project/dataset/image_grayscale"
OUTPUT_DIR = "/Users/malikaeryomina/SEI_AI_ultrasound/manual_segmentations"

# Define distinct colors for your specific classes (BGR format for OpenCV)
CLASS_COLORS = {
    0: (255, 100, 100),  # Abdomen (Blue-ish)
    1: (100, 255, 100),  # Stomach (Green-ish)
    2: (100, 100, 255),  # Umbilical Vein (Red-ish)
    3: (0, 255, 255)  # Ribs (Yellow)
}


def draw_transparent_polygons(image, segmentations, global_cat_map, alpha=0.4):
    """Draws all polygons from a dictionary of annotations onto an image."""
    overlay = image.copy()
    has_labels = False

    for ann in segmentations:
        has_labels = True
        color = CLASS_COLORS.get(ann['category_id'], (255, 255, 255))
        poly = ann['segmentation'][0]

        # Convert flat JSON array [x1, y1, x2, y2...] into an OpenCV usable matrix
        points = []
        for i in range(0, len(poly), 2):
            points.append([int(poly[i]), int(poly[i + 1])])

        pts_array = np.array(points, dtype=np.int32)
        cv2.fillPoly(overlay, [pts_array], color)
        cv2.polylines(overlay, [pts_array], isClosed=True, color=color, thickness=2)

    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0), has_labels


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Parsing all MakeSense JSON files...")

    # 1. Gather and consolidate metadata from all files
    annotations_by_filename = {}
    global_cat_map = {}

    for jpath in JSON_PATHS:
        if not os.path.exists(jpath):
            print(f"  -> Warning: {jpath} not found. Skipping.")
            continue

        with open(jpath, 'r') as f:
            data = json.load(f)

        # Establish global category maps to prevent indexing mismatches
        if not global_cat_map:
            global_cat_map = {cat['name']: idx for idx, cat in enumerate(data['categories'])}

        local_id_to_fname = {img['id']: img['file_name'] for img in data['images']}

        for ann in data['annotations']:
            fname = local_id_to_fname[ann['image_id']]
            if fname not in annotations_by_filename:
                annotations_by_filename[fname] = []

            local_cat_name = next(c['name'] for c in data['categories'] if c['id'] == ann['category_id'])
            global_id = global_cat_map[local_cat_name]

            annotations_by_filename[fname].append({
                'category_id': global_id,
                'segmentation': ann['segmentation']
            })

    # 2. Look inside the master image directory and process every image found in the JSONs
    processed_count = 0
    all_json_images = list(annotations_by_filename.keys())
    print(f"Identified {len(all_json_images)} unique annotated images to stitch.")

    for img_name in all_json_images:
        src_img_path = os.path.join(IMG_SOURCE_DIR, img_name)

        if not os.path.exists(src_img_path):
            continue

        raw_img = cv2.imread(src_img_path)
        if raw_img is None:
            continue

        # --- LEFT PANEL: COMPLETELY RAW SCAN ---
        img_left = raw_img.copy()
        cv2.putText(img_left, "Raw Ultrasound Scan", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)

        # --- RIGHT PANEL: ANNOTATED STRUCTURE ---
        img_right = raw_img.copy()
        img_right, contains_labels = draw_transparent_polygons(img_right, annotations_by_filename[img_name],
                                                               global_cat_map)

        if contains_labels:
            cv2.putText(img_right, "Ground Truth (MakeSense)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                        2, cv2.LINE_AA)
        else:
            cv2.putText(img_right, "Background / Negative Control", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                        2, cv2.LINE_AA)

        # --- CONCATENATION & STORAGE ---
        combined_view = np.hstack((img_left, img_right))
        output_path = os.path.join(OUTPUT_DIR, f"master_audit_{img_name}")
        cv2.imwrite(output_path, combined_view)
        processed_count += 1

    print(f"\nCompleted! Generated side-by-side images for all {processed_count} files in:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()