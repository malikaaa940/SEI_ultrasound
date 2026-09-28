import os
import csv
import cv2
import math
import numpy as np
from ultralytics import YOLO

from metrics_types import SegmentationMetrics
from quality_evaluator import calculate_basic_quality_score

CLASS_MAP = {"AC": 0, "Stomach": 1, "Vein": 2, "Rib": 3, "Spine": 4}


def calculate_shape_metrics(ellipse_tuple, mask_polygon):
    if ellipse_tuple is None or len(mask_polygon) < 3:
        return None

    (cx, cy), (dim1, dim2), angle = ellipse_tuple
    # We only need the major diameter to calculate the abdomen's radius for the vein depth check
    major_diameter = max(dim1, dim2)

    contour = np.array(mask_polygon, dtype=np.float32)
    actual_area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = actual_area / hull_area if hull_area > 0 else 0

    cv_circularity = (4 * math.pi * actual_area) / (perimeter ** 2) if perimeter > 0 else 0

    return {
        "center_x": cx,
        "center_y": cy,
        "major_diameter": round(major_diameter, 2),
        "cv_circularity": round(cv_circularity, 4),
        "solidity": round(solidity, 4)
    }


def calculate_vein_location(ac_metrics, vein_metrics):
    if not ac_metrics or not vein_metrics:
        return "N/A"

    ac_cx, ac_cy = ac_metrics["center_x"], ac_metrics["center_y"]
    vein_cx, vein_cy = vein_metrics["center_x"], vein_metrics["center_y"]

    pixel_distance = math.hypot(vein_cx - ac_cx, vein_cy - ac_cy)
    ac_radius = ac_metrics["major_diameter"] / 2.0

    if ac_radius == 0:
        return "N/A"

    return round(pixel_distance / ac_radius, 4)


def get_centroid(mask_polygon):
    """Calculates the exact center of mass of a segmentation using Image Moments."""
    contour = np.array(mask_polygon, dtype=np.float32)
    M = cv2.moments(contour)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return cx, cy
    return None, None


def get_clock_position(ref_x, ref_y, obj_x, obj_y):
    """Maps a spatial relationship between two points onto a standard 12-hour clock face."""
    if ref_x is None or obj_x is None:
        return "N/A"
    dx = obj_x - ref_x
    dy = obj_y - ref_y  # Image coordinates have +y pointing downwards

    angle = math.degrees(math.atan2(dy, dx))
    adjusted_angle = (angle + 90) % 360
    clock = round(adjusted_angle / 30)

    if clock == 0:
        clock = 12
    return f"{clock}:00"


def get_rib_length(mask_polygon):
    """Finds the maximum linear length of a curved rib using a Minimum Area Rectangle."""
    contour = np.array(mask_polygon, dtype=np.float32)
    if len(contour) >= 3:
        rect = cv2.minAreaRect(contour)
        width, height = rect[1]
        return round(max(width, height), 2)
    return 0.0


def batch_process_folder(image_dir: str, model_path: str, output_csv: str, output_image_dir: str):
    print(f"Loading YOLO AI from {model_path}...")
    model = YOLO(model_path)
    os.makedirs(output_image_dir, exist_ok=True)

    results_data = []
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(image_files)} images. Starting batch scoring...\n")

    for filename in image_files:
        image_path = os.path.join(image_dir, filename)
        img = cv2.imread(image_path)
        if img is None: continue

        total_area = img.shape[0] * img.shape[1]
        debug_img = img.copy()

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #actual scan area
        _, scan_mask = cv2.threshold(gray_img, 15, 255, cv2.THRESH_BINARY)
        actual_scan_area = cv2.countNonZero(scan_mask)

        #focus variance Laplacian edge detection
        focus_variance = cv2.Laplacian(gray_img, cv2.CV_64F).var()

        results = model.predict(source=image_path, conf=0.25, iou=0.1, verbose=False)
        result = results[0]

        ac_area, rib_count = 0, 0
        stomach_present, vein_present = False, False
        ac_conf, vein_conf, stomach_conf = 0.0, 0.0, 0.0
        rib_confs = []
        ac_ellipse, vein_ellipse = None, None
        ac_metrics, vein_metrics = None, None

        stomach_cx, stomach_cy = None, None
        rib_data = []

        if result.masks is not None:
            for i, box in enumerate(result.boxes):
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                mask_polygon = result.masks.xy[i]

                if class_id == CLASS_MAP["AC"]:
                    # COORDINATE-FRAME FIX: rasterise the AC polygon (masks.xy is in
                    # ORIGINAL full-res image coordinates) onto a canvas the size of
                    # gray_img, so ac_area and actual_scan_area live in the SAME frame.
                    # (Previously used masks.data[i], which is at model resolution ->
                    #  numerator and denominator were on different scales.)
                    ac_mask = np.zeros(gray_img.shape[:2], dtype=np.uint8)
                    ac_poly = np.array(mask_polygon, dtype=np.int32)
                    if len(ac_poly) >= 3:
                        cv2.fillPoly(ac_mask, [ac_poly.reshape(-1, 1, 2)], 1)
                    ac_area = int(ac_mask.sum())
                    ac_conf = confidence
                    if len(mask_polygon) >= 5:
                        ac_ellipse = cv2.fitEllipse(np.array(mask_polygon, dtype=np.float32))
                        ac_metrics = calculate_shape_metrics(ac_ellipse, mask_polygon)

                        # Just draw the clean fitted ellipse in Green
                        cv2.ellipse(debug_img, ac_ellipse, (0, 255, 0), 2)

                        # --- VISUALIZE CENTROID AND CIRCULARITY ---
                        if ac_metrics:
                            cx, cy = int(ac_metrics["center_x"]), int(ac_metrics["center_y"])

                            # Draw the centroid of the AC as a solid Green dot (-1 thickness fills it in)
                            cv2.circle(debug_img, (cx, cy), 5, (0, 255, 0), -1)

                            # Use half the major diameter as the "1.0" or 100% reference size
                            base_radius = int(ac_metrics["major_diameter"] / 2)

                            if base_radius > 0:
                                circularity_radius = int(base_radius * ac_metrics["cv_circularity"])

                                # Draw Circularity Circle in YELLOW
                                cv2.circle(debug_img, (cx, cy), circularity_radius, (0, 255, 255), 2)

                                # Overlay Circularity Text in top-left
                                circ_text = f"AC Circularity (Yellow): {ac_metrics['cv_circularity']:.2f}"
                                cv2.putText(debug_img, circ_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                            (0, 255, 255), 2)

                elif class_id == CLASS_MAP["Vein"]:
                    vein_present = True
                    vein_conf = confidence
                    if len(mask_polygon) >= 5:
                        vein_ellipse = cv2.fitEllipse(np.array(mask_polygon, dtype=np.float32))
                        vein_metrics = calculate_shape_metrics(vein_ellipse, mask_polygon)

                elif class_id == CLASS_MAP["Rib"]:
                    rib_count += 1
                    rib_confs.append(confidence)

                    rcx, rcy = get_centroid(mask_polygon)
                    length = get_rib_length(mask_polygon)
                    rib_data.append({"cx": rcx, "cy": rcy, "length": length})
                    if rcx is not None:
                        cv2.circle(debug_img, (rcx, rcy), 4, (0, 165, 255), -1)

                elif class_id == CLASS_MAP["Stomach"]:
                    stomach_present = True
                    stomach_conf = confidence

                    stomach_cx, stomach_cy = get_centroid(mask_polygon)
                    if stomach_cx is not None:
                        cv2.circle(debug_img, (stomach_cx, stomach_cy), 5, (0, 255, 255), -1)

        cv2.imwrite(os.path.join(output_image_dir, f"fitted_{filename}"), debug_img)

        stomach_clock = "N/A"
        if ac_metrics and stomach_cx is not None:
            stomach_clock = get_clock_position(ac_metrics["center_x"], ac_metrics["center_y"], stomach_cx, stomach_cy)

        rib_clocks = []
        rib_lengths = []
        for rd in rib_data:
            rib_lengths.append(rd["length"])
            if ac_metrics and rd["cx"] is not None:
                rc = get_clock_position(ac_metrics["center_x"], ac_metrics["center_y"], rd["cx"], rd["cy"])
                rib_clocks.append(rc)
            else:
                rib_clocks.append("N/A")

        metrics = SegmentationMetrics(
            total_image_area=total_area,
            actual_scan_area=actual_scan_area,
            focus_variance=focus_variance,
            ac_area=ac_area,
            rib_count=rib_count,
            vein_present=vein_present,
            stomach_present=stomach_present,
            ac_conf=ac_conf,
            vein_conf=vein_conf,
            stomach_conf=stomach_conf,
            rib_confs=rib_confs,
            ac_ellipse=ac_ellipse,
            vein_ellipse=vein_ellipse,
            stomach_clock=stomach_clock,
            rib_clocks=rib_clocks,
            rib_lengths=rib_lengths
        )

        try:
            vein_location_ratio = calculate_vein_location(ac_metrics, vein_metrics)

            # Parse parameters safely for the new scoring evaluator
            v_solidity = vein_metrics["solidity"] if vein_metrics else 0.0
            v_position = float(vein_location_ratio) if vein_location_ratio != "N/A" else 0.0
            longest_rib = max(rib_lengths) if rib_lengths else 0.0

            score_result = calculate_basic_quality_score(
                metrics=metrics,
                vein_solidity=v_solidity,
                vein_position=v_position,
                max_rib_len=longest_rib
            )

            rib_lengths_str = " | ".join(map(str, rib_lengths)) if rib_lengths else "N/A"
            rib_clocks_str = " | ".join(rib_clocks) if rib_clocks else "N/A"
            rib_confs_str = " | ".join([f"{c:.2f}" for c in rib_confs]) if rib_confs else "N/A"

            results_data.append({
                "Filename": filename, "Score": score_result.final_score, "Grade": score_result.quality_label,
                "AC_Area": ac_area,
                "AC_Confidence": ac_conf,
                "AC_CV_Circularity": ac_metrics["cv_circularity"] if ac_metrics else "N/A",
                "Ribs": rib_count,
                "Rib_Confidences": rib_confs_str,
                "Rib_Lengths": rib_lengths_str,
                "Rib_Clocks": rib_clocks_str,
                "Vein": vein_present,
                "Vein_Solidity": vein_metrics["solidity"] if vein_metrics else "N/A",
                "Vein_Position": vein_location_ratio,
                "Stomach": stomach_present,
                "Stomach_Clock": stomach_clock,
                "Error": ""
            })
            print(f"[{filename}] -> {score_result.final_score}/100")
        except Exception as e:
            print(f"[{filename}] -> ERROR: {str(e)}")

    with open(output_csv, mode='w', newline='') as file:
        # Cleaned headers list (no Axis Ratio or Diameters)
        headers = ["Filename", "Score", "Grade", "AC_Area", "AC_Confidence", "AC_CV_Circularity",
                   "Ribs", "Rib_Confidences", "Rib_Lengths", "Rib_Clocks",
                   "Vein", "Vein_Solidity", "Vein_Position",
                   "Stomach", "Stomach_Clock", "Error"]
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results_data)


if __name__ == "__main__":
    IMAGE_FOLDER = "/Users/malikaeryomina/SEI_AI_ultrasound/Ultrasound_AI_Project/dataset/image_cropped"
    MODEL_WEIGHTS = "/Users/malikaeryomina/runs/segment/train-13/weights/best.pt"
    OUTPUT_FILE = "/Users/malikaeryomina/SEI_AI_ultrasound/ai_scoring_report.csv"
    OUTPUT_IMAGE_DIR = "/Users/malikaeryomina/SEI_AI_ultrasound/fitted_shapes_output"

    batch_process_folder(IMAGE_FOLDER, MODEL_WEIGHTS, OUTPUT_FILE, OUTPUT_IMAGE_DIR)