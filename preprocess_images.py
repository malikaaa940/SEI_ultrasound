import os
import cv2
import numpy as np


DEST_DIR    = 'Ultrasound_AI_Project'
IMAGES_DIR  = os.path.join(DEST_DIR, 'dataset', 'images')          # pristine source (untouched)
CROP_DIR    = os.path.join(DEST_DIR, 'dataset', 'image_cropped')   # cropped + grayscale output
REVIEW_DIR  = os.path.join(DEST_DIR, 'dataset', 'crop_failures')   # full images that need hand-cropping

os.makedirs(CROP_DIR, exist_ok=True)
os.makedirs(REVIEW_DIR, exist_ok=True)


# --- AC ELLIPSE DETECTION + CROP ---
# --- NEW: ULTRASOUND CONE DETECTION + CROP ---
def detect_ultrasound_cone(gray):
    """Finds the general ultrasound scan area (the gray cone) and returns a bounding box."""
    h_img, w_img = gray.shape

    # Threshold at 15 instead of 220.
    # This captures all tissue (gray/white) and ignores the empty background (black).
    _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

    # Apply a slight blur to smooth out noise before finding contours
    mask = cv2.medianBlur(mask, 15)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not cnts:
        return None

    # Assume the largest non-black blob on the screen is the ultrasound scan itself
    largest_contour = max(cnts, key=cv2.contourArea)

    # If the largest blob is tiny, the image is mostly blank/corrupted
    if cv2.contourArea(largest_contour) < 0.1 * h_img * w_img:
        return None

    # Draw a bounding box around the entire scan area
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Add a small 20-pixel safety margin so we don't accidentally shave off the edges
    margin = 20
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(w_img, x + w + margin)
    y2 = min(h_img, y + h + margin)

    return x1, y1, x2, y2


# --- PROCESS ONE IMAGE ---
def process(filename):
    image = cv2.imread(os.path.join(IMAGES_DIR, filename))
    if image is None:
        return 'unreadable'

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    box = detect_ultrasound_cone(gray)

    if box is None:
        # Save the full grayscale image so nothing is lost
        cv2.imwrite(os.path.join(REVIEW_DIR, filename), gray)
        return 'failed'

    x1, y1, x2, y2 = box
    cropped = gray[y1:y2, x1:x2]
    cv2.imwrite(os.path.join(CROP_DIR, filename), cropped)
    return 'ok'


# --- EXECUTION ---
ok = failed = unreadable = 0
for file in os.listdir(IMAGES_DIR):
    if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    result = process(file)
    if result == 'ok':
        ok += 1
    elif result == 'failed':
        failed += 1
        print(f'CROP FAILED (sent to crop_failures/): {file}')
    else:
        unreadable += 1
        print(f'UNREADABLE: {file}')

print('\nProcessing complete!')
print(f'  Cropped successfully : {ok}  ->  {CROP_DIR}')
print(f'  Need hand-cropping   : {failed}  ->  {REVIEW_DIR}')
print(f'  Unreadable files     : {unreadable}')