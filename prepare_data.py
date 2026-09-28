import os
import shutil
import csv

SOURCE_DIR = "sei_data"
DEST_DIR = "Ultrasound_AI_Project"


IMAGES_DIR = os.path.join(DEST_DIR, "dataset", "images")
ANNOTATED_DIR = os.path.join(DEST_DIR, "dataset", "annotated_reference")
CSV_PATH = os.path.join(DEST_DIR, "metadata.csv")

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(ANNOTATED_DIR, exist_ok=True)

csv_data = []


# --- PROCESSING FUNCTION ---
def process_folder(folder_path, ground_truth_label):
    if not os.path.exists(folder_path):
        return

    for item in os.listdir(folder_path):
        file_path = os.path.join(folder_path, item)

        # Only process image files
        if os.path.isfile(file_path) and item.lower().endswith(('.jpg', '.jpeg', '.png')):


            # If the filename contains a hyphen, it is an annotated image
            if "-" in item:
                # Move to the safe isolation folder (do not log in CSV for training)
                dest_path = os.path.join(ANNOTATED_DIR, f"{ground_truth_label}_{item}")
                shutil.copy2(file_path, dest_path)

            else:
                new_filename = f"{ground_truth_label}_{item}"
                dest_path = os.path.join(IMAGES_DIR, new_filename)
                shutil.copy2(file_path, dest_path)

                # Log the clean data for the CSV
                csv_data.append({
                    "new_filename": new_filename,
                    "original_filename": item,
                    "ground_truth": ground_truth_label,
                    "original_location": folder_path
                })


process_folder(os.path.join(SOURCE_DIR, "Optimal AC examples"), "Optimal")
process_folder(os.path.join(SOURCE_DIR, "Suboptimal AC examples"), "Suboptimal")
process_folder(SOURCE_DIR, "Unlabeled")

# --- CSV GENERATION ---
with open(CSV_PATH, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=["new_filename", "original_filename", "ground_truth", "original_location"])
    writer.writeheader()
    writer.writerows(csv_data)

print(f"Success! Processed {len(csv_data)} clean training images.")
print(f"Descriptive annotations safely isolated in: {ANNOTATED_DIR}")
print(f"Metadata saved to: {CSV_PATH}")