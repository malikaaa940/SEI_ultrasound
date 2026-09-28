import json
import os
import shutil
import random

# ------------------------------------------------------------------
# CHANGE 1: Reproducibility -- seed AND freeze the split.
#
#   SEED alone makes the split reproducible *given the same input files*.
#   FREEZE_SPLIT goes further: it writes a manifest mapping filename -> split
#   on the first run and reuses it on every later run, so the test set stays
#   fixed even if you add or relabel images. That is what makes the phrase
#   "held-out test set" defensible: the same images are always held out.
# ------------------------------------------------------------------
SEED = 42
FREEZE_SPLIT = True
MANIFEST_PATH = "split_manifest.json"   # filename -> "train"/"val"/"test"


def setup_yolo_dataset():
    json_paths = [
        "labels_my-project-name_2026-06-16-08-03-48.json",
        "labels_my-project-name_2026-06-16-06-25-35.json",
        "labels_my-project-name_2026-06-17-04-13-07.json",
        "labels_my-project-name_2026-06-18-01-31-23.json",
        "labels_my-project-name_2026-06-18-02-54-11.json",
        "labels_my-project-name_2026-06-18-05-47-35.json",
        "labels_my-project-name_2026-06-19-03-15-02.json"
    ]
    img_source_dir = "/Users/malikaeryomina/SEI_AI_ultrasound/Ultrasound_AI_Project/dataset/image_cropped"
    out_dir = "/Users/malikaeryomina/SEI_AI_ultrasound/dataset_yolo"

    print("Starting dataset conversion.")
    print(f"  seed={SEED}  freeze_split={FREEZE_SPLIT}  manifest={MANIFEST_PATH}")

    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(out_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "labels", split), exist_ok=True)

    all_images_info = []
    annotations_by_filename = {}
    class_names = []
    global_cat_map = {}

    for jpath in json_paths:
        if not os.path.exists(jpath):
            print(f"  -> Warning: Could not find {jpath}. Skipping.")
            continue

        with open(jpath, 'r') as f:
            data = json.load(f)

        if not class_names:
            class_names = [cat['name'] for cat in data['categories']]
            global_cat_map = {cat['name']: idx for idx, cat in enumerate(data['categories'])}
            print(f"Found classes: {class_names}")

        local_id_to_fname = {img['id']: img['file_name'] for img in data['images']}

        for img in data['images']:
            if not any(i['file_name'] == img['file_name'] for i in all_images_info):
                all_images_info.append({
                    'file_name': img['file_name'],
                    'width': img['width'],
                    'height': img['height']
                })

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

    # ------------------------------------------------------------------
    # Deterministic ordering BEFORE any shuffle. os.listdir / JSON order is
    # not guaranteed across machines or runs, so we sort first; only then does
    # a seed make the shuffle reproducible.
    # ------------------------------------------------------------------
    all_images_info.sort(key=lambda x: x['file_name'])

    # Decide the split assignment.
    assignment = {}   # file_name -> split
    if FREEZE_SPLIT and os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH) as f:
            assignment = json.load(f)
        print(f"Loaded frozen split from {MANIFEST_PATH} "
              f"({sum(v=='train' for v in assignment.values())} train / "
              f"{sum(v=='val' for v in assignment.values())} val / "
              f"{sum(v=='test' for v in assignment.values())} test).")

        # Any image not already in the manifest is new -> assign to train and
        # append to the manifest, so the frozen val/test sets never change.
        new_files = [im['file_name'] for im in all_images_info
                     if im['file_name'] not in assignment]
        if new_files:
            print(f"  -> {len(new_files)} new image(s) not in manifest; "
                  f"assigning to TRAIN so val/test stay frozen.")
            for fn in new_files:
                assignment[fn] = 'train'
            with open(MANIFEST_PATH, 'w') as f:
                json.dump(assignment, f, indent=2, sort_keys=True)
    else:
        # First run (or freezing disabled): seeded shuffle -> 80/10/10.
        rng = random.Random(SEED)                 # isolated RNG, not global state
        order = list(all_images_info)
        rng.shuffle(order)
        n = len(order)
        train_end = int(n * 0.75)
        val_end = int(n * 0.85)
        for i, im in enumerate(order):
            if i < train_end:
                assignment[im['file_name']] = 'train'
            elif i < val_end:
                assignment[im['file_name']] = 'val'
            else:
                assignment[im['file_name']] = 'test'
        if FREEZE_SPLIT:
            with open(MANIFEST_PATH, 'w') as f:
                json.dump(assignment, f, indent=2, sort_keys=True)
            print(f"Wrote frozen split manifest -> {MANIFEST_PATH}")

    info_by_name = {im['file_name']: im for im in all_images_info}
    train_images = [info_by_name[fn] for fn in assignment if assignment[fn] == 'train' and fn in info_by_name]
    val_images   = [info_by_name[fn] for fn in assignment if assignment[fn] == 'val'   and fn in info_by_name]
    test_images  = [info_by_name[fn] for fn in assignment if assignment[fn] == 'test'  and fn in info_by_name]

    def process_split(img_list, split_name):
        processed_count = 0
        for img in img_list:
            img_name = img['file_name']
            w, h = img['width'], img['height']

            src_img_path = os.path.join(img_source_dir, img_name)
            dst_img_path = os.path.join(out_dir, "images", split_name, img_name)
            txt_path = os.path.join(out_dir, "labels", split_name, img_name.replace('.jpg', '.txt'))

            if not os.path.exists(src_img_path):
                print(f"  -> Warning: Could not find {img_name} in {img_source_dir}. Skipping.")
                continue

            shutil.copy(src_img_path, dst_img_path)

            with open(txt_path, 'w') as txt_file:
                if img_name in annotations_by_filename:
                    for ann in annotations_by_filename[img_name]:
                        yolo_class_id = ann['category_id']
                        # NOTE: still using segmentation[0] -> keeps only the first
                        # polygon component. Disconnected structures lose extra parts.
                        # (Left as-is; out of scope for the reproducibility change.)
                        poly = ann['segmentation'][0]
                        norm_poly = []
                        for i in range(0, len(poly), 2):
                            norm_poly.append(str(poly[i] / w))
                            norm_poly.append(str(poly[i + 1] / h))
                        txt_file.write(f"{yolo_class_id} " + " ".join(norm_poly) + "\n")
            processed_count += 1
        print(f"Processed {processed_count} images for {split_name} set.")

    process_split(train_images, 'train')
    process_split(val_images, 'val')
    process_split(test_images, 'test')

    yaml_content = f"path: {os.path.abspath(out_dir)}\n"
    yaml_content += "train: images/train\n"
    yaml_content += "val: images/val\n"
    yaml_content += "test: images/test\n\n"
    yaml_content += "names:\n"
    for idx, name in enumerate(class_names):
        yaml_content += f"  {idx}: {name}\n"

    with open("dataset.yaml", "w") as f:
        f.write(yaml_content)

    print("\nSuccess. Dataset saved to 'dataset_yolo'.")
    print("The 'dataset.yaml' file has been updated")


if __name__ == "__main__":
    setup_yolo_dataset()