#!/usr/bin/env python3
"""
Collect all "blank" images into a separate folder.

Scans SOURCE_DIR for image files whose name contains "blank"
(case-insensitive) and copies them into DEST_DIR.

Usage:
    python collect_blank_images.py                     # uses defaults below
    python collect_blank_images.py /path/to/source     # custom source
    python collect_blank_images.py /path/to/source /path/to/dest
"""

import sys
import shutil
from pathlib import Path

# --- Config -----------------------------------------------------------------
SOURCE_DIR = "Triplicate images"              # folder containing your images
DEST_DIR = "blank_images"     # folder to gather the blank images into
KEYWORD = "blank"             # substring to match in the filename
MOVE_INSTEAD_OF_COPY = False  # set True to move files instead of copying
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
# ----------------------------------------------------------------------------


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(SOURCE_DIR)
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(DEST_DIR)

    if not src.is_dir():
        print(f"Source folder does not exist: {src.resolve()}")
        sys.exit(1)

    dst.mkdir(parents=True, exist_ok=True)

    matches = [
        p for p in src.iterdir()
        if p.is_file()
        and KEYWORD.lower() in p.name.lower()
        and p.suffix.lower() in IMAGE_EXTS
    ]

    if not matches:
        print(f"No image files containing '{KEYWORD}' found in {src.resolve()}")
        return

    action = "Moving" if MOVE_INSTEAD_OF_COPY else "Copying"
    print(f"{action} {len(matches)} file(s) into {dst.resolve()}\n")

    for p in sorted(matches):
        target = dst / p.name
        if target.resolve() == p.resolve():
            continue  # skip if already in the destination
        if MOVE_INSTEAD_OF_COPY:
            shutil.move(str(p), str(target))
        else:
            shutil.copy2(str(p), str(target))
        print(f"  {p.name}")

    print(f"\nDone. {len(matches)} file(s) now in {dst.resolve()}")


if __name__ == "__main__":
    main()