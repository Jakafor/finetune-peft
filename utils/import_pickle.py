import pickle
path = "C:\\Users\\Jan Karl Forstner\\experiment\\data\\annotations\\label_mapping.pkl"

with open(path, "rb") as f:
    obj = pickle.load(f)   # DANGEROUS on untrusted files
print(obj)

# import_pickle.py
import numpy as np
from PIL import Image
from pathlib import Path
import argparse

def fix_rgb_masks_in_place(mask_dir, pattern="*.png", tol=16, dry_run=False, backup_ext=None):
    """
    Convert colored PNG masks in-place to single-channel ID masks:
      black→0, red→1 (good), blue→2 (bad), green→3 (medium).
    Already-correct VOC masks (modes 'L' or 'P') are skipped.

    Args:
        mask_dir (str | Path): folder with PNG masks.
        pattern (str): glob pattern, default '*.png'.
        tol (int): per-channel tolerance for matching colors.
        dry_run (bool): if True, don't write—just print what would happen.
        backup_ext (str | None): if set, save a backup copy, e.g. '.bak.png'.
    """
    mask_dir = Path(mask_dir)
    total = converted = skipped = 0

    def is_color_close(arr, rgb):
        return np.all(np.abs(arr - np.array(rgb, np.uint8)) <= tol, axis=-1)

    for p in sorted(mask_dir.glob(pattern)):
        total += 1
        with Image.open(p) as im:
            mode = im.mode
            if mode in ("L", "P", "I", "I;16"):
                skipped += 1
                continue
            a = np.asarray(im.convert("RGB"), dtype=np.uint8)

        out = np.zeros(a.shape[:2], dtype=np.uint8)
        out[is_color_close(a, (255,   0,   0))] = 1  # red   -> good
        out[is_color_close(a, (  0,   0, 255))] = 2  # blue  -> bad
        out[is_color_close(a, (  0, 255,   0))] = 3  # green -> medium
        # everything else stays 0 (background)

        if dry_run:
            print(f"[dry] {p.name}: unique IDs -> {sorted(np.unique(out))}")
            continue


        Image.fromarray(out, mode="L").save(p)
        converted += 1

    print(f"[mask-fix] total={total} converted={converted} skipped_as_ID={skipped}")

def main():
    ap = argparse.ArgumentParser(
        description="Overwrite RGB/RGBA color masks with ID masks "
                    "(black→0, red→1, blue→2, green→3). L/P/I masks are left as-is."
    )
    ap.add_argument("mask_dir", nargs="?", default=r"C:\Users\Jan Karl Forstner\experiment\data\voc_dataset\SegmentationClass")
    ap.add_argument("--pattern", default="*.png")
    ap.add_argument("--tol", type=int, default=16)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    fix_rgb_masks_in_place(a.mask_dir, pattern=a.pattern, tol=a.tol, dry_run=a.dry_run)

if __name__ == "__main__":
    main()