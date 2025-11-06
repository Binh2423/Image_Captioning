#!/usr/bin/env python3
"""
prepare_flickr8k_from_csv_windows.py

Reads a CSV captions file with header including columns "image" and "caption",
groups captions per image, creates Flickr8k.token.txt and train/dev/test split lists,
and (optionally) copies/moves images into the dataset folder.

Example (PowerShell):
  python prepare_flickr8k_from_csv_windows.py `
    --images_dir "D:\DACN\data\images" `
    --captions_csv "D:\DACN\data\captions.txt" `
    --out_root "D:\DACN\data\dataset" `
    --train_ratio 0.8 --val_ratio 0.1 --seed 42 --copy_images

Notes:
 - The CSV is parsed with csv.DictReader (encoding utf-8-sig) so headers with BOM are handled.
 - The script matches image filenames by basename. If captions reference files not present in images_dir, a warning is printed.
"""
import os
import argparse
import csv
import random
import shutil
from collections import defaultdict

VALID_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

def list_images(images_dir):
    imgs = []
    for root, _, files in os.walk(images_dir):
        for f in files:
            if os.path.splitext(f.lower())[1] in VALID_EXTS:
                imgs.append((f, os.path.join(root, f)))
    # return dict basename -> fullpath (first occurrence)
    out = {}
    for name, full in imgs:
        if name not in out:
            out[name] = full
    return out

def read_captions_csv(csv_path, image_col_candidates=('image','img','filename'), caption_col_candidates=('caption','cap','text')):
    """
    Reads CSV and returns dict: basename -> list of captions
    Uses heuristics to find image and caption columns.
    """
    captions = defaultdict(list)
    if not os.path.exists(csv_path):
        print("Captions CSV not found:", csv_path)
        return captions

    with open(csv_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
        reader = csv.DictReader(f)
        # find columns
        headers = [h.strip().lower() for h in reader.fieldnames] if reader.fieldnames else []
        # determine image column
        img_col = None
        cap_col = None
        for c in reader.fieldnames or []:
            cl = c.strip().lower()
            if img_col is None and cl in image_col_candidates:
                img_col = c
            if cap_col is None and cl in caption_col_candidates:
                cap_col = c
        # fallback heuristics
        if img_col is None and reader.fieldnames:
            img_col = reader.fieldnames[0]
        if cap_col is None and reader.fieldnames and len(reader.fieldnames) > 1:
            cap_col = reader.fieldnames[1]
        if img_col is None or cap_col is None:
            # try simple split by comma per line as last resort
            f.seek(0)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',', 1)
                if len(parts) == 2:
                    img = os.path.basename(parts[0].strip())
                    cap = parts[1].strip()
                    captions[img].append(cap)
            return captions

        # parse rows
        for row in reader:
            img_raw = str(row.get(img_col, '')).strip()
            cap_raw = str(row.get(cap_col, '')).strip()
            if not img_raw:
                continue
            # normalize basename
            img_name = os.path.basename(img_raw)
            if img_name:
                captions[img_name].append(cap_raw)
    return captions

def write_list(file_path, items):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for it in items:
            f.write(it + '\n')

def write_token_file(token_path, captions_dict, images_present_set=None):
    """
    Writes Flickr8k.token.txt style:
      image.jpg#0<TAB>caption
    Only writes entries for images in images_present_set if provided.
    """
    os.makedirs(os.path.dirname(token_path), exist_ok=True)
    with open(token_path, 'w', encoding='utf-8') as f:
        for img in sorted(captions_dict.keys()):
            if images_present_set is not None and img not in images_present_set:
                continue
            caps = captions_dict[img]
            for i, c in enumerate(caps):
                f.write(f"{img}#{i}\t{c}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', required=True, help='Folder containing images (e.g. D:\\DACN\\data\\images)')
    parser.add_argument('--captions_csv', required=True, help='CSV file with columns image,caption (e.g. D:\\DACN\\data\\captions.txt)')
    parser.add_argument('--out_root', default='dataset', help='Output root (will create out_root\\Flickr8k_text and out_root\\Flickr_8k_Dataset)')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--move_images', action='store_true', help='Move images into out_root\\Flickr_8k_Dataset (instead of only writing lists)')
    parser.add_argument('--copy_images', action='store_true', help='Copy images into out_root\\Flickr_8k_Dataset')
    args = parser.parse_args()

    images_map = list_images(args.images_dir)  # basename -> fullpath
    if not images_map:
        print("No images found in", args.images_dir)
        return
    print(f"Found {len(images_map)} images (unique basenames) under {args.images_dir}")

    captions = read_captions_csv(args.captions_csv)
    if captions:
        print(f"Parsed captions for {len(captions)} distinct basenames from {args.captions_csv}")
    else:
        print("No captions parsed from CSV; proceeding with images only (no token file will be created).")

    # Optionally filter to images that have captions
    if captions:
        # find intersection of basenames
        images_with_caps = [img for img in images_map.keys() if img in captions]
        if len(images_with_caps) == 0:
            print("Warning: none of the image basenames in captions CSV matched files in images_dir.")
            # continue but write token file only for parsed captions (images not present)
            images_list = list(images_map.keys())
        else:
            images_list = images_with_caps
            print(f"Using {len(images_list)} images that have captions (others skipped).")
    else:
        images_list = list(images_map.keys())

    # deterministic shuffle & splits
    random.seed(args.seed)
    random.shuffle(images_list)
    total = len(images_list)
    n_train = int(total * args.train_ratio)
    n_val = int(total * args.val_ratio)
    n_test = total - n_train - n_val
    train = images_list[:n_train]
    val = images_list[n_train:n_train+n_val]
    test = images_list[n_train+n_val:]
    print(f"Split sizes -> train: {len(train)}, val: {len(val)}, test: {len(test)} (total {total})")

    text_dir = os.path.join(args.out_root, 'Flickr8k_text')
    img_out_dir = os.path.join(args.out_root, 'Flickr_8k_Dataset')
    os.makedirs(text_dir, exist_ok=True)
    os.makedirs(img_out_dir, exist_ok=True)

    # write split lists (filenames only)
    write_list(os.path.join(text_dir, 'Flickr_8k.trainImages.txt'), train)
    write_list(os.path.join(text_dir, 'Flickr_8k.devImages.txt'), val)
    write_list(os.path.join(text_dir, 'Flickr_8k.testImages.txt'), test)
    print("Wrote split files to", text_dir)

    # write token file only for images that exist in images_map (so training loader can find them)
    if captions:
        token_path = os.path.join(text_dir, 'Flickr8k.token.txt')
        write_token_file(token_path, captions, images_present_set=set(images_map.keys()))
        print("Wrote token file to", token_path)

    # optionally copy or move images
    if args.move_images or args.copy_images:
        action = "move" if args.move_images else "copy"
        count = 0
        for im in images_list:
            src = images_map.get(im)
            if src is None:
                print("Warning: image listed but not found:", im)
                continue
            dst = os.path.join(img_out_dir, im)
            if os.path.abspath(src) == os.path.abspath(dst):
                count += 1
                continue
            try:
                if args.move_images:
                    shutil.move(src, dst)
                else:
                    shutil.copy2(src, dst)
                count += 1
            except Exception as e:
                print("Error during", action, im, ":", e)
        print(f"{action.title()}d {count} images to {img_out_dir}")
    else:
        print("Images not moved or copied. If you want images in the dataset folder, re-run with --move_images or --copy_images.")

    print("Done. Point your training script --data_root to:", args.out_root)

if __name__ == '__main__':
    main()