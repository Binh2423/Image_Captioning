
import os
import argparse
import json
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch

# transformer feature extractor import: try the recommended AutoFeatureExtractor first
try:
    from transformers import AutoFeatureExtractor as FeatureExtractor  # newer versions
except Exception:
    try:
        from transformers import ViTFeatureExtractor as FeatureExtractor  # older API
    except Exception:
        FeatureExtractor = None

from transformers import ViTModel
from torch.cuda.amp import autocast

COMMON_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]

def list_image_keys_from_captions(captions_json):
    with open(captions_json, "r", encoding="utf-8") as f:
        caps = json.load(f)
    # caps might be dict mapping image_name -> captions OR list of items
    if isinstance(caps, dict):
        keys = list(caps.keys())
    elif isinstance(caps, list):
        # try to extract filename-like fields
        keys = []
        for item in caps:
            if isinstance(item, dict):
                for field in ("image", "img", "image_id", "file_name", "filename", "image_name"):
                    if field in item:
                        keys.append(str(item[field]))
                        break
                else:
                    # fallback to using index as a key
                    keys.append(str(len(keys)))
            else:
                keys.append(str(len(keys)))
    else:
        raise ValueError("Unsupported captions JSON structure; expected dict or list")
    return keys

def resolve_image_path(key, images_dir):
    # if key is an absolute path and exists, return it
    if os.path.isabs(key) and os.path.exists(key):
        return key
    # if key looks like a path relative to images_dir, try directly
    candidate = os.path.join(images_dir, key)
    if os.path.exists(candidate):
        return candidate
    # try common extensions appended
    base = os.path.splitext(os.path.basename(key))[0]
    for ext in COMMON_EXTS:
        candidate = os.path.join(images_dir, base + ext)
        if os.path.exists(candidate):
            return candidate
    # try searching for filenames that contain base substring (case-insensitive)
    try:
        for p in os.listdir(images_dir):
            if base.lower() in p.lower():
                candidate = os.path.join(images_dir, p)
                if os.path.isfile(candidate):
                    return candidate
    except Exception:
        pass
    return None

def load_image(path):
    try:
        img = Image.open(path).convert("RGB")
        return img
    except Exception:
        return None

def l2_normalize_rows(x: np.ndarray, eps: float = 1e-12):
    # x: (N, D)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = norms + eps
    return x / norms

def make_output_path(out_dir, img_key):
    safe_name = str(img_key).replace("/", "_").replace("\\", "_")
    return os.path.join(out_dir, safe_name + ".npz")

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    if FeatureExtractor is None:
        raise SystemExit("transformers FeatureExtractor not available; install transformers>=4.x")

    print(f"Loading feature extractor and model '{args.model_name_or_path}' ...")
    processor = FeatureExtractor.from_pretrained(args.model_name_or_path)
    model = ViTModel.from_pretrained(args.model_name_or_path)
    model.to(args.device)
    model.eval()

    keys = list_image_keys_from_captions(args.captions_json)
    total = len(keys)
    print(f"Found {total} image keys in captions json.")

    batch_size = max(1, int(args.batch_size))
    pbar = tqdm(range(0, total, batch_size), desc="Batches")
    skipped = 0
    written = 0
    for start in pbar:
        end = min(start + batch_size, total)
        batch_keys = keys[start:end]

        images = []
        out_paths = []
        resolved_keys = []
        for key in batch_keys:
            out_path = make_output_path(args.out_dir, key)
            if (not args.overwrite) and os.path.exists(out_path):
                # skip existing output
                skipped += 1
                continue
            img_path = resolve_image_path(key, args.images_dir)
            if img_path is None:
                print(f"Warning: could not resolve image for key '{key}', skipping")
                skipped += 1
                continue
            img = load_image(img_path)
            if img is None:
                print(f"Warning: failed to load image at {img_path}, skipping")
                skipped += 1
                continue
            images.append(img)
            out_paths.append(out_path)
            resolved_keys.append(key)

        if len(images) == 0:
            pbar.set_postfix(done=end, written=written, skipped=skipped)
            continue

        # preprocess (feature extractor -> pixel_values)
        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(args.device)

        # run model (with optional fp16)
        with torch.no_grad():
            if args.fp16 and args.device != "cpu":
                with autocast():
                    outputs = model(pixel_values)
            else:
                outputs = model(pixel_values)

        # last_hidden_state: (B, seq_len, hidden_dim)
        last_h = outputs.last_hidden_state  # tensor on device

        # convert to cpu numpy
        last_h = last_h.cpu().numpy().astype(np.float32)

        # include or drop cls token
        if args.include_cls:
            patch_feats_all = last_h  # (B, seq_len, D), CLS first token included
        else:
            patch_feats_all = last_h[:, 1:, :]  # (B, num_patches, D)

        # per-patch L2 normalize (row-wise) and save per image
        for j in range(patch_feats_all.shape[0]):
            feats = patch_feats_all[j]  # (P, D)
            if args.normalize:
                feats = l2_normalize_rows(feats)
            # ensure dtype
            feats = feats.astype(np.float32, copy=False)
            out_path = out_paths[j]
            # ensure parent directory exists
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            # save compressed npz with key 'patch_feats'
            np.savez_compressed(out_path, patch_feats=feats)
            written += 1

        pbar.set_postfix(done=end, written=written, skipped=skipped)

    print(f"Extraction finished. Written: {written}, Skipped: {skipped}. Features saved into: {args.out_dir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--images_dir", default="D:/DACN/dataset/Flickr8k/images",
                   help="Directory with images (joined with keys from captions JSON)")
    p.add_argument("--captions_json", default="D:/DACN/dataset/Flickr8k/Flickr8k_text/captions.json",
                   help="Captions json; keys enumerate image filenames")
    p.add_argument("--out_dir", default="D:/DACN/dataset/Flickr8k/features",
                   help="Directory to save per-image .npz files")
    p.add_argument("--model_name_or_path", default="google/vit-base-patch16-224-in21k",
                   help="ViT model id from Hugging Face")
    p.add_argument("--device", default="cuda", help="cuda or cpu")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--include_cls", action="store_true",
                   help="If set, include the [CLS] token in the saved patch_feats (first token).")
    p.add_argument("--fp16", action="store_true", help="Use autocast fp16 on CUDA to speed up and reduce memory.")
    p.add_argument("--overwrite", action="store_true",
                   help="If set, overwrite existing .npz outputs. Otherwise existing files are skipped.")
    p.add_argument("--normalize", action="store_true",
                   help="If set, L2-normalize each patch feature vector before saving (recommended).")
    args = p.parse_args()
    main(args)