#!/usr/bin/env python3
"""
CaptionPatchDataset with collate_fn.
Encodes captions on-the-fly using tokenizer_utils.text_to_ids (consistent with vocab.json).
Loads per-image features (.npy/.npz/.pt/.pth) and returns (feats_tensor, tgt_tensor, img_id).
"""
import os
import json
import numpy as np
from typing import Dict, List, Any, Tuple
from pathlib import Path

import torch
from torch.utils.data import Dataset

from tokenizer_utils import load_tokenizer, text_to_ids

def _load_json_captions(path: str) -> Dict[str, List[str]]:
    data = json.load(open(path, "r", encoding="utf-8"))
    mapping = {}
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, list):
                mapping[str(k)] = [str(x).strip() for x in v]
            else:
                mapping[str(k)] = [str(v).strip()]
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            caption = None
            img_id = None
            for key in ("caption", "captions", "text"):
                if key in item:
                    caption = item[key]
                    break
            for key in ("image", "img", "image_id", "file_name", "filename"):
                if key in item:
                    img_id = item[key]
                    break
            if img_id is None:
                img_id = str(idx)
            if isinstance(caption, list):
                mapping[str(img_id)] = [str(x).strip() for x in caption]
            elif caption is not None:
                mapping[str(img_id)] = [str(caption).strip()]
    else:
        raise ValueError("Unsupported captions JSON structure: must be dict or list")
    return mapping

class CaptionPatchDataset(Dataset):
    def __init__(self, captions_json: str, features_dir: str, tokenizer_dir: str,
                 max_len: int = 33, max_patches: int = 196,
                 feature_norm: bool = True, dtype = np.float32):
        super().__init__()
        self.captions_json = captions_json
        self.features_dir = features_dir
        self.tokenizer_dir = tokenizer_dir
        self.max_len = max_len
        self.max_patches = max_patches
        self.feature_norm = feature_norm
        self.dtype = dtype

        self.raw_caps = _load_json_captions(captions_json)
        self.tk = load_tokenizer(tokenizer_dir)
        if self.tk.get("vocab") is None:
            raise SystemExit("vocab.json not found in tokenizer_dir: " + tokenizer_dir)

        self.caps: Dict[str, List[List[int]]] = {}
        pad_id = self.tk["pad_id"]
        bos_id = self.tk["bos_id"]
        eos_id = self.tk["eos_id"]

        for img_id, caps in self.raw_caps.items():
            encs = []
            for c in caps:
                txt = c.strip()
                ids = text_to_ids(txt, self.tk)
                ids = ids[:(self.max_len - 2)]
                ids = [bos_id] + ids + [eos_id]
                if len(ids) < self.max_len:
                    ids = ids + [pad_id] * (self.max_len - len(ids))
                encs.append(ids)
            if not encs:
                ids = [bos_id, eos_id] + [pad_id] * (self.max_len - 2)
                encs = [ids]
            self.caps[str(img_id)] = encs

        self.ids = list(self.caps.keys())

    def __len__(self):
        return len(self.ids)

    def _find_feature_file(self, img_id: str):
        base = os.path.join(self.features_dir, str(img_id))
        for ext in [".npy", ".npz", ".pt", ".pth"]:
            p = base + ext
            if os.path.exists(p):
                return p
        # try matching by substring
        for p in Path(self.features_dir).glob(f"*{img_id}*"):
            if p.is_file():
                return str(p)
        return None

    def _load_feature_array(self, path: str):
        ext = Path(path).suffix.lower()
        if ext == ".npy":
            arr = np.load(path)
        elif ext == ".npz":
            data = np.load(path)
            try:
                arr = data["patch_feats"]
            except Exception:
                try:
                    arr = data["arr_0"]
                except Exception:
                    keys = [k for k in data.files]
                    arr = data[keys[0]]
        elif ext in (".pt", ".pth"):
            obj = torch.load(path, map_location="cpu")
            if isinstance(obj, dict) and "features" in obj:
                arr = obj["features"]
            else:
                arr = obj
            if isinstance(arr, torch.Tensor):
                arr = arr.numpy()
        else:
            raise FileNotFoundError(f"Unsupported feature file type: {path}")
        return arr

    def _normalize_features(self, feats: np.ndarray):
        if feats.ndim == 1:
            feats = feats[np.newaxis, :]
        if feats.ndim == 3:
            feats = np.squeeze(feats)
            if feats.ndim == 1:
                feats = feats[np.newaxis, :]
        N, D = feats.shape[0], feats.shape[1]
        if N > self.max_patches:
            feats = feats[:self.max_patches, :]
        elif N < self.max_patches:
            pad = np.zeros((self.max_patches - N, D), dtype=feats.dtype)
            feats = np.vstack([feats, pad])
        feats = feats.astype(self.dtype, copy=False)
        if self.feature_norm:
            norms = np.linalg.norm(feats, axis=1, keepdims=True)
            norms = norms + 1e-12
            feats = feats / norms
        return feats

    def __getitem__(self, idx: int):
        img_id = self.ids[idx]
        fpath = self._find_feature_file(img_id)
        if fpath is None:
            raise FileNotFoundError(f"No feature file for image id {img_id} in {self.features_dir}")
        arr = self._load_feature_array(fpath)
        feats = self._normalize_features(arr)
        feats_tensor = torch.from_numpy(feats).float()
        caps_for_img = self.caps[img_id]
        tgt_ids = caps_for_img[np.random.randint(len(caps_for_img))]
        tgt_tensor = torch.tensor(tgt_ids, dtype=torch.long)
        return feats_tensor, tgt_tensor, img_id

def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, Any]]):
    import torch as _torch
    feats_list = [item[0] for item in batch]
    tgts_list = [item[1] for item in batch]
    ids = [item[2] for item in batch]
    feats_list = [_torch.as_tensor(f).float() for f in feats_list]
    tgts_list = [_torch.as_tensor(t).long() for t in tgts_list]
    feats_batch = _torch.stack(feats_list, dim=0)
    tgts_batch = _torch.stack(tgts_list, dim=0)
    return feats_batch, tgts_batch, ids