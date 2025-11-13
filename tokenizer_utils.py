
import json
import os
import re
from typing import List, Dict, Any

try:
    import sentencepiece as spm
    _HAS_SP = True
except Exception:
    _HAS_SP = False

def load_tokenizer(tokenizer_dir: str) -> Dict[str, Any]:
    tk = {"sp": None, "vocab": None, "rev": None, "pad_id": None, "bos_id": None, "eos_id": None, "unk_id": None}
    vocab_path = os.path.join(tokenizer_dir, "vocab.json")
    rev_path = os.path.join(tokenizer_dir, "rev_vocab.json")
    sp_model_path = os.path.join(tokenizer_dir, "sp.model")

    if os.path.exists(vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            tk["vocab"] = json.load(f)
    if os.path.exists(rev_path):
        with open(rev_path, "r", encoding="utf-8") as f:
            tk["rev"] = json.load(f)
    if _HAS_SP and os.path.exists(sp_model_path):
        sp = spm.SentencePieceProcessor()
        sp.load(sp_model_path)
        tk["sp"] = sp

    if tk["vocab"]:
        tk["pad_id"] = int(tk["vocab"].get("<pad>", 0))
        tk["bos_id"] = int(tk["vocab"].get("<bos>", 1))
        tk["eos_id"] = int(tk["vocab"].get("<eos>", 2))
        tk["unk_id"] = int(tk["vocab"].get("<unk>", 3))
    return tk

def normalize_detokenized(s: str) -> str:
   
    if s is None:
        return ""
   
    s = str(s)

  
    s = re.sub(r'(\s*[.?!,:;]\s*){2,}', lambda m: m.group(0).strip().split()[0], s)

   
    s = re.sub(r'\s+([.?!,:;])', r'\1', s)

   
    s = re.sub(r'\b(\w+)(?:\s+\1){2,}\b', r'\1', s, flags=re.IGNORECASE)

   
    s = re.sub(r'([.?!]){2,}', r'\1', s)

   
    s = re.sub(r'\s+', ' ', s).strip()

   
    s = re.sub(r'(\s+[.?!])+$', lambda m: m.group(0).strip(), s).strip()

    return s

def ids_to_text(ids, tk: Dict[str, Any]) -> str:
   
    try:
        import torch
        if isinstance(ids, torch.Tensor):
            ids_list = ids.detach().cpu().tolist()
        else:
            ids_list = list(ids)
    except Exception:
        ids_list = list(ids)

    rev = tk.get("rev", {})
   
    pad = tk.get("pad_id")
    bos = tk.get("bos_id")
    eos = tk.get("eos_id")
    unk = tk.get("unk_id")

    pieces = []
    for i in ids_list:
        try:
            ii = int(i)
        except Exception:
            continue
        if ii in (pad, bos, eos, unk):
            continue
        piece = rev.get(str(ii), "")
        if piece:
            pieces.append(piece)

    if not pieces:
        return ""

    sp = tk.get("sp", None)
    if sp is not None:
        try:
            text = sp.decode_pieces(pieces).strip()
        except Exception:
            text = "".join(pieces).replace("▁", " ").strip()
    else:
        text = "".join(pieces).replace("▁", " ").strip()

    text = normalize_detokenized(text)
    return text

def text_to_ids(text: str, tk: Dict[str, Any]) -> List[int]:
  
    sp = tk.get("sp", None)
    vocab = tk.get("vocab", {})
    unk_id = tk.get("unk_id", 3)

    if sp is not None:
        pieces = sp.encode_as_pieces(text)
        ids = []
        for p in pieces:
            vid = vocab.get(p, None)
            if vid is None:
                vid = vocab.get(p.strip(), None)
            ids.append(int(vid) if vid is not None else int(unk_id))
        return ids

    toks = text.strip().split()
    return [int(vocab.get(t, unk_id)) for t in toks]