#!/usr/bin/env python3
"""
Grid search for beam decoding hyperparameters.

Usage example:
  python beam_grid_search.py \
    --val_captions DATA/captions_val.json \
    --features_dir DATA/features \
    --tokenizer_dir DATA/tokenizer \
    --ckpt PATH/to/best.pth \
    --output_csv beam_grid_results.csv \
    --beam_sizes 3 5 8 --length_penalties 0.6 0.7 0.9 --no_repeat_ngrams 0 3

This script will evaluate combinations of beam_size x length_penalty x no_repeat_ngram_size
and write results including sacreBLEU and NLTK BLEU-4.
"""
import os
import csv
import argparse
import time
from itertools import product

import numpy as np
import torch
from torch.utils.data import DataLoader

from tokenizer_utils import load_tokenizer, ids_to_text, normalize_detokenized
from dataset import CaptionPatchDataset, collate_fn
from model import DecoderLSTMWithAttention

from sacrebleu import corpus_bleu
from nltk.translate.bleu_score import corpus_bleu as nltk_corpus_bleu, SmoothingFunction
smooth = SmoothingFunction().method1

def compute_bleu4_nltk(hyps, list_of_refs):
    hyps_tok = [h.strip().split() for h in hyps]
    refs_tok = [[r.strip().split() for r in refs] for refs in list_of_refs]
    try:
        score = nltk_corpus_bleu(refs_tok, hyps_tok, weights=(0.25,0.25,0.25,0.25), smoothing_function=smooth)
        return float(score * 100.0)
    except Exception:
        return 0.0

def eval_with_beam(model, loader, tk, device, max_len=30, beam_size=3, length_penalty=1.0, no_repeat_ngram_size=0):
    model.eval()
    hyps = []
    refs = []
    with torch.no_grad():
        for feats, tgts, ids in loader:
            feats = feats.to(device)
            seqs = model.beam_decode(feats, tk["bos_id"], tk["eos_id"], max_len=max_len,
                                     beam_size=beam_size, length_penalty=length_penalty,
                                     no_repeat_ngram_size=no_repeat_ngram_size)
            seqs = seqs.cpu().numpy()
            for seq, img_id in zip(seqs, ids):
                hyp = normalize_detokenized(ids_to_text(seq, tk))
                raw_refs = loader.dataset.raw_caps.get(str(img_id), loader.dataset.raw_caps.get(img_id, [""]))
                if isinstance(raw_refs, str):
                    raw_refs = [raw_refs]
                ref_texts = []
                for r in raw_refs:
                    if isinstance(r, list):
                        ref_texts.append(normalize_detokenized(ids_to_text(r, tk)))
                    else:
                        ref_texts.append(normalize_detokenized(str(r)))
                hyps.append(hyp)
                refs.append(ref_texts)
    # sacrebleu
    max_refs = max(len(r) for r in refs) if refs else 1
    ref_streams = []
    for k in range(max_refs):
        ref_streams.append([r[k] if k < len(r) else r[0] for r in refs])
    sacre = corpus_bleu(hyps, ref_streams).score
    bleu4 = compute_bleu4_nltk(hyps, refs)
    return sacre, bleu4

def run_grid(args):
    device = torch.device(args.device)
    tk = load_tokenizer(args.tokenizer_dir)
    val_ds = CaptionPatchDataset(args.val_captions, args.features_dir, args.tokenizer_dir, max_len=args.max_len, max_patches=args.max_patches)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=max(1,args.num_workers//2), pin_memory=True)

    model = DecoderLSTMWithAttention(vocab_size=len(tk["vocab"]), embed_dim=args.embed_dim, feat_dim=768,
                                 hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout=args.dropout, pad_id=tk["pad_id"])
    model = model.to(device)
    if args.ckpt and os.path.exists(args.ckpt):
        sd = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(sd.get("model", sd), strict=False)
        print("Loaded checkpoint:", args.ckpt)

    beam_sizes = args.beam_sizes
    length_penalties = args.length_penalties
    no_repeat_ngrams = args.no_repeat_ngrams

    combos = list(product(beam_sizes, length_penalties, no_repeat_ngrams))
    print(f"Running {len(combos)} combos on val set (size={len(val_ds)})")

    results = []
    for beam_size, lp, ngram in combos:
        t0 = time.time()
        sacre, bleu4 = eval_with_beam(model, val_loader, tk, device, max_len=args.max_len, beam_size=beam_size, length_penalty=lp, no_repeat_ngram_size=ngram)
        dt = time.time() - t0
        print(f"beam={beam_size} lp={lp} ngram={ngram} => sacre={sacre:.3f}, bleu4={bleu4:.3f} (t={dt:.1f}s)")
        results.append({"beam_size": beam_size, "length_penalty": lp, "no_repeat_ngram_size": ngram, "sacrebleu": sacre, "bleu4": bleu4, "time_s": dt})

    # write csv
    keys = ["beam_size","length_penalty","no_repeat_ngram_size","sacrebleu","bleu4","time_s"]
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print("Grid search finished. Results written to", args.output_csv)
    # print best by bleu4
    best = max(results, key=lambda x: x["bleu4"])
    print("Best by BLEU-4:", best)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--val_captions", default="D:/DACN/dataset/Flickr8k/Flickr8k_text/captions_val.json", help="path to validation captions json file")
    p.add_argument("--features_dir", default="D:/DACN/dataset/Flickr8k/features", help="directory of image features")
    p.add_argument("--tokenizer_dir", default="D:/DACN/dataset/Flickr8k/tokenizer", help="directory of tokenizer")
    p.add_argument("--ckpt", default="D:/DACN/checkpoints_vit_lstm/best.pth", help="Path to XE checkpoint (best.pth) to initialize weights")
    p.add_argument("--output_csv", default="beam_grid_results.csv")
    p.add_argument("--beam_sizes", type=int, nargs="+", default=[3,5,8])
    p.add_argument("--length_penalties", type=float, nargs="+", default=[0.6,0.7,0.9])
    p.add_argument("--no_repeat_ngrams", type=int, nargs="+", default=[0,3])
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_len", type=int, default=30)
    p.add_argument("--embed_dim", type=int, default=512)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--max_patches", type=int, default=196)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    run_grid(args)