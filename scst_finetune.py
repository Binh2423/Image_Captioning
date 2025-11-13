
import os
import json
import time
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tokenizer_utils import load_tokenizer, ids_to_text, normalize_detokenized
from dataset import CaptionPatchDataset, collate_fn
from model import DecoderLSTMWithAttention

try:
    from pycocoevalcap.cider.cider import Cider
    _HAS_CIDER = True
except Exception:
    _HAS_CIDER = False

try:
    import sacrebleu
    _HAS_SACREBLEU = True
except Exception:
    _HAS_SACREBLEU = False

def compute_cider_scores(hyps, list_of_refs):
   
    if not _HAS_CIDER:
        raise RuntimeError("pycocoevalcap.cider not available")
    refs_dict = {}
    hyps_dict = {}
    for i, (h, rlist) in enumerate(zip(hyps, list_of_refs)):
        refs_dict[i] = rlist
        hyps_dict[i] = [h]
    cider = Cider()
    score, scores = cider.compute_score(refs_dict, hyps_dict)
    return np.array(scores, dtype=np.float32)

def compute_sentence_sacrebleu(hyps, list_of_refs):
    scores = []
    for h, refs in zip(hyps, list_of_refs):
        try:
           
            sc = sacrebleu.sentence_bleu(h, refs).score
        except Exception:
          
            if _HAS_SACREBLEU:
                ref_streams = [[r] for r in refs]
                sc = sacrebleu.corpus_bleu([h], [ [r for r in refs] ]).score
            else:
                sc = 0.0
        scores.append(float(sc))
    return np.array(scores, dtype=np.float32)

def compute_rewards(hyps, list_of_refs, metric="cider"):
    if metric == "cider" and _HAS_CIDER:
        return compute_cider_scores(hyps, list_of_refs)
   
    if _HAS_SACREBLEU:
        return compute_sentence_sacrebleu(hyps, list_of_refs)
   
    return np.zeros(len(hyps), dtype=np.float32)

def detok_and_normalize_seqids(seq_ids, tk):
   
    text = ids_to_text(seq_ids, tk)
    return normalize_detokenized(text)

def evaluate_for_logging(model, loader, tk, device, max_len=30, use_beam=False, beam_size=3, beam_length_penalty=1.0, no_repeat_ngram_size=0):
   
    model.eval()
    hyps = []
    refs = []
    samples = []
    with torch.no_grad():
        for feats, tgts, ids in loader:
            feats = feats.to(device)
            if use_beam:
                seqs = model.beam_decode(feats, tk["bos_id"], tk["eos_id"], max_len=max_len,
                                         beam_size=beam_size, length_penalty=beam_length_penalty,
                                         no_repeat_ngram_size=no_repeat_ngram_size)
            else:
                seqs = model.greedy_decode(feats, tk["bos_id"], tk["eos_id"], max_len=max_len)
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
                samples.append({"image_id": str(img_id), "pred": hyp, "refs": ref_texts})
   
    try:
        from sacrebleu import corpus_bleu
        max_refs = max(len(r) for r in refs) if refs else 1
        ref_streams = []
        for k in range(max_refs):
            ref_streams.append([r[k] if k < len(r) else r[0] for r in refs])
        sacre = corpus_bleu(hyps, ref_streams).score
    except Exception:
        sacre = 0.0
    return sacre, samples

def run_scst(args):
    device = torch.device(args.device)
    tk = load_tokenizer(args.tokenizer_dir)
    if tk.get("vocab") is None:
        raise SystemExit("tokenizer vocab.json missing in tokenizer_dir")

    train_ds = CaptionPatchDataset(args.train_captions, args.features_dir, args.tokenizer_dir, max_len=args.max_len, max_patches=args.max_patches)
    val_ds = CaptionPatchDataset(args.val_captions, args.features_dir, args.tokenizer_dir, max_len=args.max_len, max_patches=args.max_patches)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=max(1,args.num_workers//2), pin_memory=True)

    model = DecoderLSTMWithAttention(vocab_size=len(tk["vocab"]), embed_dim=args.embed_dim, feat_dim=args.feat_dim,
                                     hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout=args.dropout, pad_id=tk["pad_id"])
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 0
    if args.ckpt and os.path.exists(args.ckpt):
        sd = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(sd.get("model", sd), strict=False)
        print("Loaded checkpoint:", args.ckpt)
        

    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        epoch_loss = 0.0
        n_batches = 0
        for feats, tgts, ids in train_loader:
            feats = feats.to(device)
            
            seqs_sampled, logp_sum = model.sample_decode(feats, tk["bos_id"], tk["eos_id"], max_len=args.max_len, temperature=args.sample_temperature)
            seqs_greedy = model.greedy_decode(feats, tk["bos_id"], tk["eos_id"], max_len=args.max_len)

          
            bs = seqs_sampled.size(0)
            sampled_texts = [normalize_detokenized(ids_to_text(seqs_sampled[i].cpu().numpy(), tk)) for i in range(bs)]
            greedy_texts  = [normalize_detokenized(ids_to_text(seqs_greedy[i].cpu().numpy(), tk)) for i in range(bs)]

           
            refs_for_batch = []
            for img_id in ids:
                raw_refs = train_ds.raw_caps.get(str(img_id), train_ds.raw_caps.get(img_id, [""]))
                if isinstance(raw_refs, str):
                    raw_refs = [raw_refs]
                ref_texts = []
                for r in raw_refs:
                    if isinstance(r, list):
                        ref_texts.append(normalize_detokenized(ids_to_text(r, tk)))
                    else:
                        ref_texts.append(normalize_detokenized(str(r)))
                refs_for_batch.append(ref_texts)

            
            rewards_sample = compute_rewards(sampled_texts, refs_for_batch, metric=args.reward)
            rewards_greedy = compute_rewards(greedy_texts, refs_for_batch, metric=args.reward)

            
            adv = rewards_sample - rewards_greedy
            
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

           
            adv_t = torch.tensor(adv, dtype=torch.float32, device=device)
            logp_sum = logp_sum.to(device)
            scst_loss = - (adv_t * logp_sum).mean()

            
            if args.xe_weight > 0.0:
               
                inp = tgts.to(device)[:, :-1]
                target = tgts.to(device)[:, 1:]
                logits = model(feats, inp)
                loss_fn = nn.CrossEntropyLoss(ignore_index=tk["pad_id"])
                xe_loss = loss_fn(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
                loss = args.xe_weight * xe_loss + (1.0 - args.xe_weight) * scst_loss
            else:
                loss = scst_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += float(loss.item())
            n_batches += 1

        epoch_time = time.time() - t0
        avg_loss = epoch_loss / max(1, n_batches)
      
        sacre, samples = evaluate_for_logging(model, val_loader, tk, device, max_len=args.max_len,
                                              use_beam=args.use_beam, beam_size=args.beam_size,
                                              beam_length_penalty=args.beam_length_penalty,
                                              no_repeat_ngram_size=args.no_repeat_ngram_size)
        
        ckpt_path = os.path.join(args.output_dir, f"scst_epoch{epoch}.pth")
        torch.save({"epoch": epoch, "model": model.state_dict(), "best": sacre}, ckpt_path)
        print(f"SCST Epoch {epoch} done in {epoch_time:.1f}s, avg_loss={avg_loss:.4f}, val_sacrebleu={sacre:.4f}, saved {ckpt_path}")

    print("SCST fine-tuning finished.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_captions", default="D:/DACN/dataset/Flickr8k/Flickr8k_text/captions_train.json", help="path to train captions json file")
    p.add_argument("--val_captions", default="D:/DACN/dataset/Flickr8k/Flickr8k_text/captions_val.json", help="path to validation captions json file")
    p.add_argument("--features_dir", default="D:/DACN/dataset/Flickr8k/features", help="directory of image features")
    p.add_argument("--tokenizer_dir", default="D:/DACN/dataset/Flickr8k/tokenizer", help="directory of tokenizer")
    p.add_argument("--ckpt", default="D:/DACN/checkpoints_vit_lstm/best.pth", help="Path to XE checkpoint (best.pth) to initialize weights")
    p.add_argument("--output_dir", default="ckpt_scst")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--embed_dim", type=int, default=512)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--max_len", type=int, default=30)
    p.add_argument("--max_patches", type=int, default=196)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--sample_n", type=int, default=5, help="Number of samples per image (ignored, model.sample_decode samples once with temperature)")
    p.add_argument("--sample_temperature", type=float, default=1.0)
    p.add_argument("--reward", type=str, default="cider", choices=["cider","bleu"], help="Reward metric for SCST")
    p.add_argument("--xe_weight", type=float, default=0.0, help="If >0, mix XE auxiliary loss with SCST loss")
    p.add_argument("--use_beam", action="store_true", help="Use beam in eval logging")
    p.add_argument("--beam_size", type=int, default=5)
    p.add_argument("--beam_length_penalty", type=float, default=0.7)
    p.add_argument("--no_repeat_ngram_size", type=int, default=3)
    args = p.parse_args()
    run_scst(args)