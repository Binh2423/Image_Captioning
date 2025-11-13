
import argparse
import os
import json
import time
import math
from regex import D
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tokenizer_utils import load_tokenizer, ids_to_text
from dataset import CaptionPatchDataset, collate_fn
from model import DecoderLSTMWithAttention
from tokenizer_utils import load_tokenizer, ids_to_text, normalize_detokenized
from sacrebleu import corpus_bleu
from nltk.translate.bleu_score import corpus_bleu as nltk_corpus_bleu, SmoothingFunction

smooth = SmoothingFunction().method1

def pad_shift_targets(tgt_batch, pad_id):
    inp = tgt_batch[:, :-1]
    target = tgt_batch[:, 1:]
    return inp, target

def compute_bleu1_4_nltk(hypotheses, list_of_references):
    hyps_tok = [h.strip().split() for h in hypotheses]
    refs_tok = [[r.strip().split() for r in refs] for refs in list_of_references]
    weights_list = [
        (1.0, 0, 0, 0),
        (0.5, 0.5, 0, 0),
        (1/3, 1/3, 1/3, 0),
        (0.25, 0.25, 0.25, 0.25),
    ]
    results = {}
    for i, weights in enumerate(weights_list, start=1):
        try:
            score = nltk_corpus_bleu(refs_tok, hyps_tok, weights=weights, smoothing_function=smooth)
            results[f"bleu{i}"] = float(score * 100.0)
        except Exception:
            results[f"bleu{i}"] = 0.0
    return results

def compute_sacrebleu(hyps, list_of_refs):
    max_refs = max(len(r) for r in list_of_refs) if len(list_of_refs) > 0 else 1
    ref_streams = []
    for k in range(max_refs):
        ref_streams.append([r[k] if k < len(r) else r[0] for r in list_of_refs])
    return corpus_bleu(hyps, ref_streams).score

class LabelSmoothingLoss(nn.Module):
    def __init__(self, label_smoothing: float, tgt_vocab_size: int, ignore_index: int = -100):
        super().__init__()
        assert 0.0 <= label_smoothing < 1.0
        self.smoothing = float(label_smoothing)
        self.confidence = 1.0 - self.smoothing
        self.vocab_size = tgt_vocab_size
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        log_prob = torch.log_softmax(pred, dim=-1)
        with torch.no_grad():
            true_dist = pred.new_full(pred.size(), self.smoothing / (self.vocab_size - 1))
            target_unsq = target.unsqueeze(1)
            true_dist.scatter_(1, target_unsq, self.confidence)
            mask = (target == self.ignore_index).unsqueeze(1)
            true_dist.masked_fill_(mask, 0.0)
        loss = (-true_dist * log_prob).sum(dim=1)
        loss = loss.masked_fill(target == self.ignore_index, 0.0)
        return loss.mean()

def scheduled_sampling_prob(global_step, total_steps, mode="inverse_sigmoid", k=1000, end_ratio=0.5):
    if mode is None or mode == "none":
        return 1.0
    if mode == "inverse_sigmoid":
        try:
            val = float(k) / (float(k) + math.exp(float(global_step) / float(k)))
        except OverflowError:
            val = 0.0
        return float(val)
    if mode == "linear":
        denom = max(1, int(total_steps * float(end_ratio)))
        p = max(0.0, 1.0 - float(global_step) / denom)
        return float(p)
    return 1.0

def _resolve_refs_for_img(loader_dataset, img_id):
    caps = getattr(loader_dataset, "raw_caps", None) or getattr(loader_dataset, "caps", None)
    if caps is None:
        return []
    if str(img_id) in caps:
        refs = caps[str(img_id)]
    elif img_id in caps:
        refs = caps[img_id]
    else:
        for k in caps:
            if str(img_id) == str(k) or str(img_id) in str(k):
                refs = caps[k]
                break
        else:
            refs = []
    if isinstance(refs, str):
        refs = [refs]
    return refs

def evaluate(model, loader, tk, device, max_len=30, save_preds_path=None,
             use_beam=False, beam_size=3, beam_length_penalty=1.0, no_repeat_ngram_size=4):
    model.eval()
    hyps = []
    refs = []
    unk_count = 0
    total_tokens = 0
    samples = []
    with torch.no_grad():
        for feats, tgts, ids in tqdm(loader, desc="Eval", leave=False):
            feats = feats.to(device, non_blocking=True)
            if use_beam: 
                seqs = model.beam_decode(feats, tk["bos_id"], tk["eos_id"],
                                         max_len=max_len, beam_size=beam_size,
                                         length_penalty=beam_length_penalty,
                                         no_repeat_ngram_size=no_repeat_ngram_size)
            else:
                seqs = model.greedy_decode(feats, tk["bos_id"], tk["eos_id"], max_len=max_len)
            seqs = seqs.cpu().numpy()
            for seq, img_id in zip(seqs, ids):
                hyp_text = ids_to_text(seq, tk)
                hyp_text = normalize_detokenized(hyp_text)
                hyps.append(hyp_text)

                raw_refs = _resolve_refs_for_img(loader.dataset, img_id)
                if not raw_refs:
                    raw_refs = [""]
                decoded_refs = []
                for r in raw_refs:
                    if isinstance(r, list):
                        rtxt = ids_to_text(r, tk)
                        decoded_refs.append(normalize_detokenized(rtxt))
                    else:
                        if isinstance(r, str) and r.strip().startswith("[") and r.strip().endswith("]"):
                            try:
                                arr = json.loads(r)
                                if isinstance(arr, list):
                                    decoded_refs.append(normalize_detokenized(ids_to_text(arr, tk)))
                                    continue
                            except Exception:
                                pass
                        decoded_refs.append(normalize_detokenized(str(r)))
                refs.append(decoded_refs)

                unk = tk.get("unk_id", None)
                if unk is not None:
                    unk_count += sum(1 for x in list(seq) if int(x) == int(unk))
                    total_tokens += len(list(seq))

                samples.append({"image_id": str(img_id), "pred": hyp_text, "refs": decoded_refs})

    bleu_scores = compute_bleu1_4_nltk(hyps, refs)
    sacre = compute_sacrebleu(hyps, refs)
    unk_pct = (100.0 * unk_count / max(1, total_tokens)) if total_tokens > 0 else 0.0
    if save_preds_path:
        with open(save_preds_path, "w", encoding="utf-8") as f:
            json.dump({"meta": {"bleu": bleu_scores, "sacre": sacre, "unk_pct": unk_pct}, "samples": samples}, f, ensure_ascii=False, indent=2)
    return bleu_scores, sacre, unk_pct, samples

def main(args):
    device = torch.device(args.device)
    tk = load_tokenizer(args.tokenizer_dir)
    vocab = tk.get("vocab")
    if vocab is None:
        raise SystemExit("vocab.json not found in tokenizer_dir: " + str(args.tokenizer_dir))
    pad_id = tk["pad_id"]
    bos_id = tk["bos_id"]
    eos_id = tk["eos_id"]

    train_ds = CaptionPatchDataset(args.train_captions, args.features_dir, args.tokenizer_dir, max_len=args.max_len, max_patches=args.max_patches)
    val_ds   = CaptionPatchDataset(args.val_captions,   args.features_dir, args.tokenizer_dir, max_len=args.max_len, max_patches=args.max_patches)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
                              num_workers=max(1, args.num_workers//2), pin_memory=True)

    model = DecoderLSTMWithAttention(vocab_size=len(vocab), embed_dim=args.embed_dim, feat_dim=args.feat_dim,
                                     hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout=args.dropout, pad_id=pad_id)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * max(1, args.epochs)
    warmup_steps = args.warmup_steps if (args.warmup_steps and args.warmup_steps>0) else max(1, int(total_steps * args.warmup_ratio))
    scheduler = make_lr_scheduler(optimizer, total_steps, warmup_steps)

    if args.label_smoothing > 0.0:
        loss_fn = LabelSmoothingLoss(label_smoothing=args.label_smoothing, tgt_vocab_size=len(vocab), ignore_index=pad_id)
    else:
        loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)

    os.makedirs(args.output_dir, exist_ok=True)
    best_bleu = -1.0
    start_epoch = 0

    if args.resume:
        ckpt_path = args.resume_ckpt if args.resume_ckpt else os.path.join(args.output_dir, "best.pth")
        if os.path.exists(ckpt_path):
            print("Resuming from", ckpt_path)
            sd = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(sd.get("model", sd), strict=False)
            try:
                optimizer.load_state_dict(sd.get("optim", {}))
            except Exception:
                print("Warning: couldn't load optimizer state (shapes mismatch).")
            start_epoch = sd.get("epoch", 0) + 1
            best_bleu = sd.get("best_bleu", best_bleu)

    global_step = 0
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        step = 0
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=True)
        for feats, tgts, ids in pbar:
            feats = feats.to(device, non_blocking=True)
            tgts  = tgts.to(device, non_blocking=True)
            inp, target = pad_shift_targets(tgts, pad_id=pad_id)

            optimizer.zero_grad()
            logits_tf = model(feats, inp)
            with torch.no_grad():
                greedy_preds = logits_tf.argmax(dim=-1)
                sos_token = torch.full((greedy_preds.size(0), 1), bos_id, dtype=torch.long, device=device)
                greedy_preds_shifted = torch.cat([sos_token, greedy_preds[:, :-1]], dim=1)

            p_teacher = scheduled_sampling_prob(global_step, total_steps, mode=args.scheduled_sampling, k=args.ss_k, end_ratio=args.ss_end_ratio)
            mixed_inp = inp
            if args.scheduled_sampling and args.scheduled_sampling.lower() != "none":
                prob = torch.full((inp.size(0), inp.size(1)), p_teacher, device=inp.device, dtype=torch.float32)
                bern = torch.bernoulli(prob)
                mask = bern.bool()
                mask[:, 0] = True
                mixed_inp = torch.where(mask, inp, greedy_preds_shifted)

            logits = model(feats, mixed_inp)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), target.reshape(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                try:
                    scheduler.step()
                except Exception:
                    pass

            running_loss += float(loss.item())
            step += 1
            global_step += 1
            if step % args.log_every == 0:
                pbar.set_postfix(loss=(running_loss/step), lr=optimizer.param_groups[0]["lr"], ss_p=f"{p_teacher:.4f}")

        epoch_time = time.time() - t0
        preds_path = os.path.join(args.output_dir, f"preds_epoch{epoch}.json")
        bleu_scores, sacre, unk_pct, samples = evaluate(model, val_loader, tk, device,
                                                        max_len=args.max_len,
                                                        save_preds_path=preds_path,
                                                        use_beam=args.use_beam,
                                                        beam_size=args.beam_size,
                                                        beam_length_penalty=args.beam_length_penalty)
        print(f"Epoch {epoch} done in {epoch_time:.1f}s, train_loss={running_loss/ max(1,step):.4f}, val_sacrebleu={sacre:.4f}, unk%={unk_pct:.2f}")
        print(f"Validation NLTK BLEU-1..4: {bleu_scores.get('bleu1',0):.3f}, {bleu_scores.get('bleu2',0):.3f}, {bleu_scores.get('bleu3',0):.3f}, {bleu_scores.get('bleu4',0):.3f}")

        ckpt = {"epoch": epoch, "model": model.state_dict(), "optim": optimizer.state_dict(), "vocab": vocab, "best_bleu": best_bleu}
        if bleu_scores.get("bleu4", 0) > best_bleu:
            best_bleu = bleu_scores.get("bleu4", 0)
            ckpt["best_bleu"] = best_bleu
            torch.save(ckpt, os.path.join(args.output_dir, "best.pth"))
            print("Saved best model (BLEU-4={:.3f}).".format(best_bleu))

    print("Training finished. Best BLEU-4:", best_bleu)

def make_lr_scheduler(optimizer, total_steps, warmup_steps):
    if total_steps <= 0:
        return None
    warmup_steps = max(1, int(warmup_steps))
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_captions", default="D:/DACN/dataset/Flickr8k/Flickr8k_text/captions_train.json", help="path to train captions json file")
    p.add_argument("--val_captions", default="D:/DACN/dataset/Flickr8k/Flickr8k_text/captions_val.json", help="path to validation captions json file")
    p.add_argument("--features_dir", default="D:/DACN/dataset/Flickr8k/features", help="directory of image features")
    p.add_argument("--tokenizer_dir", default="D:/DACN/dataset/Flickr8k/tokenizer", help="directory of tokenizer")
    p.add_argument("--output_dir", default="checkpoints_vit_lstm")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max_len", type=int, default=30)
    p.add_argument("--embed_dim", type=int, default=512)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--feat_dim", type=int, default=768)
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--max_patches", type=int, default=196)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--warmup_steps", type=int, default=None)
    p.add_argument("--warmup_ratio", type=float, default=0.01)
    p.add_argument("--scheduled_sampling", type=str, default="none", choices=["none","inverse_sigmoid","linear"])
    p.add_argument("--ss_k", type=float, default=1000.0)
    p.add_argument("--ss_end_ratio", type=float, default=0.5)
    # Beam decoding options
    p.add_argument("--use_beam", action="store_true", help="Use beam search at evaluation time")
    p.add_argument("--beam_size", type=int, default=5, help="Beam width for beam search")
    p.add_argument("--beam_length_penalty", type=float, default=0.7, help="Length penalty exponent for beam scoring")
    # resume
    p.add_argument("--resume", action="store_true")
    p.add_argument("--resume_ckpt", type=str, default=None)
    args = p.parse_args() 
    main(args)