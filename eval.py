import os
import argparse
import json
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from models import ViTEncoder, DecoderLSTM
from dataset import Flickr8kDataset
from utils import collate_fn, ids_to_sentences

# optional cider
try:
    from pycocoevalcap.cider.cider import Cider
    HAVE_CIDER = True
except Exception:
    HAVE_CIDER = False

# optional METEOR (NLTK)
try:
    from nltk.translate.meteor_score import single_meteor_score
    HAVE_METEOR = True
except Exception:
    HAVE_METEOR = False

# optional ROUGE (rouge-score)
try:
    from rouge_score import rouge_scorer
    HAVE_ROUGE = True
except Exception:
    HAVE_ROUGE = False

# fallback BLEU (NLTK) if CIDEr not available
try:
    from nltk.translate.bleu_score import corpus_bleu
    HAVE_BLEU = True
except Exception:
    HAVE_BLEU = False


def load_vocab_from_checkpoint(cp_path):
    try:
        cp = torch.load(cp_path, map_location='cpu')
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint '{cp_path}': {e}")
    if isinstance(cp, dict) and 'vocab' in cp:
        itos = cp['vocab']
        stoi = {w: i for i, w in enumerate(itos)}
        return itos, stoi, cp
    else:
        # if checkpoint is plain state_dict, caller should provide vocab separately
        return None, None, cp


class SimpleVocab:
    def __init__(self, itos):
        self.itos = itos
        self.stoi = {w: i for i, w in enumerate(itos)}


def generate_greedy(encoder, decoder, images, vocab, max_len):
    """
    images: tensor (B,3,H,W) on device
    returns: list of generated token id lists (no <start>, stops at <end>)
    """
    device = images.device
    with torch.no_grad():
        enc_feats, cls = encoder(images)
        gen_ids = decoder.sample_sequence(cls, enc_feats,
                                          start_token=vocab.stoi['<start>'],
                                          end_token=vocab.stoi['<end>'],
                                          max_len=max_len, sample=False)
    return gen_ids


def beam_search_single(encoder, decoder, enc_feats, cls, vocab, beam_size=3, max_len=30, length_norm=True):
    """
    enc_feats: (1, N, D)
    cls: (1, D)
    Return: single best sequence list (token ids) (excluding <start>, stopping at <end>)
    Beam search implemented per sample (done under no_grad).
    """
    device = enc_feats.device
    start_token = vocab.stoi['<start>']
    end_token = vocab.stoi['<end>']
    vocab_size = len(vocab.itos)

    # initialize beam: list of tuples (tokens_list, logprob, h, c, finished_flag)
    with torch.no_grad():
        h0, c0 = decoder.init_states(cls)  # expecting shape (1, dec_dim)
        # ensure shapes (1, dec_dim)
        h0 = h0.detach()
        c0 = c0.detach()

        beams = [{
            'tokens': [start_token],
            'logprob': 0.0,
            'h': h0,  # (1, dec_dim)
            'c': c0,
            'finished': False
        }]

        for t in range(max_len):
            all_candidates = []
            for b in beams:
                if b['finished']:
                    all_candidates.append(b)
                    continue
                # last token
                last_token = torch.tensor([b['tokens'][-1]], dtype=torch.long, device=device)
                emb_t = decoder.embed(last_token)  # (1, E)
                # forward_step expects (emb, h_prev, c_prev, enc_feats)
                out, h_new, c_new, _ = decoder.forward_step(emb_t, b['h'], b['c'], enc_feats)
                # out: (1, V)
                logp = torch.log_softmax(out, dim=-1).squeeze(0)  # (V,)
                # topk
                topk = min(beam_size, vocab_size)
                topk_logp, topk_idx = torch.topk(logp, topk)
                for k in range(topk_idx.size(0)):
                    token_id = int(topk_idx[k].item())
                    token_logp = float(topk_logp[k].item())
                    new_tokens = b['tokens'] + [token_id]
                    finished = (token_id == end_token)
                    candidate = {
                        'tokens': new_tokens,
                        'logprob': b['logprob'] + token_logp,
                        'h': h_new.detach(),  # (1, dec_dim)
                        'c': c_new.detach(),
                        'finished': finished
                    }
                    all_candidates.append(candidate)
            # scoring and pruning
            def score_fn(cand):
                lp = cand['logprob']
                if length_norm:
                    L = len(cand['tokens'])
                    return lp / (L ** 0.7)
                else:
                    return lp
            all_candidates.sort(key=score_fn, reverse=True)
            beams = all_candidates[:beam_size]
            if all([b['finished'] for b in beams]):
                break

        finished_beams = [b for b in beams if b['finished']]
        if finished_beams:
            best = max(finished_beams, key=score_fn)
        else:
            best = max(beams, key=score_fn)
        # remove start token and cut at end token
        toks = []
        for tid in best['tokens'][1:]:
            if tid == end_token:
                break
            toks.append(tid)
        return toks


def generate_beam(encoder, decoder, images, vocab, beam_size=3, max_len=30):
    """
    images: (B,3,H,W)
    return: list of id lists (B elements)
    Beam search performed per-sample sequentially.
    """
    device = images.device
    results = []
    with torch.no_grad():
        enc_feats_all, cls_all = encoder(images)
        B = images.size(0)
        for i in range(B):
            enc_feats = enc_feats_all[i:i+1]  # (1,N,D)
            cls = cls_all[i:i+1]              # (1,D)
            best_seq = beam_search_single(encoder, decoder, enc_feats, cls, vocab, beam_size=beam_size, max_len=max_len)
            results.append(best_seq)
    return results


def compute_metrics_all(hyp_dict, ref_dict, use_cider=True, use_meteor=True, use_rouge=True, force_bleu=False):
    """
    hyp_dict: id -> hyp string
    ref_dict: id -> list of refs strings
    returns dict of metrics (CIDEr/BLEU/METEOR/ROUGE-L average)
    """
    metrics = {}
    # CIDEr
    if HAVE_CIDER and use_cider and not force_bleu:
        try:
            cider = Cider()
            G = {k: ref_dict[k] for k in ref_dict}
            R = {k: [hyp_dict[k]] for k in hyp_dict}
            score, _ = cider.compute_score(G, R)
            metrics['CIDEr'] = float(score)
        except Exception as e:
            metrics['CIDEr_error'] = str(e)

    # BLEU-4
    if HAVE_BLEU:
        refs_list = []
        hyps_list = []
        for k in hyp_dict:
            hyps_list.append(hyp_dict[k].split())
            refs_tok = [r.split() for r in ref_dict[k]]
            refs_list.append(refs_tok)
        try:
            bleu4 = corpus_bleu(refs_list, hyps_list, weights=(0.25, 0.25, 0.25, 0.25))
            metrics['BLEU-4'] = float(bleu4)
        except Exception as e:
            metrics['BLEU-4_error'] = str(e)

    # METEOR (average sentence-level)
    if HAVE_METEOR and use_meteor:
        try:
            scores = []
            for k in hyp_dict:
                hyp = hyp_dict[k]
                score = single_meteor_score(ref_dict[k], hyp)
                scores.append(score)
            metrics['METEOR'] = float(sum(scores) / max(1, len(scores)))
        except Exception as e:
            metrics['METEOR_error'] = str(e)

    # ROUGE (average F1 of ROUGE-L)
    if HAVE_ROUGE and use_rouge:
        try:
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            vals = []
            for k in hyp_dict:
                hyp = hyp_dict[k]
                refs = ref_dict[k]
                best_f = 0.0
                for r in refs:
                    sc = scorer.score(r, hyp)
                    f = sc['rougeL'].fmeasure
                    if f > best_f:
                        best_f = f
                vals.append(best_f)
            metrics['ROUGE-L'] = float(sum(vals) / max(1, len(vals)))
        except Exception as e:
            metrics['ROUGE-L_error'] = str(e)

    return metrics


def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.force_cpu else 'cpu')
    itos, stoi, cp = None, None, None
    if args.checkpoint is not None:
        try:
            itos, stoi, cp = load_vocab_from_checkpoint(args.checkpoint)
        except Exception as e:
            raise RuntimeError(f"Error loading checkpoint: {e}")
    if itos is None:
        # allow passing a vocab file
        if args.vocab_file is not None:
            with open(args.vocab_file, 'r', encoding='utf-8') as vf:
                itos = json.load(vf)
        else:
            raise RuntimeError("Checkpoint must contain 'vocab' (list itos) or provide --vocab_file. Provide --checkpoint that includes the vocab or --vocab_file.")
    vocab = SimpleVocab(itos)

    val_ds = Flickr8kDataset(args.data_root, split='val', vocab=vocab, max_len=args.max_len)
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=collate_fn
    )

    encoder = ViTEncoder(freeze=True).to(device)
    decoder = DecoderLSTM(vocab_size=len(itos), embed_dim=args.embed_dim,
                          enc_dim=encoder.hidden_size, dec_dim=args.dec_dim, attn_dim=args.attn_dim).to(device)

    if cp is not None:
        # try multiple keys
        state = None
        if isinstance(cp, dict):
            if 'decoder_state' in cp:
                state = cp['decoder_state']
            elif 'model_state' in cp:
                state = cp['model_state']
            elif 'state_dict' in cp:
                state = cp['state_dict']
            else:
                state = cp
        try:
            decoder.load_state_dict(state, strict=False)
            print("Loaded decoder weights from checkpoint (strict=False).")
        except Exception as e:
            print("Warning: failed to strictly load decoder state:", e)

    encoder.eval()
    decoder.eval()

    results_hyp = {}   # img_id -> generated string
    results_ref = {}   # img_id -> list of reference strings

    use_beam = args.beam_size > 1

    for images, captions, img_names, refs in tqdm(val_loader, desc="Evaluating"):
        images = images.to(device, non_blocking=True)
        if use_beam:
            gen_ids_batch = generate_beam(encoder, decoder, images, vocab, beam_size=args.beam_size, max_len=args.max_len)
        else:
            gen_ids_batch = generate_greedy(encoder, decoder, images, vocab, max_len=args.max_len)

        gen_sents = ids_to_sentences(gen_ids_batch, vocab)
        for i, name in enumerate(img_names):
            key = name
            results_hyp[key] = gen_sents[i]
            results_ref[key] = refs[i]

    metrics = compute_metrics_all(results_hyp, results_ref, use_cider=not args.force_bleu, use_meteor=not args.no_meteor, use_rouge=not args.no_rouge, force_bleu=args.force_bleu)

    avg_len = sum(len(s.split()) for s in results_hyp.values()) / max(1, len(results_hyp))
    metrics['avg_gen_len'] = avg_len
    metrics['n_samples'] = len(results_hyp)

    os.makedirs(args.output_dir, exist_ok=True)
    out_json = os.path.join(args.output_dir, f"eval_results.json")
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump({'metrics': metrics, 'hyps': results_hyp, 'refs': results_ref}, f, ensure_ascii=False, indent=2)

    print("Evaluation finished. Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print("Saved results:", out_json)
    # return metrics + results for programmatic use
    return metrics, results_hyp, results_ref


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True, help='root folder of Flickr8k dataset')
    parser.add_argument('--checkpoint', required=True, help='path to XE/SCST checkpoint that contains vocab')
    parser.add_argument('--vocab_file', type=str, default=None, help='optional vocab file (JSON list) if checkpoint does not include vocab')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_len', type=int, default=30)
    parser.add_argument('--embed_dim', type=int, default=512)
    parser.add_argument('--dec_dim', type=int, default=512)
    parser.add_argument('--attn_dim', type=int, default=512)
    parser.add_argument('--pin_memory', action='store_true', help='use pin_memory in DataLoader')
    parser.add_argument('--output_dir', type=str, default='eval_outputs')
    parser.add_argument('--force_cpu', action='store_true', help='force CPU even if CUDA available')
    parser.add_argument('--force_bleu', action='store_true', help='force BLEU fallback even if CIDEr available')
    parser.add_argument('--beam_size', type=int, default=1, help='Beam size; 1 = greedy')
    parser.add_argument('--no_meteor', action='store_true', help='disable METEOR computation')
    parser.add_argument('--no_rouge', action='store_true', help='disable ROUGE computation')
    args = parser.parse_args()

    evaluate(args)