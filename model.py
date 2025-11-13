#!/usr/bin/env python3
"""
Decoder LSTM with additive attention over projected ViT patch features.

Added: beam_decode(...) implementation (per-sample beam search).
Keep greedy_decode and sample_decode as before.
"""
import torch
import torch.nn as nn
from typing import Optional, List, Tuple


class AdditiveAttention(nn.Module):
    def __init__(self, query_dim, key_dim, attn_dim):
        super().__init__()
        self.q_proj = nn.Linear(query_dim, attn_dim, bias=False)
        self.k_proj = nn.Linear(key_dim, attn_dim, bias=False)
        self.v_att = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, keys, query, mask: Optional[torch.Tensor] = None):
        B, N, _ = keys.size()
        q = self.q_proj(query).unsqueeze(1)
        k = self.k_proj(keys)
        e = torch.tanh(q + k)
        scores = self.v_att(e).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        attn = torch.softmax(scores, dim=-1)
        context = torch.bmm(attn.unsqueeze(1), keys).squeeze(1)
        return context, attn


class DecoderLSTMWithAttention(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 512, feat_dim: int = 768,
                 hidden_dim: int = 512, num_layers: int = 1, dropout: float = 0.5,
                 pad_id: int = 0, tie_embeddings: bool = False, proj_dim: Optional[int] = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.pad_id = pad_id
        self.proj_dim = proj_dim or hidden_dim

        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.embed_dropout = nn.Dropout(dropout)

        self.feat_proj = nn.Linear(feat_dim, self.proj_dim)
        self.feat_ln = nn.LayerNorm(self.proj_dim)
        self.feat_dropout = nn.Dropout(dropout)

        self.attn = AdditiveAttention(query_dim=hidden_dim, key_dim=self.proj_dim, attn_dim=max(64, self.proj_dim // 2))

        self.lstm = nn.LSTM(input_size=embed_dim + self.proj_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)

        self.init_h = nn.Linear(feat_dim, hidden_dim)
        self.init_c = nn.Linear(feat_dim, hidden_dim)

        self.fc = nn.Linear(hidden_dim, vocab_size)

        if tie_embeddings:
            try:
                self.fc.weight = self.embed.weight
            except Exception:
                pass

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
        for name, p in self.lstm.named_parameters():
            if 'bias' in name:
                n = p.size(0)
                start, end = n // 4, n // 2
                with torch.no_grad():
                    p[start:end].fill_(1.0)

    def forward(self, feats, inp):
        B, N, _ = feats.size()
        T = inp.size(1)

        proj_feats = self.feat_proj(feats)
        proj_feats = self.feat_ln(proj_feats)
        proj_feats = self.feat_dropout(proj_feats)

        feat_mean = feats.mean(dim=1)
        h0 = torch.tanh(self.init_h(feat_mean)).unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = torch.tanh(self.init_c(feat_mean)).unsqueeze(0).repeat(self.num_layers, 1, 1)

        emb = self.embed(inp)
        emb = self.embed_dropout(emb)

        outputs = []
        h, c = h0, c0
        for t in range(T):
            emb_t = emb[:, t, :]
            query = h[-1]
            ctx, _ = self.attn(proj_feats, query)
            lstm_in = torch.cat([emb_t, ctx], dim=1).unsqueeze(1)
            out, (h, c) = self.lstm(lstm_in, (h, c))
            out = out.squeeze(1)
            logits = self.fc(out)
            outputs.append(logits.unsqueeze(1))
        logits = torch.cat(outputs, dim=1)
        return logits

    @torch.no_grad()
    def greedy_decode(self, feats, sos_id, eos_id, max_len=30):
        device = feats.device
        B = feats.size(0)
        proj_feats = self.feat_proj(feats)
        proj_feats = self.feat_ln(proj_feats)
        proj_feats = self.feat_dropout(proj_feats)

        feat_mean = feats.mean(dim=1)
        h = torch.tanh(self.init_h(feat_mean)).unsqueeze(0).repeat(self.num_layers, 1, 1)
        c = torch.tanh(self.init_c(feat_mean)).unsqueeze(0).repeat(self.num_layers, 1, 1)

        ys = torch.full((B, 1), sos_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for t in range(max_len - 1):
            last = ys[:, -1]
            emb = self.embed(last)
            query = h[-1]
            ctx, _ = self.attn(proj_feats, query)
            lstm_in = torch.cat([emb, ctx], dim=1).unsqueeze(1)
            out, (h, c) = self.lstm(lstm_in, (h, c))
            logits = self.fc(out.squeeze(1))
            next_tok = logits.argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_tok], dim=1)
            finished = finished | (next_tok.squeeze(1) == eos_id)
            if finished.all():
                break

        if ys.size(1) < max_len:
            pad_len = max_len - ys.size(1)
            pad = torch.full((B, pad_len), eos_id, dtype=torch.long, device=device)
            ys = torch.cat([ys, pad], dim=1)
        return ys

    @torch.no_grad()
    def beam_decode(self,
                    feats: torch.Tensor,
                    sos_id: int,
                    eos_id: int,
                    max_len: int = 30,
                    beam_size: int = 3,
                    length_penalty: float = 1.0,
                    early_stopping: bool = True,
                    no_repeat_ngram_size: int = 0) -> torch.LongTensor:
        """
        Beam search with optional no_repeat_ngram_size blocking.

        no_repeat_ngram_size: if >0, block any candidate that would create an ngram
        of that size which already appeared in the prefix.
        """
        device = feats.device
        B, N, D = feats.size()

        proj_feats_all = self.feat_proj(feats)
        proj_feats_all = self.feat_ln(proj_feats_all)
        proj_feats_all = self.feat_dropout(proj_feats_all)

        feat_mean = feats.mean(dim=1)
        h0_all = torch.tanh(self.init_h(feat_mean))  # (B, H)
        c0_all = torch.tanh(self.init_c(feat_mean))  # (B, H)

        results = []
        for b in range(B):
            proj_feats = proj_feats_all[b:b+1]  # (1, N, P)
            h0 = h0_all[b].unsqueeze(0).repeat(self.num_layers, 1, 1).to(device)
            c0 = c0_all[b].unsqueeze(0).repeat(self.num_layers, 1, 1).to(device)

            # beam entries: (tokens_list, logprob, h, c, finished)
            beams = [([sos_id], 0.0, h0.clone(), c0.clone(), False)]
            for t in range(max_len - 1):
                all_candidates = []
                for tokens, logp, h_beam, c_beam, finished in beams:
                    if finished:
                        all_candidates.append((tokens, logp, h_beam, c_beam, True))
                        continue
                    last_tok = torch.tensor([tokens[-1]], dtype=torch.long, device=device)
                    emb = self.embed(last_tok)
                    query = h_beam[-1]
                    ctx, _ = self.attn(proj_feats, query)
                    lstm_in = torch.cat([emb, ctx], dim=1).unsqueeze(1)
                    out, (h_new, c_new) = self.lstm(lstm_in, (h_beam, c_beam))
                    logits = self.fc(out.squeeze(1))
                    log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)

                    topk_logp, topk_ids = torch.topk(log_probs, k=min(beam_size, log_probs.size(0)))
                    topk_logp = topk_logp.cpu().tolist()
                    topk_ids = topk_ids.cpu().tolist()

                    for k_logp, k_id in zip(topk_logp, topk_ids):
                        # check no_repeat_ngram constraint
                        if no_repeat_ngram_size > 0 and len(tokens) >= no_repeat_ngram_size - 1:
                            # candidate tokens = tokens + [k_id], check if last ngram appeared before
                            cand = tokens + [int(k_id)]
                            n = no_repeat_ngram_size
                            # build set of ngrams in prefix (except the last candidate ngram)
                            seen = set()
                            for i in range(len(tokens) - n + 1):
                                seen.add(tuple(tokens[i:i+n]))
                            last_ngram = tuple(cand[-n:])
                            if last_ngram in seen:
                                # skip this candidate (would repeat an ngram)
                                continue

                        new_tokens = tokens + [int(k_id)]
                        new_logp = logp + float(k_logp)
                        new_h = h_new.clone()
                        new_c = c_new.clone()
                        new_finished = (k_id == eos_id)
                        all_candidates.append((new_tokens, new_logp, new_h, new_c, new_finished))

                if not all_candidates:
                    break

                # select top beam_size by normalized score
                scored = []
                for cand in all_candidates:
                    tok_list, lp_cand, _, _, fin = cand
                    length = max(1, len(tok_list))
                    score = lp_cand / (length ** length_penalty)
                    scored.append((score, cand))
                scored.sort(key=lambda x: x[0], reverse=True)
                beams = [entry[1] for entry in scored[:beam_size]]

                if early_stopping and all(b[4] for b in beams):
                    break

            finished_beams = [b for b in beams if b[4]]
            if len(finished_beams) > 0:
                best = max(finished_beams, key=lambda x: x[1] / (max(1, len(x[0])) ** length_penalty))
            else:
                best = max(beams, key=lambda x: x[1] / (max(1, len(x[0])) ** length_penalty))

            seq = best[0]
            if len(seq) < max_len:
                seq = seq + [eos_id] * (max_len - len(seq))
            else:
                seq = seq[:max_len]
            results.append(seq)

        out = torch.tensor(results, dtype=torch.long, device=device)
        return out

    @torch.no_grad()
    def sample_decode(self, feats, sos_id, eos_id, max_len=30, temperature=1.0):
        device = feats.device
        B = feats.size(0)
        proj_feats = self.feat_proj(feats)
        proj_feats = self.feat_ln(proj_feats)
        proj_feats = self.feat_dropout(proj_feats)

        feat_mean = feats.mean(dim=1)
        h = torch.tanh(self.init_h(feat_mean)).unsqueeze(0).repeat(self.num_layers, 1, 1)
        c = torch.tanh(self.init_c(feat_mean)).unsqueeze(0).repeat(self.num_layers, 1, 1)

        ys = torch.full((B, 1), sos_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        logp_sum = torch.zeros(B, device=device)

        for t in range(max_len - 1):
            last = ys[:, -1]
            emb = self.embed(last)
            query = h[-1]
            ctx, _ = self.attn(proj_feats, query)
            lstm_in = torch.cat([emb, ctx], dim=1).unsqueeze(1)
            out, (h, c) = self.lstm(lstm_in, (h, c))
            logits = self.fc(out.squeeze(1))
            if temperature != 1.0:
                logits = logits / max(1e-6, float(temperature))
            probs = torch.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            sel_prob = probs.gather(1, next_tok).squeeze(1)
            mask = (~finished).float()
            logp_sum = logp_sum + (torch.log(sel_prob + 1e-12) * mask)
            ys = torch.cat([ys, next_tok], dim=1)
            finished = finished | (next_tok.squeeze(1) == eos_id)
            if finished.all():
                break

        if ys.size(1) < max_len:
            pad_len = max_len - ys.size(1)
            pad = torch.full((B, pad_len), eos_id, dtype=torch.long, device=device)
            ys = torch.cat([ys, pad], dim=1)
        return ys, logp_sum