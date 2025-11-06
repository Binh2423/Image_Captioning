import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel

class ViTEncoder(nn.Module):
    """
    Wrap HuggingFace ViTModel to return:
      - patch_tokens: (B, N, D) including CLS token removed from patches
      - cls_token: (B, D)
    If freeze=True, parameters are set requires_grad=False and model.eval() is recommended.
    """
    def __init__(self, model_name='google/vit-base-patch16-224-in21k', freeze=True):
        super().__init__()
        self.vit = ViTModel.from_pretrained(model_name)
        if freeze:
            for p in self.vit.parameters():
                p.requires_grad = False
            self.vit.eval()
        self.hidden_size = self.vit.config.hidden_size

    def forward(self, images):
        # images: normalized tensor (B, 3, H, W)
        outputs = self.vit(pixel_values=images, return_dict=True)
        last_hidden = outputs.last_hidden_state
        cls = last_hidden[:, 0, :]           # (B, D)
        patches = last_hidden[:, 1:, :]      # (B, N, D)
        return patches, cls


class Attention(nn.Module):
    """
    Additive (Bahdanau) attention over encoder patch tokens.
    """
    def __init__(self, enc_dim, dec_dim, attn_dim):
        super().__init__()
        self.enc_proj = nn.Linear(enc_dim, attn_dim)
        self.dec_proj = nn.Linear(dec_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1)

    def forward(self, enc_feats, dec_hidden):
        # enc_feats: (B, N, enc_dim)
        # dec_hidden: (B, dec_dim)
        enc_e = self.enc_proj(enc_feats)                  # (B, N, attn_dim)
        dec_e = self.dec_proj(dec_hidden).unsqueeze(1)    # (B, 1, attn_dim)
        e = torch.tanh(enc_e + dec_e)                     # (B, N, attn_dim)
        scores = self.v(e).squeeze(-1)                    # (B, N)
        alpha = F.softmax(scores, dim=1)                  # (B, N)
        context = (alpha.unsqueeze(-1) * enc_feats).sum(1) # (B, enc_dim)
        return context, alpha


class DecoderLSTM(nn.Module):
    """
    Decoder implemented with efficient embedding of whole sequence to reduce python overhead.
    Uses LSTMCell for stepwise recurrence because attention depends on hidden state each step.
    """
    def __init__(self, vocab_size, embed_dim, enc_dim, dec_dim, attn_dim, padding_idx=0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.attention = Attention(enc_dim, dec_dim, attn_dim)
        self.lstm = nn.LSTMCell(embed_dim + enc_dim, dec_dim)
        self.fc = nn.Linear(dec_dim, vocab_size)
        self.init_h = nn.Linear(enc_dim, dec_dim)
        self.init_c = nn.Linear(enc_dim, dec_dim)
        self.dropout = nn.Dropout(0.5)

    def init_states(self, cls_feat):
        # cls_feat: (B, enc_dim)
        h0 = torch.tanh(self.init_h(cls_feat))
        c0 = torch.tanh(self.init_c(cls_feat))
        return h0, c0

    def forward_step(self, emb_t, h, c, enc_feats):
        """
        emb_t: (B, embed_dim) already looked up
        h, c: (B, dec_dim)
        enc_feats: (B, N, enc_dim)
        """
        context, alpha = self.attention(enc_feats, h)  # context (B, enc_dim)
        lstm_input = torch.cat([emb_t, context], dim=-1) # (B, embed+enc)
        h, c = self.lstm(lstm_input, (h, c))
        out = self.fc(self.dropout(h))    # (B, vocab)
        return out, h, c, alpha

    def forward(self, captions, cls_feat, enc_feats):
        # captions: (B, T) input tokens (with <start> prefix)
        B, T = captions.size()
        device = captions.device
        # embed whole sequence once -> (B, T, E)
        emb_seq = self.embed(captions)  # (B, T, E)
        h, c = self.init_states(cls_feat)
        outputs = []
        alphas = []
        for t in range(T):
            emb_t = emb_seq[:, t, :]  # (B, E)
            out, h, c, alpha = self.forward_step(emb_t, h, c, enc_feats)
            outputs.append(out.unsqueeze(1))
            alphas.append(alpha.unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)  # (B, T, V)
        alphas = torch.cat(alphas, dim=1)    # (B, T, N)
        return outputs, alphas

    def sample_sequence(self, cls_feat, enc_feats, start_token, end_token, max_len=20, sample=False):
        """
        Generate sequence either by greedy argmax (sample=False) or sampling (sample=True).
        Sampling is done under no_grad() to save memory when called during training for SCST baseline/gen.
        """
        device = enc_feats.device
        with torch.no_grad():
            h, c = self.init_states(cls_feat)
            batch_size = enc_feats.size(0)
            word = torch.full((batch_size,), start_token, dtype=torch.long, device=device)
            seqs = [[] for _ in range(batch_size)]
            finished = [False] * batch_size
            for t in range(max_len):
                emb_t = self.embed(word)  # (B, E)
                out, h, c, alpha = self.forward_step(emb_t, h, c, enc_feats)
                probs = torch.softmax(out, dim=-1)  # (B, V)
                if sample:
                    word = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    word = torch.argmax(probs, dim=-1)
                for i in range(batch_size):
                    if not finished[i]:
                        token = word[i].item()
                        seqs[i].append(token)
                        if token == end_token:
                            finished[i] = True
                if all(finished):
                    break
        return seqs