"""
Vision Transformer + LSTM model for image captioning.
Includes beam search with length penalty, n-gram blocking, and repetition penalty.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ViTLSTMCaptioner(nn.Module):
    """
    Image captioning model with LSTM decoder.
    Takes pre-extracted patch features as input.
    """
    
    def __init__(self, feature_dim, embed_dim, hidden_dim, vocab_size, num_layers=1, dropout=0.5):
        """
        Args:
            feature_dim: Dimension of input patch features
            embed_dim: Dimension of word embeddings
            hidden_dim: Dimension of LSTM hidden state
            vocab_size: Size of vocabulary
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        # Feature projection
        self.feature_proj = nn.Linear(feature_dim, embed_dim)
        
        # Word embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # LSTM decoder
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Output projection
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, features, captions):
        """
        Forward pass for training.
        
        Args:
            features: (batch_size, num_patches, feature_dim)
            captions: (batch_size, seq_len) - includes <sos> but not <eos> in input
        
        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        batch_size = features.shape[0]
        
        # Pool features (mean pooling)
        feat_pooled = features.mean(dim=1)  # (batch_size, feature_dim)
        feat_proj = self.feature_proj(feat_pooled)  # (batch_size, embed_dim)
        
        # Initialize hidden state with features
        h0 = feat_proj.unsqueeze(0).repeat(self.num_layers, 1, 1)  # (num_layers, batch_size, embed_dim -> hidden_dim)
        # Adjust if embed_dim != hidden_dim
        if self.embed_dim != self.hidden_dim:
            h0 = torch.randn(self.num_layers, batch_size, self.hidden_dim, device=features.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=features.device)
        
        # Embed captions
        embeds = self.embedding(captions)  # (batch_size, seq_len, embed_dim)
        embeds = self.dropout(embeds)
        
        # LSTM forward
        lstm_out, _ = self.lstm(embeds, (h0, c0))  # (batch_size, seq_len, hidden_dim)
        
        # Project to vocabulary
        logits = self.fc(lstm_out)  # (batch_size, seq_len, vocab_size)
        
        return logits
    
    def greedy_decode(self, features, max_len, sos_id, eos_id, pad_id):
        """
        Greedy decoding.
        
        Args:
            features: (batch_size, num_patches, feature_dim)
            max_len: Maximum sequence length
            sos_id: Start-of-sequence token ID
            eos_id: End-of-sequence token ID
            pad_id: Padding token ID
        
        Returns:
            sequences: (batch_size, max_len) - generated token IDs
        """
        batch_size = features.shape[0]
        device = features.device
        
        # Initialize
        feat_pooled = features.mean(dim=1)
        feat_proj = self.feature_proj(feat_pooled)
        
        h = feat_proj.unsqueeze(0).repeat(self.num_layers, 1, 1)
        if self.embed_dim != self.hidden_dim:
            h = torch.randn(self.num_layers, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        
        # Start with <sos>
        sequences = torch.full((batch_size, max_len), pad_id, dtype=torch.long, device=device)
        sequences[:, 0] = sos_id
        
        current_input = torch.full((batch_size, 1), sos_id, dtype=torch.long, device=device)
        
        for t in range(1, max_len):
            embeds = self.embedding(current_input)  # (batch_size, 1, embed_dim)
            lstm_out, (h, c) = self.lstm(embeds, (h, c))
            logits = self.fc(lstm_out.squeeze(1))  # (batch_size, vocab_size)
            
            # Greedy selection
            predicted = logits.argmax(dim=-1)  # (batch_size,)
            sequences[:, t] = predicted
            
            current_input = predicted.unsqueeze(1)
        
        return sequences
    
    def sample_decode(self, features, max_len, sos_id, eos_id, pad_id, temperature=1.0):
        """
        Sampling-based decoding.
        
        Args:
            features: (batch_size, num_patches, feature_dim)
            max_len: Maximum sequence length
            sos_id: Start-of-sequence token ID
            eos_id: End-of-sequence token ID
            pad_id: Padding token ID
            temperature: Sampling temperature
        
        Returns:
            sequences: (batch_size, max_len) - generated token IDs
        """
        batch_size = features.shape[0]
        device = features.device
        
        # Initialize
        feat_pooled = features.mean(dim=1)
        feat_proj = self.feature_proj(feat_pooled)
        
        h = feat_proj.unsqueeze(0).repeat(self.num_layers, 1, 1)
        if self.embed_dim != self.hidden_dim:
            h = torch.randn(self.num_layers, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        
        # Start with <sos>
        sequences = torch.full((batch_size, max_len), pad_id, dtype=torch.long, device=device)
        sequences[:, 0] = sos_id
        
        current_input = torch.full((batch_size, 1), sos_id, dtype=torch.long, device=device)
        
        for t in range(1, max_len):
            embeds = self.embedding(current_input)
            lstm_out, (h, c) = self.lstm(embeds, (h, c))
            logits = self.fc(lstm_out.squeeze(1))  # (batch_size, vocab_size)
            
            # Sample from distribution
            probs = F.softmax(logits / temperature, dim=-1)
            predicted = torch.multinomial(probs, 1).squeeze(1)  # (batch_size,)
            sequences[:, t] = predicted
            
            current_input = predicted.unsqueeze(1)
        
        return sequences
    
    def beam_decode(self, features, max_len, sos_id, eos_id, pad_id, 
                   beam_size=5, length_penalty=1.0, no_repeat_ngram_size=0,
                   repetition_penalty=1.0, min_len_eos=0):
        """
        Beam search decoding with length penalty, n-gram blocking, and repetition penalty.
        Performs per-sample beam search.
        
        Args:
            features: (batch_size, num_patches, feature_dim)
            max_len: Maximum sequence length
            sos_id: Start-of-sequence token ID
            eos_id: End-of-sequence token ID
            pad_id: Padding token ID
            beam_size: Number of beams
            length_penalty: Length penalty factor (> 1.0 favors longer sequences)
            no_repeat_ngram_size: Block n-grams that already appeared (0 = disabled)
            repetition_penalty: Penalty for repeated tokens (> 1.0 discourages repetition)
            min_len_eos: Minimum length before allowing <eos>
        
        Returns:
            sequences: (batch_size, max_len) - best sequences for each sample
        """
        batch_size = features.shape[0]
        device = features.device
        
        all_sequences = []
        
        # Process each sample independently
        for b in range(batch_size):
            sample_feat = features[b:b+1]  # (1, num_patches, feature_dim)
            
            # Initialize
            feat_pooled = sample_feat.mean(dim=1)
            feat_proj = self.feature_proj(feat_pooled)
            
            h = feat_proj.unsqueeze(0).repeat(self.num_layers, 1, 1)
            if self.embed_dim != self.hidden_dim:
                h = torch.randn(self.num_layers, 1, self.hidden_dim, device=device)
            c = torch.zeros(self.num_layers, 1, self.hidden_dim, device=device)
            
            # Beam search initialization
            beams = [([], 0.0, h, c)]  # (sequence, score, hidden, cell)
            completed_beams = []
            
            for t in range(max_len):
                candidates = []
                
                for seq, score, h_state, c_state in beams:
                    # Current input
                    if len(seq) == 0:
                        current_token = sos_id
                    else:
                        current_token = seq[-1]
                    
                    current_input = torch.tensor([[current_token]], dtype=torch.long, device=device)
                    embeds = self.embedding(current_input)
                    lstm_out, (h_new, c_new) = self.lstm(embeds, (h_state, c_state))
                    logits = self.fc(lstm_out.squeeze(1))  # (1, vocab_size)
                    log_probs = F.log_softmax(logits, dim=-1).squeeze(0)  # (vocab_size,)
                    
                    # Apply repetition penalty
                    if repetition_penalty != 1.0 and len(seq) > 0:
                        for prev_token in set(seq):
                            log_probs[prev_token] /= repetition_penalty
                    
                    # Block n-grams
                    if no_repeat_ngram_size > 0 and len(seq) >= no_repeat_ngram_size:
                        blocked_tokens = self._get_blocked_ngrams(seq, no_repeat_ngram_size)
                        for token in blocked_tokens:
                            log_probs[token] = float('-inf')
                    
                    # Block <eos> before min_len
                    if len(seq) < min_len_eos:
                        log_probs[eos_id] = float('-inf')
                    
                    # Get top-k tokens
                    topk_log_probs, topk_indices = torch.topk(log_probs, beam_size)
                    
                    for k in range(beam_size):
                        token = topk_indices[k].item()
                        token_score = topk_log_probs[k].item()
                        new_seq = seq + [token]
                        new_score = score + token_score
                        
                        if token == eos_id:
                            # Apply length penalty
                            length_norm = ((len(new_seq)) ** length_penalty)
                            normalized_score = new_score / length_norm
                            completed_beams.append((new_seq, normalized_score))
                        else:
                            candidates.append((new_seq, new_score, h_new, c_new))
                
                # Select top beam_size candidates
                if len(candidates) == 0:
                    break
                
                candidates.sort(key=lambda x: x[1], reverse=True)
                beams = candidates[:beam_size]
                
                # Early stopping if we have enough completed beams
                if len(completed_beams) >= beam_size:
                    break
            
            # Add remaining beams as completed
            for seq, score, _, _ in beams:
                if len(seq) > 0:
                    length_norm = ((len(seq)) ** length_penalty)
                    normalized_score = score / length_norm
                    completed_beams.append((seq, normalized_score))
            
            # Select best beam
            if len(completed_beams) > 0:
                completed_beams.sort(key=lambda x: x[1], reverse=True)
                best_seq = completed_beams[0][0]
            else:
                best_seq = [sos_id]
            
            # Convert to fixed-length sequence
            seq_tensor = torch.full((max_len,), pad_id, dtype=torch.long, device=device)
            for i, token in enumerate(best_seq[:max_len]):
                seq_tensor[i] = token
            
            all_sequences.append(seq_tensor)
        
        return torch.stack(all_sequences)
    
    def _get_blocked_ngrams(self, sequence, ngram_size):
        """
        Get tokens that would create repeated n-grams.
        
        Args:
            sequence: Current sequence (list of token ids)
            ngram_size: Size of n-grams to block
        
        Returns:
            Set of blocked token ids
        """
        blocked = set()
        
        if len(sequence) < ngram_size - 1:
            return blocked
        
        # Get the last (ngram_size - 1) tokens
        current_ngram_prefix = tuple(sequence[-(ngram_size - 1):])
        
        # Find all previous occurrences of this prefix
        for i in range(len(sequence) - ngram_size + 1):
            ngram_prefix = tuple(sequence[i:i + ngram_size - 1])
            if ngram_prefix == current_ngram_prefix:
                # Block the token that follows this prefix
                next_token = sequence[i + ngram_size - 1]
                blocked.add(next_token)
        
        return blocked
