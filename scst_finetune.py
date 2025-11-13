"""
Self-Critical Sequence Training (SCST) for image captioning.
Uses CIDEr as the reward metric with optional XE mixing and entropy regularization.
"""
import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

try:
    from pycocoevalcap.cider.cider import Cider
except ImportError:
    print("Warning: pycocoevalcap not installed. Install with: pip install pycocoevalcap")
    Cider = None

import tokenizer_utils
from dataset import CaptionPatchDataset, collate_fn
from model_vit_lstm import ViTLSTMCaptioner


def compute_cider_scores(hypotheses, references):
    """
    Compute CIDEr scores for hypotheses given references.
    
    Args:
        hypotheses: List of hypothesis strings
        references: List of reference lists (each with one or more references)
    
    Returns:
        List of CIDEr scores (one per hypothesis)
    """
    if Cider is None:
        raise ImportError("pycocoevalcap not installed")
    
    # Format for CIDEr: dict of {id: [hypothesis]} and {id: [references]}
    hyp_dict = {i: [hyp] for i, hyp in enumerate(hypotheses)}
    ref_dict = {i: refs for i, refs in enumerate(references)}
    
    cider_scorer = Cider()
    score, scores = cider_scorer.compute_score(ref_dict, hyp_dict)
    
    return scores


def scst_train_epoch(model, dataloader, optimizer, sp_processor, vocab, rev_vocab, 
                    device, sos_id, eos_id, pad_id, max_len, baseline_method='greedy',
                    xe_weight=0.0, entropy_weight=0.0):
    """
    Train one epoch with SCST.
    
    Args:
        model: Captioning model
        dataloader: Training dataloader
        optimizer: Optimizer
        sp_processor: SentencePiece processor
        vocab: Vocabulary dict
        rev_vocab: Reverse vocabulary dict
        device: Device
        sos_id: Start token ID
        eos_id: End token ID
        pad_id: Padding token ID
        max_len: Maximum sequence length
        baseline_method: 'greedy' or 'sample' for baseline
        xe_weight: Weight for cross-entropy mixing (0.0 = pure SCST)
        entropy_weight: Weight for entropy regularization
    
    Returns:
        Average reward
    """
    model.train()
    total_reward = 0
    num_batches = 0
    
    for features, targets, _ in tqdm(dataloader, desc="SCST Training"):
        features = features.to(device)
        targets = targets.to(device)
        batch_size = features.shape[0]
        
        # Sample from model
        sample_seqs = model.sample_decode(features, max_len, sos_id, eos_id, pad_id, temperature=1.0)
        
        # Generate baseline
        if baseline_method == 'greedy':
            with torch.no_grad():
                baseline_seqs = model.greedy_decode(features, max_len, sos_id, eos_id, pad_id)
        else:
            with torch.no_grad():
                baseline_seqs = model.sample_decode(features, max_len, sos_id, eos_id, pad_id, temperature=1.0)
        
        # Decode sequences to text
        sample_texts = []
        baseline_texts = []
        reference_texts = []
        
        for i in range(batch_size):
            # Sample
            sample_ids = sample_seqs[i].cpu().tolist()
            sample_text = tokenizer_utils.ids_to_text(sample_ids, sp_processor, rev_vocab)
            sample_text = tokenizer_utils.normalize_detokenized(sample_text)
            sample_texts.append(sample_text)
            
            # Baseline
            baseline_ids = baseline_seqs[i].cpu().tolist()
            baseline_text = tokenizer_utils.ids_to_text(baseline_ids, sp_processor, rev_vocab)
            baseline_text = tokenizer_utils.normalize_detokenized(baseline_text)
            baseline_texts.append(baseline_text)
            
            # Reference
            ref_ids = targets[i].cpu().tolist()
            ref_text = tokenizer_utils.ids_to_text(ref_ids, sp_processor, rev_vocab)
            ref_text = tokenizer_utils.normalize_detokenized(ref_text)
            reference_texts.append([ref_text])
        
        # Compute CIDEr scores
        sample_scores = compute_cider_scores(sample_texts, reference_texts)
        baseline_scores = compute_cider_scores(baseline_texts, reference_texts)
        
        # Compute rewards (advantage)
        rewards = torch.tensor(sample_scores, device=device) - torch.tensor(baseline_scores, device=device)
        
        # Compute log probabilities for sampled sequences
        input_seq = sample_seqs[:, :-1]
        target_seq = sample_seqs[:, 1:]
        
        logits = model(features, input_seq)  # (batch_size, seq_len, vocab_size)
        log_probs = torch.log_softmax(logits, dim=-1)
        
        # Gather log probs of selected tokens
        selected_log_probs = log_probs.gather(2, target_seq.unsqueeze(2)).squeeze(2)
        
        # Mask padding
        mask = (target_seq != pad_id).float()
        selected_log_probs = selected_log_probs * mask
        
        # Sum log probs per sequence
        seq_log_probs = selected_log_probs.sum(dim=1)  # (batch_size,)
        
        # SCST loss (negative expected reward)
        scst_loss = -(seq_log_probs * rewards).mean()
        
        # Optional: Add cross-entropy mixing
        xe_loss = 0
        if xe_weight > 0:
            criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
            target_input = targets[:, :-1]
            target_output = targets[:, 1:]
            logits_xe = model(features, target_input)
            xe_loss = criterion(logits_xe.reshape(-1, logits_xe.shape[-1]), target_output.reshape(-1))
        
        # Optional: Add entropy regularization
        entropy_loss = 0
        if entropy_weight > 0:
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
            entropy_loss = -entropy.mean()  # Negative because we want to maximize entropy
        
        # Total loss
        loss = scst_loss + xe_weight * xe_loss + entropy_weight * entropy_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_reward += rewards.mean().item()
        num_batches += 1
    
    return total_reward / num_batches


def main():
    parser = argparse.ArgumentParser(description='SCST finetuning for image captioning')
    
    # Data arguments
    parser.add_argument('--train_annotations', type=str, required=True)
    parser.add_argument('--val_annotations', type=str, required=True)
    parser.add_argument('--train_features_dir', type=str, required=True)
    parser.add_argument('--val_features_dir', type=str, required=True)
    parser.add_argument('--sp_model', type=str, required=True)
    parser.add_argument('--vocab_json', type=str, required=True)
    parser.add_argument('--rev_vocab_json', type=str, default=None)
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to pretrained checkpoint')
    parser.add_argument('--feature_dim', type=int, default=768)
    parser.add_argument('--embed_dim', type=int, default=512)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.5)
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--max_len', type=int, default=50)
    parser.add_argument('--normalize_features', action='store_true')
    
    # SCST arguments
    parser.add_argument('--baseline_method', type=str, default='greedy', choices=['greedy', 'sample'])
    parser.add_argument('--xe_weight', type=float, default=0.0, help='Cross-entropy mixing weight')
    parser.add_argument('--entropy_weight', type=float, default=0.0, help='Entropy regularization weight')
    
    # Misc arguments
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir', type=str, default='scst_checkpoints')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load vocabulary
    with open(args.vocab_json, 'r') as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
    
    pad_id = vocab.get('<pad>', 0)
    sos_id = vocab.get('<sos>', 2)
    eos_id = vocab.get('<eos>', 3)
    
    # Create datasets
    train_dataset = CaptionPatchDataset(
        args.train_annotations, args.train_features_dir,
        args.sp_model, args.vocab_json, args.rev_vocab_json,
        max_len=args.max_len, normalize_features=args.normalize_features
    )
    
    val_dataset = CaptionPatchDataset(
        args.val_annotations, args.val_features_dir,
        args.sp_model, args.vocab_json, args.rev_vocab_json,
        max_len=args.max_len, normalize_features=args.normalize_features
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                             collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                           collate_fn=collate_fn, num_workers=4)
    
    # Create model
    model = ViTLSTMCaptioner(
        feature_dim=args.feature_dim,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        vocab_size=vocab_size,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(args.device)
    
    # Load pretrained checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded pretrained model from {args.checkpoint}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Load tokenizer
    sp_processor, vocab_dict, rev_vocab = tokenizer_utils.load_tokenizer(
        args.sp_model, args.vocab_json, args.rev_vocab_json
    )
    
    # Training loop
    best_reward = float('-inf')
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        
        # Train with SCST
        avg_reward = scst_train_epoch(
            model, train_loader, optimizer, sp_processor, vocab_dict, rev_vocab,
            args.device, sos_id, eos_id, pad_id, args.max_len,
            baseline_method=args.baseline_method,
            xe_weight=args.xe_weight,
            entropy_weight=args.entropy_weight
        )
        
        print(f"Average reward: {avg_reward:.4f}")
        
        # Save checkpoint
        if avg_reward > best_reward:
            best_reward = avg_reward
            checkpoint_path = os.path.join(args.save_dir, 'best_scst_model.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'reward': best_reward,
            }, checkpoint_path)
            print(f"Saved best model with reward: {best_reward:.4f}")
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(args.save_dir, f'scst_epoch_{epoch+1}.pt')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
    
    print("\nSCST training completed!")


if __name__ == '__main__':
    main()
