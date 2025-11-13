"""
Cross-entropy training script for image captioning.
Includes proper evaluation with detokenization and BLEU/sacreBLEU computation.
"""
import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import sacrebleu
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

import tokenizer_utils
from dataset import CaptionPatchDataset, collate_fn
from model_vit_lstm import ViTLSTMCaptioner


def evaluate(model, dataloader, sp_processor, vocab, rev_vocab, device, 
            use_beam=False, beam_size=5, beam_length_penalty=1.0, 
            no_repeat_ngram_size=0, sos_id=2, eos_id=3, pad_id=0, max_len=50):
    """
    Evaluate model on validation set with BLEU scores.
    Decodes both hypotheses and references before computing metrics.
    
    Args:
        model: Captioning model
        dataloader: Validation dataloader
        sp_processor: SentencePiece processor
        vocab: Vocabulary dictionary
        rev_vocab: Reverse vocabulary dictionary
        device: Device to run on
        use_beam: Whether to use beam search
        beam_size: Beam size for beam search
        beam_length_penalty: Length penalty for beam search
        no_repeat_ngram_size: N-gram blocking size
        sos_id: Start token ID
        eos_id: End token ID
        pad_id: Padding token ID
        max_len: Maximum sequence length
    
    Returns:
        Dictionary with BLEU scores and sample outputs
    """
    model.eval()
    
    all_hypotheses = []
    all_references = []
    
    with torch.no_grad():
        for features, targets, img_ids in tqdm(dataloader, desc="Evaluating"):
            features = features.to(device)
            batch_size = features.shape[0]
            
            # Generate captions
            if use_beam:
                generated = model.beam_decode(
                    features, max_len, sos_id, eos_id, pad_id,
                    beam_size=beam_size, 
                    length_penalty=beam_length_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size
                )
            else:
                generated = model.greedy_decode(features, max_len, sos_id, eos_id, pad_id)
            
            # Decode hypotheses
            for i in range(batch_size):
                # Hypothesis
                hyp_ids = generated[i].cpu().tolist()
                hyp_text = tokenizer_utils.ids_to_text(hyp_ids, sp_processor, rev_vocab)
                hyp_text = tokenizer_utils.normalize_detokenized(hyp_text)
                all_hypotheses.append(hyp_text)
                
                # Reference
                ref_ids = targets[i].cpu().tolist()
                # Handle case where refs might be stored as stringified lists
                if isinstance(ref_ids, str):
                    try:
                        ref_ids = json.loads(ref_ids)
                    except:
                        ref_ids = [int(x) for x in ref_ids.split() if x.isdigit()]
                
                ref_text = tokenizer_utils.ids_to_text(ref_ids, sp_processor, rev_vocab)
                ref_text = tokenizer_utils.normalize_detokenized(ref_text)
                all_references.append([ref_text])  # BLEU expects list of references per hypothesis
    
    # Compute BLEU scores
    # NLTK BLEU (corpus-level)
    smoothing = SmoothingFunction()
    bleu1 = corpus_bleu(all_references, all_hypotheses, weights=(1.0, 0, 0, 0), 
                        smoothing_function=smoothing.method1)
    bleu4 = corpus_bleu(all_references, all_hypotheses, weights=(0.25, 0.25, 0.25, 0.25),
                        smoothing_function=smoothing.method1)
    
    # SacreBLEU
    # Flatten references for sacrebleu (expects list of reference lists)
    refs_flat = [[ref[0] for ref in all_references]]
    sacre_bleu = sacrebleu.corpus_bleu(all_hypotheses, refs_flat)
    
    results = {
        'bleu1': bleu1 * 100,
        'bleu4': bleu4 * 100,
        'sacrebleu': sacre_bleu.score,
        'samples': list(zip(all_hypotheses[:5], [r[0] for r in all_references[:5]]))
    }
    
    return results


def train_epoch(model, dataloader, criterion, optimizer, device, vocab_size, pad_id):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for features, targets, _ in tqdm(dataloader, desc="Training"):
        features = features.to(device)
        targets = targets.to(device)
        
        # Forward pass
        # Input: all tokens except last, Target: all tokens except first
        input_seq = targets[:, :-1]
        target_seq = targets[:, 1:]
        
        logits = model(features, input_seq)
        
        # Compute loss
        loss = criterion(logits.reshape(-1, vocab_size), target_seq.reshape(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description='Train image captioning model with cross-entropy')
    
    # Data arguments
    parser.add_argument('--train_annotations', type=str, required=True, help='Path to training annotations JSON')
    parser.add_argument('--val_annotations', type=str, required=True, help='Path to validation annotations JSON')
    parser.add_argument('--train_features_dir', type=str, required=True, help='Directory with training features')
    parser.add_argument('--val_features_dir', type=str, required=True, help='Directory with validation features')
    parser.add_argument('--sp_model', type=str, required=True, help='Path to SentencePiece model')
    parser.add_argument('--vocab_json', type=str, required=True, help='Path to vocab.json')
    parser.add_argument('--rev_vocab_json', type=str, default=None, help='Path to rev_vocab.json')
    
    # Model arguments
    parser.add_argument('--feature_dim', type=int, default=768, help='Feature dimension')
    parser.add_argument('--embed_dim', type=int, default=512, help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=512, help='LSTM hidden dimension')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout probability')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--max_len', type=int, default=50, help='Maximum sequence length')
    parser.add_argument('--normalize_features', action='store_true', help='L2 normalize features')
    
    # Evaluation arguments
    parser.add_argument('--use_beam', action='store_true', help='Use beam search for evaluation')
    parser.add_argument('--beam_size', type=int, default=5, help='Beam size')
    parser.add_argument('--beam_length_penalty', type=float, default=1.0, help='Length penalty for beam search')
    parser.add_argument('--no_repeat_ngram_size', type=int, default=0, help='Block repeated n-grams')
    
    # Misc arguments
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--eval_every', type=int, default=1, help='Evaluate every N epochs')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load vocabulary to get vocab size
    with open(args.vocab_json, 'r') as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
    
    # Get special token IDs
    pad_id = vocab.get('<pad>', 0)
    sos_id = vocab.get('<sos>', 2)
    eos_id = vocab.get('<eos>', 3)
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Special tokens - PAD: {pad_id}, SOS: {sos_id}, EOS: {eos_id}")
    
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
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
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
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Load tokenizer for evaluation
    sp_processor, vocab_dict, rev_vocab = tokenizer_utils.load_tokenizer(
        args.sp_model, args.vocab_json, args.rev_vocab_json
    )
    
    # Training loop
    best_bleu4 = 0.0
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, args.device, vocab_size, pad_id)
        print(f"Train loss: {train_loss:.4f}")
        
        # Evaluate
        if (epoch + 1) % args.eval_every == 0:
            results = evaluate(
                model, val_loader, sp_processor, vocab_dict, rev_vocab, args.device,
                use_beam=args.use_beam, beam_size=args.beam_size,
                beam_length_penalty=args.beam_length_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                sos_id=sos_id, eos_id=eos_id, pad_id=pad_id, max_len=args.max_len
            )
            
            print(f"BLEU-1: {results['bleu1']:.2f}, BLEU-4: {results['bleu4']:.2f}, SacreBLEU: {results['sacrebleu']:.2f}")
            print("\nSample outputs:")
            for i, (hyp, ref) in enumerate(results['samples']):
                print(f"  {i+1}. Hyp: {hyp}")
                print(f"     Ref: {ref}")
            
            # Save best model
            if results['bleu4'] > best_bleu4:
                best_bleu4 = results['bleu4']
                checkpoint_path = os.path.join(args.save_dir, 'best_model.pt')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'bleu4': best_bleu4,
                }, checkpoint_path)
                print(f"Saved best model with BLEU-4: {best_bleu4:.2f}")
        
        # Save checkpoint every epoch
        checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pt')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
    
    print("\nTraining completed!")


if __name__ == '__main__':
    main()
