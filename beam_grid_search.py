"""
Grid search for beam search hyperparameters on validation set.
Searches over beam_size, length_penalty, and no_repeat_ngram_size.
"""
import os
import json
import argparse
import csv
import torch
from torch.utils.data import DataLoader
from itertools import product
from tqdm import tqdm
import sacrebleu

import tokenizer_utils
from dataset import CaptionPatchDataset, collate_fn
from model_vit_lstm import ViTLSTMCaptioner


def evaluate_with_params(model, dataloader, sp_processor, vocab, rev_vocab, device,
                        beam_size, length_penalty, no_repeat_ngram_size,
                        sos_id, eos_id, pad_id, max_len):
    """
    Evaluate model with specific beam search parameters.
    
    Returns:
        Dictionary with BLEU scores
    """
    model.eval()
    
    all_hypotheses = []
    all_references = []
    
    with torch.no_grad():
        for features, targets, _ in dataloader:
            features = features.to(device)
            batch_size = features.shape[0]
            
            # Generate with beam search
            generated = model.beam_decode(
                features, max_len, sos_id, eos_id, pad_id,
                beam_size=beam_size,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size
            )
            
            # Decode
            for i in range(batch_size):
                hyp_ids = generated[i].cpu().tolist()
                hyp_text = tokenizer_utils.ids_to_text(hyp_ids, sp_processor, rev_vocab)
                hyp_text = tokenizer_utils.normalize_detokenized(hyp_text)
                all_hypotheses.append(hyp_text)
                
                ref_ids = targets[i].cpu().tolist()
                ref_text = tokenizer_utils.ids_to_text(ref_ids, sp_processor, rev_vocab)
                ref_text = tokenizer_utils.normalize_detokenized(ref_text)
                all_references.append([ref_text])
    
    # Compute BLEU
    refs_flat = [[ref[0] for ref in all_references]]
    sacre_bleu = sacrebleu.corpus_bleu(all_hypotheses, refs_flat)
    
    return {
        'sacrebleu': sacre_bleu.score,
        'hypotheses': all_hypotheses[:5],
        'references': [r[0] for r in all_references[:5]]
    }


def main():
    parser = argparse.ArgumentParser(description='Grid search for beam search parameters')
    
    # Data arguments
    parser.add_argument('--val_annotations', type=str, required=True)
    parser.add_argument('--val_features_dir', type=str, required=True)
    parser.add_argument('--sp_model', type=str, required=True)
    parser.add_argument('--vocab_json', type=str, required=True)
    parser.add_argument('--rev_vocab_json', type=str, default=None)
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--feature_dim', type=int, default=768)
    parser.add_argument('--embed_dim', type=int, default=512)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.5)
    
    # Grid search parameters
    parser.add_argument('--beam_sizes', type=int, nargs='+', default=[3, 5, 7, 10],
                       help='Beam sizes to try')
    parser.add_argument('--length_penalties', type=float, nargs='+', default=[0.8, 1.0, 1.2, 1.5],
                       help='Length penalties to try')
    parser.add_argument('--no_repeat_ngram_sizes', type=int, nargs='+', default=[0, 2, 3],
                       help='N-gram blocking sizes to try')
    
    # Misc arguments
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_len', type=int, default=50)
    parser.add_argument('--normalize_features', action='store_true')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_csv', type=str, default='beam_search_results.csv')
    
    args = parser.parse_args()
    
    # Load vocabulary
    with open(args.vocab_json, 'r') as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
    
    pad_id = vocab.get('<pad>', 0)
    sos_id = vocab.get('<sos>', 2)
    eos_id = vocab.get('<eos>', 3)
    
    # Create dataset
    val_dataset = CaptionPatchDataset(
        args.val_annotations, args.val_features_dir,
        args.sp_model, args.vocab_json, args.rev_vocab_json,
        max_len=args.max_len, normalize_features=args.normalize_features
    )
    
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
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from {args.checkpoint}")
    
    # Load tokenizer
    sp_processor, vocab_dict, rev_vocab = tokenizer_utils.load_tokenizer(
        args.sp_model, args.vocab_json, args.rev_vocab_json
    )
    
    # Grid search
    param_grid = list(product(args.beam_sizes, args.length_penalties, args.no_repeat_ngram_sizes))
    
    print(f"Running grid search over {len(param_grid)} parameter combinations...")
    
    results = []
    
    for beam_size, length_penalty, no_repeat_ngram_size in tqdm(param_grid, desc="Grid search"):
        print(f"\nTesting: beam_size={beam_size}, length_penalty={length_penalty}, no_repeat_ngram_size={no_repeat_ngram_size}")
        
        eval_results = evaluate_with_params(
            model, val_loader, sp_processor, vocab_dict, rev_vocab, args.device,
            beam_size, length_penalty, no_repeat_ngram_size,
            sos_id, eos_id, pad_id, args.max_len
        )
        
        result = {
            'beam_size': beam_size,
            'length_penalty': length_penalty,
            'no_repeat_ngram_size': no_repeat_ngram_size,
            'sacrebleu': eval_results['sacrebleu']
        }
        
        results.append(result)
        print(f"SacreBLEU: {eval_results['sacrebleu']:.2f}")
    
    # Sort by BLEU score
    results.sort(key=lambda x: x['sacrebleu'], reverse=True)
    
    # Save to CSV
    with open(args.output_csv, 'w', newline='') as csvfile:
        fieldnames = ['beam_size', 'length_penalty', 'no_repeat_ngram_size', 'sacrebleu']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print(f"\nResults saved to {args.output_csv}")
    print("\nTop 5 configurations:")
    for i, result in enumerate(results[:5], 1):
        print(f"{i}. Beam={result['beam_size']}, Length Penalty={result['length_penalty']}, "
              f"No-Repeat N-gram={result['no_repeat_ngram_size']}: BLEU={result['sacrebleu']:.2f}")


if __name__ == '__main__':
    main()
