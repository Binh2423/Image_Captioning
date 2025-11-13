"""
Train SentencePiece tokenizer and generate vocab.json and rev_vocab.json with special tokens.
"""
import os
import json
import argparse
import sentencepiece as spm


def train_sentencepiece(input_file, model_prefix, vocab_size=10000, 
                       character_coverage=1.0, model_type='bpe'):
    """
    Train SentencePiece model.
    
    Args:
        input_file: Text file with training data (one sentence per line)
        model_prefix: Prefix for output files (.model and .vocab)
        vocab_size: Vocabulary size
        character_coverage: Character coverage (1.0 for all characters)
        model_type: Model type ('bpe' or 'unigram')
    """
    # Define special tokens
    user_defined_symbols = []
    
    # Train SentencePiece
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type=model_type,
        pad_id=0,
        unk_id=1,
        bos_id=2,  # Will map to <sos>
        eos_id=3,
        pad_piece='<pad>',
        unk_piece='<unk>',
        bos_piece='<sos>',
        eos_piece='<eos>',
        user_defined_symbols=user_defined_symbols
    )
    
    print(f"SentencePiece model trained: {model_prefix}.model")


def generate_vocab_files(sp_model_path, vocab_json_path, rev_vocab_json_path):
    """
    Generate vocab.json and rev_vocab.json from SentencePiece model.
    
    Args:
        sp_model_path: Path to .model file
        vocab_json_path: Output path for vocab.json (piece -> id)
        rev_vocab_json_path: Output path for rev_vocab.json (id -> piece)
    """
    # Load SentencePiece model
    sp = spm.SentencePieceProcessor()
    sp.load(sp_model_path)
    
    # Create vocab dictionaries
    vocab = {}
    rev_vocab = {}
    
    for i in range(sp.get_piece_size()):
        piece = sp.id_to_piece(i)
        vocab[piece] = i
        rev_vocab[str(i)] = piece
    
    # Ensure special tokens are present
    special_tokens = {
        '<pad>': 0,
        '<unk>': 1,
        '<sos>': 2,
        '<eos>': 3
    }
    
    for token, idx in special_tokens.items():
        if token not in vocab:
            vocab[token] = idx
        if str(idx) not in rev_vocab:
            rev_vocab[str(idx)] = token
    
    # Save vocab.json
    with open(vocab_json_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    
    # Save rev_vocab.json
    with open(rev_vocab_json_path, 'w', encoding='utf-8') as f:
        json.dump(rev_vocab, f, ensure_ascii=False, indent=2)
    
    print(f"Vocabulary files created:")
    print(f"  - {vocab_json_path} ({len(vocab)} entries)")
    print(f"  - {rev_vocab_json_path} ({len(rev_vocab)} entries)")


def prepare_captions_file(annotations_file, output_file, clean=True):
    """
    Extract captions from annotations JSON and prepare text file for training.
    
    Args:
        annotations_file: Path to annotations JSON (COCO format or custom)
        output_file: Output text file path
        clean: Whether to clean captions
    """
    with open(annotations_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    captions = []
    
    if 'annotations' in data:
        # COCO format
        for ann in data['annotations']:
            caption = ann['caption']
            if clean:
                caption = caption.strip().lower()
            captions.append(caption)
    elif isinstance(data, list):
        # Custom format: list of {caption: ...} or {text: ...}
        for item in data:
            caption = item.get('caption') or item.get('text', '')
            if clean:
                caption = caption.strip().lower()
            captions.append(caption)
    else:
        raise ValueError("Unknown annotations format")
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        for caption in captions:
            f.write(caption + '\n')
    
    print(f"Prepared {len(captions)} captions in {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Train SentencePiece tokenizer for image captioning')
    
    parser.add_argument('--annotations', type=str, required=True, 
                       help='Path to annotations JSON file')
    parser.add_argument('--output_dir', type=str, default='tokenizer',
                       help='Output directory for tokenizer files')
    parser.add_argument('--model_prefix', type=str, default='sp',
                       help='Prefix for SentencePiece model files')
    parser.add_argument('--vocab_size', type=int, default=10000,
                       help='Vocabulary size')
    parser.add_argument('--character_coverage', type=float, default=1.0,
                       help='Character coverage')
    parser.add_argument('--model_type', type=str, default='bpe', choices=['bpe', 'unigram'],
                       help='SentencePiece model type')
    parser.add_argument('--no_clean', action='store_true',
                       help='Do not clean captions (keep original case and whitespace)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare captions file
    captions_file = os.path.join(args.output_dir, 'captions.txt')
    prepare_captions_file(args.annotations, captions_file, clean=not args.no_clean)
    
    # Train SentencePiece
    model_path = os.path.join(args.output_dir, args.model_prefix)
    train_sentencepiece(
        captions_file, model_path, 
        vocab_size=args.vocab_size,
        character_coverage=args.character_coverage,
        model_type=args.model_type
    )
    
    # Generate vocab files
    sp_model_path = model_path + '.model'
    vocab_json_path = os.path.join(args.output_dir, 'vocab.json')
    rev_vocab_json_path = os.path.join(args.output_dir, 'rev_vocab.json')
    
    generate_vocab_files(sp_model_path, vocab_json_path, rev_vocab_json_path)
    
    print("\nTokenizer training completed!")
    print(f"Files created in {args.output_dir}:")
    print(f"  - {args.model_prefix}.model (SentencePiece model)")
    print(f"  - {args.model_prefix}.vocab (SentencePiece vocab)")
    print(f"  - vocab.json (piece -> id mapping)")
    print(f"  - rev_vocab.json (id -> piece mapping)")


if __name__ == '__main__':
    main()
