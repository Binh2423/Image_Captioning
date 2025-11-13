"""
Example script demonstrating how to use the image captioning system.
This creates synthetic data and demonstrates the full workflow.
"""
import os
import json
import tempfile
import numpy as np
import torch
import sentencepiece as spm

from tokenizer_utils import load_tokenizer, text_to_ids, ids_to_text, normalize_detokenized
from model_vit_lstm import ViTLSTMCaptioner


def create_example_vocab():
    """Create a simple example vocabulary and SentencePiece model."""
    print("Creating example vocabulary...")
    
    # Create temporary directory
    tmpdir = tempfile.mkdtemp()
    
    # Create sample captions file
    captions = [
        "a cat sitting on a couch",
        "a dog playing in the park",
        "a bird flying in the sky",
        "a person riding a bike",
        "a car driving on the road"
    ] * 20  # Repeat to have enough training data
    
    captions_file = os.path.join(tmpdir, 'captions.txt')
    with open(captions_file, 'w') as f:
        for caption in captions:
            f.write(caption + '\n')
    
    # Train SentencePiece
    model_prefix = os.path.join(tmpdir, 'sp')
    spm.SentencePieceTrainer.train(
        input=captions_file,
        model_prefix=model_prefix,
        vocab_size=100,
        character_coverage=1.0,
        model_type='bpe',
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece='<pad>',
        unk_piece='<unk>',
        bos_piece='<sos>',
        eos_piece='<eos>'
    )
    
    # Create vocab.json and rev_vocab.json
    sp = spm.SentencePieceProcessor()
    sp.load(model_prefix + '.model')
    
    vocab = {}
    rev_vocab = {}
    for i in range(sp.get_piece_size()):
        piece = sp.id_to_piece(i)
        vocab[piece] = i
        rev_vocab[str(i)] = piece
    
    vocab_file = os.path.join(tmpdir, 'vocab.json')
    rev_vocab_file = os.path.join(tmpdir, 'rev_vocab.json')
    
    with open(vocab_file, 'w') as f:
        json.dump(vocab, f)
    
    with open(rev_vocab_file, 'w') as f:
        json.dump(rev_vocab, f)
    
    return model_prefix + '.model', vocab_file, rev_vocab_file, tmpdir


def example_tokenization():
    """Example: Text tokenization and detokenization."""
    print("\n" + "=" * 60)
    print("Example 1: Tokenization and Detokenization")
    print("=" * 60)
    
    sp_model, vocab_file, rev_vocab_file, tmpdir = create_example_vocab()
    
    # Load tokenizer
    sp_processor, vocab, rev_vocab = load_tokenizer(sp_model, vocab_file, rev_vocab_file)
    
    # Encode text
    text = "a cat sitting on a couch"
    ids = text_to_ids(text, sp_processor, vocab)
    print(f"Original text: '{text}'")
    print(f"Encoded IDs: {ids}")
    
    # Decode back
    decoded_text = ids_to_text(ids, sp_processor, rev_vocab)
    print(f"Decoded text: '{decoded_text}'")
    
    # Normalize
    normalized = normalize_detokenized(decoded_text)
    print(f"Normalized: '{normalized}'")
    
    # Cleanup
    import shutil
    shutil.rmtree(tmpdir)
    
    print("\n✓ Tokenization example completed")


def example_model_inference():
    """Example: Model inference with different decoding strategies."""
    print("\n" + "=" * 60)
    print("Example 2: Model Inference")
    print("=" * 60)
    
    sp_model, vocab_file, rev_vocab_file, tmpdir = create_example_vocab()
    sp_processor, vocab, rev_vocab = load_tokenizer(sp_model, vocab_file, rev_vocab_file)
    
    vocab_size = len(vocab)
    
    # Create model
    model = ViTLSTMCaptioner(
        feature_dim=768,
        embed_dim=256,
        hidden_dim=256,
        vocab_size=vocab_size,
        num_layers=1,
        dropout=0.1
    )
    model.eval()
    
    # Create synthetic features (simulating ViT patch features)
    batch_size = 1
    num_patches = 49  # 7x7 patches
    feature_dim = 768
    features = torch.randn(batch_size, num_patches, feature_dim)
    
    print(f"Input features shape: {features.shape}")
    
    # Get special token IDs
    pad_id = vocab['<pad>']
    sos_id = vocab['<sos>']
    eos_id = vocab['<eos>']
    max_len = 20
    
    # Greedy decoding
    print("\n1. Greedy Decoding:")
    with torch.no_grad():
        greedy_seq = model.greedy_decode(features, max_len, sos_id, eos_id, pad_id)
    
    greedy_text = ids_to_text(greedy_seq[0].tolist(), sp_processor, rev_vocab)
    greedy_text = normalize_detokenized(greedy_text)
    print(f"   Generated: '{greedy_text}'")
    
    # Sampling
    print("\n2. Sampling (temperature=1.0):")
    with torch.no_grad():
        sample_seq = model.sample_decode(features, max_len, sos_id, eos_id, pad_id, temperature=1.0)
    
    sample_text = ids_to_text(sample_seq[0].tolist(), sp_processor, rev_vocab)
    sample_text = normalize_detokenized(sample_text)
    print(f"   Generated: '{sample_text}'")
    
    # Beam search
    print("\n3. Beam Search (beam_size=3):")
    with torch.no_grad():
        beam_seq = model.beam_decode(
            features, max_len, sos_id, eos_id, pad_id,
            beam_size=3, length_penalty=1.0, no_repeat_ngram_size=2
        )
    
    beam_text = ids_to_text(beam_seq[0].tolist(), sp_processor, rev_vocab)
    beam_text = normalize_detokenized(beam_text)
    print(f"   Generated: '{beam_text}'")
    
    # Cleanup
    import shutil
    shutil.rmtree(tmpdir)
    
    print("\n✓ Model inference example completed")


def example_training_setup():
    """Example: How to set up training (without actually training)."""
    print("\n" + "=" * 60)
    print("Example 3: Training Setup")
    print("=" * 60)
    
    sp_model, vocab_file, rev_vocab_file, tmpdir = create_example_vocab()
    sp_processor, vocab, rev_vocab = load_tokenizer(sp_model, vocab_file, rev_vocab_file)
    
    vocab_size = len(vocab)
    
    # Create model
    model = ViTLSTMCaptioner(
        feature_dim=768,
        embed_dim=256,
        hidden_dim=256,
        vocab_size=vocab_size,
        num_layers=1,
        dropout=0.5
    )
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create loss function
    pad_id = vocab['<pad>']
    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_id)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Optimizer: Adam (lr=0.001)")
    print(f"Loss: CrossEntropyLoss (ignore_index={pad_id})")
    
    # Example forward pass
    batch_size = 4
    num_patches = 49
    feature_dim = 768
    max_len = 20
    
    features = torch.randn(batch_size, num_patches, feature_dim)
    captions = torch.randint(0, vocab_size, (batch_size, max_len))
    
    # Training step
    model.train()
    
    input_seq = captions[:, :-1]
    target_seq = captions[:, 1:]
    
    logits = model(features, input_seq)
    loss = criterion(logits.reshape(-1, vocab_size), target_seq.reshape(-1))
    
    print(f"\nExample batch:")
    print(f"  Features: {features.shape}")
    print(f"  Captions: {captions.shape}")
    print(f"  Logits: {logits.shape}")
    print(f"  Loss: {loss.item():.4f}")
    
    # Cleanup
    import shutil
    shutil.rmtree(tmpdir)
    
    print("\n✓ Training setup example completed")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Image Captioning System Examples")
    print("=" * 60)
    
    # Run examples
    example_tokenization()
    example_model_inference()
    example_training_setup()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully! ✓")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Extract features from your images using extract_features.py")
    print("2. Train a tokenizer on your captions using train_tokenizer.py")
    print("3. Train a model using train_xe.py")
    print("4. Fine-tune with SCST using scst_finetune.py")
    print("5. Find optimal beam search params using beam_grid_search.py")
    print("\nSee README.md for detailed instructions.")


if __name__ == '__main__':
    main()
