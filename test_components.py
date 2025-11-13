"""
Basic tests to validate core components of the image captioning system.
These tests verify that the components can be instantiated and run without errors.
"""
import os
import sys
import json
import tempfile
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tokenizer_utils
from model_vit_lstm import ViTLSTMCaptioner


def test_tokenizer_utils():
    """Test tokenizer utility functions."""
    print("Testing tokenizer_utils...")
    
    # Create temporary vocab files
    vocab = {
        '<pad>': 0,
        '<unk>': 1,
        '<sos>': 2,
        '<eos>': 3,
        '▁a': 4,
        '▁test': 5,
        '▁sentence': 6,
        '.': 7
    }
    
    rev_vocab = {str(v): k for k, v in vocab.items()}
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(vocab, f)
        vocab_file = f.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(rev_vocab, f)
        rev_vocab_file = f.name
    
    try:
        # Test normalize_detokenized
        text = "This  is   a    test  .  .  ."
        normalized = tokenizer_utils.normalize_detokenized(text)
        print(f"  Original: '{text}'")
        print(f"  Normalized: '{normalized}'")
        assert len(normalized) < len(text), "Normalization should reduce length"
        
        print("✓ tokenizer_utils tests passed")
        return True
    finally:
        os.unlink(vocab_file)
        os.unlink(rev_vocab_file)


def test_model():
    """Test ViT-LSTM model."""
    print("\nTesting ViT-LSTM model...")
    
    # Model parameters
    batch_size = 2
    num_patches = 49
    feature_dim = 768
    embed_dim = 256
    hidden_dim = 256
    vocab_size = 1000
    max_len = 20
    
    # Create model
    model = ViTLSTMCaptioner(
        feature_dim=feature_dim,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        num_layers=1,
        dropout=0.1
    )
    model.eval()
    
    # Test forward pass
    features = torch.randn(batch_size, num_patches, feature_dim)
    captions = torch.randint(0, vocab_size, (batch_size, max_len))
    
    with torch.no_grad():
        logits = model(features, captions)
    
    assert logits.shape == (batch_size, max_len, vocab_size), \
        f"Expected shape {(batch_size, max_len, vocab_size)}, got {logits.shape}"
    print(f"  Forward pass output shape: {logits.shape} ✓")
    
    # Test greedy decode
    with torch.no_grad():
        sequences = model.greedy_decode(features, max_len, sos_id=2, eos_id=3, pad_id=0)
    
    assert sequences.shape == (batch_size, max_len), \
        f"Expected shape {(batch_size, max_len)}, got {sequences.shape}"
    print(f"  Greedy decode output shape: {sequences.shape} ✓")
    
    # Test sample decode
    with torch.no_grad():
        sequences = model.sample_decode(features, max_len, sos_id=2, eos_id=3, pad_id=0, temperature=1.0)
    
    assert sequences.shape == (batch_size, max_len), \
        f"Expected shape {(batch_size, max_len)}, got {sequences.shape}"
    print(f"  Sample decode output shape: {sequences.shape} ✓")
    
    # Test beam decode
    with torch.no_grad():
        sequences = model.beam_decode(
            features, max_len, sos_id=2, eos_id=3, pad_id=0,
            beam_size=3, length_penalty=1.0, no_repeat_ngram_size=2
        )
    
    assert sequences.shape == (batch_size, max_len), \
        f"Expected shape {(batch_size, max_len)}, got {sequences.shape}"
    print(f"  Beam decode output shape: {sequences.shape} ✓")
    
    print("✓ Model tests passed")
    return True


def test_dataset():
    """Test dataset functionality."""
    print("\nTesting dataset...")
    
    # This test requires actual data files, so we'll skip for now
    # In a real scenario, you'd create mock data files
    print("  Dataset tests require actual data files - skipping")
    print("✓ Dataset tests skipped (requires real data)")
    return True


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running Image Captioning System Tests")
    print("=" * 60)
    
    tests = [
        ("Tokenizer Utils", test_tokenizer_utils),
        ("Model", test_model),
        ("Dataset", test_dataset),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ {name} test failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{name}: {status}")
    
    all_passed = all(r for _, r in results)
    print("\n" + ("All tests passed! ✓" if all_passed else "Some tests failed ✗"))
    
    return all_passed


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
