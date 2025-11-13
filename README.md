# Image Captioning with Vision Transformer and LSTM

A complete image captioning system using Vision Transformer (ViT) features and LSTM decoder with advanced decoding strategies including beam search, n-gram blocking, and self-critical sequence training (SCST).

## Features

- **SentencePiece Tokenization**: Consistent encoding/decoding with custom vocabulary mapping
- **Robust Dataset**: On-the-fly caption encoding with support for multiple feature formats (.npy, .npz, .pt)
- **LSTM Decoder**: Powerful decoder with multiple decoding strategies:
  - Greedy decoding
  - Sampling-based decoding
  - Beam search with length penalty, n-gram blocking, and repetition penalty
- **Cross-Entropy Training**: Standard supervised training with BLEU/SacreBLEU evaluation
- **SCST Finetuning**: Self-critical sequence training using CIDEr rewards
- **Beam Search Grid Search**: Automatic hyperparameter tuning for beam search
- **Feature Extraction**: Extract ViT patch features from images

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Extract Features from Images

Extract patch features from your images using a pretrained Vision Transformer:

```bash
python extract_features.py \
    --image_dir /path/to/images \
    --output_dir /path/to/features \
    --model_name vit_base_patch16_224 \
    --normalize \
    --batch_size 32
```

### 2. Train SentencePiece Tokenizer

Train a SentencePiece tokenizer on your caption data:

```bash
python train_tokenizer.py \
    --annotations /path/to/captions.json \
    --output_dir tokenizer \
    --vocab_size 10000 \
    --model_type bpe
```

This creates:
- `sp.model` - SentencePiece model
- `vocab.json` - Piece to ID mapping
- `rev_vocab.json` - ID to piece mapping

### 3. Train with Cross-Entropy

Train the captioning model using cross-entropy loss:

```bash
python train_xe.py \
    --train_annotations /path/to/train_captions.json \
    --val_annotations /path/to/val_captions.json \
    --train_features_dir /path/to/train_features \
    --val_features_dir /path/to/val_features \
    --sp_model tokenizer/sp.model \
    --vocab_json tokenizer/vocab.json \
    --rev_vocab_json tokenizer/rev_vocab.json \
    --feature_dim 768 \
    --embed_dim 512 \
    --hidden_dim 512 \
    --batch_size 32 \
    --num_epochs 30 \
    --lr 0.001 \
    --normalize_features \
    --use_beam \
    --beam_size 5 \
    --beam_length_penalty 1.0 \
    --no_repeat_ngram_size 3 \
    --save_dir checkpoints
```

### 4. Fine-tune with SCST

Fine-tune the model using self-critical sequence training for better CIDEr scores:

```bash
python scst_finetune.py \
    --train_annotations /path/to/train_captions.json \
    --val_annotations /path/to/val_captions.json \
    --train_features_dir /path/to/train_features \
    --val_features_dir /path/to/val_features \
    --sp_model tokenizer/sp.model \
    --vocab_json tokenizer/vocab.json \
    --rev_vocab_json tokenizer/rev_vocab.json \
    --checkpoint checkpoints/best_model.pt \
    --batch_size 16 \
    --num_epochs 10 \
    --lr 0.00001 \
    --baseline_method greedy \
    --xe_weight 0.1 \
    --entropy_weight 0.01 \
    --save_dir scst_checkpoints
```

### 5. Grid Search for Beam Search Parameters

Find optimal beam search hyperparameters:

```bash
python beam_grid_search.py \
    --val_annotations /path/to/val_captions.json \
    --val_features_dir /path/to/val_features \
    --sp_model tokenizer/sp.model \
    --vocab_json tokenizer/vocab.json \
    --rev_vocab_json tokenizer/rev_vocab.json \
    --checkpoint checkpoints/best_model.pt \
    --beam_sizes 3 5 7 10 \
    --length_penalties 0.8 1.0 1.2 1.5 \
    --no_repeat_ngram_sizes 0 2 3 \
    --output_csv beam_search_results.csv
```

## Data Format

### Annotations JSON

Supports COCO format:

```json
{
  "annotations": [
    {
      "image_id": 123456,
      "caption": "A cat sitting on a couch"
    }
  ]
}
```

Or custom format:

```json
[
  {
    "image_id": "img001",
    "caption": "A cat sitting on a couch"
  }
]
```

### Feature Files

Features should be saved in one of these formats:
- `.npy` - NumPy array of shape (num_patches, feature_dim)
- `.npz` - Compressed NumPy with key 'patch_feats' or 'features'
- `.pt` or `.pth` - PyTorch tensor

## Model Architecture

- **Input**: Pre-extracted patch features from ViT (default: 768-dim)
- **Embedding**: Word embeddings (default: 512-dim)
- **Decoder**: LSTM with configurable hidden size and layers
- **Output**: Projection to vocabulary size

## Decoding Strategies

### Greedy Decoding
Simple argmax decoding - fast but may not produce optimal results.

### Sampling
Sample from the probability distribution - introduces diversity.

### Beam Search
Advanced beam search with:
- **Length Penalty**: Prevents bias towards shorter sequences
- **N-gram Blocking**: Prevents repetitive n-grams
- **Repetition Penalty**: Discourages token repetition
- **Minimum Length for EOS**: Prevents premature stopping

## Evaluation Metrics

- **BLEU-1, BLEU-4**: Standard BLEU scores (NLTK)
- **SacreBLEU**: Standardized BLEU implementation
- **CIDEr**: Consensus-based Image Description Evaluation (for SCST)

## Key Components

### tokenizer_utils.py
- `load_tokenizer()`: Load SentencePiece model and vocab
- `text_to_ids()`: Encode text using SentencePiece pieces mapped to vocab.json
- `ids_to_text()`: Decode IDs via rev_vocab.json and SentencePiece
- `normalize_detokenized()`: Clean up tokenization artifacts

### dataset.py
- `CaptionPatchDataset`: Dataset with on-the-fly encoding
- `collate_fn()`: Batch collation with padding

### model_vit_lstm.py
- `ViTLSTMCaptioner`: Main model class
- `greedy_decode()`: Greedy decoding
- `sample_decode()`: Sampling-based decoding
- `beam_decode()`: Beam search with advanced features

### train_xe.py
- Cross-entropy training loop
- Evaluation with proper detokenization
- BLEU score computation

### scst_finetune.py
- Self-critical sequence training
- CIDEr-based rewards
- Optional XE mixing and entropy regularization

## Tips

1. **Start with XE training**: Train with cross-entropy first to get a good baseline
2. **Then use SCST**: Fine-tune with SCST for metric optimization
3. **Grid search beam params**: Use beam_grid_search.py to find optimal decoding parameters
4. **Normalize features**: Use `--normalize_features` for better training stability
5. **Use beam search for evaluation**: Beam search typically produces better captions than greedy

## License

MIT License

## Citation

If you use this code, please cite:
```
@software{image_captioning_2025,
  title={Image Captioning with ViT and LSTM},
  author={Image Captioning Contributors},
  year={2025}
}
```
