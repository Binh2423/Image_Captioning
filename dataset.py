"""
Dataset for image captioning with patch features.
Encodes captions on-the-fly using tokenizer_utils and handles various feature formats.
"""
import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
import tokenizer_utils


class CaptionPatchDataset(Dataset):
    """
    Dataset for image captioning with pre-extracted patch features.
    Encodes captions on-the-fly and supports multiple feature formats (.npy, .npz, .pt, .pth).
    """
    
    def __init__(self, annotations_file, features_dir, sp_model_path, vocab_path, 
                 rev_vocab_path=None, max_len=50, normalize_features=True):
        """
        Args:
            annotations_file: Path to JSON file with captions (COCO format or custom)
            features_dir: Directory containing pre-extracted features
            sp_model_path: Path to SentencePiece model
            vocab_path: Path to vocab.json
            rev_vocab_path: Path to rev_vocab.json (optional)
            max_len: Maximum caption length (will pad/truncate)
            normalize_features: Whether to L2-normalize patch features
        """
        self.features_dir = features_dir
        self.max_len = max_len
        self.normalize_features = normalize_features
        
        # Load tokenizer
        self.sp_processor, self.vocab, self.rev_vocab = tokenizer_utils.load_tokenizer(
            sp_model_path, vocab_path, rev_vocab_path
        )
        
        # Load annotations
        with open(annotations_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Parse annotations (support COCO format)
        self.samples = []
        if 'annotations' in data:
            # COCO format
            img_to_captions = {}
            for ann in data['annotations']:
                img_id = ann['image_id']
                caption = ann['caption']
                if img_id not in img_to_captions:
                    img_to_captions[img_id] = []
                img_to_captions[img_id].append(caption)
            
            # Create samples
            for img_id, captions in img_to_captions.items():
                for caption in captions:
                    self.samples.append({
                        'image_id': img_id,
                        'caption': caption
                    })
        else:
            # Custom format: assume list of {image_id, caption}
            self.samples = data
        
        # Get special token IDs
        self.pad_id = self.vocab.get('<pad>', 0)
        self.sos_id = self.vocab.get('<sos>', 2)
        self.eos_id = self.vocab.get('<eos>', 3)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Returns:
            features: Tensor of shape (num_patches, feature_dim)
            target_ids: Tensor of shape (max_len,) with <sos> caption <eos> <pad>...
            image_id: Image identifier
        """
        sample = self.samples[idx]
        image_id = sample['image_id']
        caption = sample['caption']
        
        # Load features
        features = self._load_features(image_id)
        
        # Normalize features if requested
        if self.normalize_features and features is not None:
            # L2 normalize each patch
            features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
        
        # Encode caption
        caption_ids = tokenizer_utils.text_to_ids(caption, self.sp_processor, self.vocab)
        
        # Add <sos> and <eos>
        target_ids = [self.sos_id] + caption_ids + [self.eos_id]
        
        # Pad or truncate to max_len
        if len(target_ids) > self.max_len:
            target_ids = target_ids[:self.max_len]
        else:
            target_ids = target_ids + [self.pad_id] * (self.max_len - len(target_ids))
        
        # Convert to tensors
        if features is not None:
            features = torch.from_numpy(features).float()
        else:
            # Return dummy features if file not found
            features = torch.zeros(1, 768).float()
        
        target_ids = torch.tensor(target_ids, dtype=torch.long)
        
        return features, target_ids, image_id
    
    def _load_features(self, image_id):
        """
        Load features from various formats (.npy, .npz, .pt, .pth).
        Robust feature file discovery.
        """
        # Try different extensions
        extensions = ['.npy', '.npz', '.pt', '.pth']
        feature_file = None
        
        for ext in extensions:
            # Try with and without prefix/suffix variations
            candidates = [
                os.path.join(self.features_dir, f"{image_id}{ext}"),
                os.path.join(self.features_dir, f"{image_id:012d}{ext}"),  # COCO format
                os.path.join(self.features_dir, str(image_id), f"features{ext}"),
            ]
            
            for candidate in candidates:
                if os.path.exists(candidate):
                    feature_file = candidate
                    break
            
            if feature_file:
                break
        
        if not feature_file:
            print(f"Warning: Feature file not found for image_id {image_id}")
            return None
        
        # Load based on extension
        ext = os.path.splitext(feature_file)[1]
        
        if ext == '.npy':
            features = np.load(feature_file)
        elif ext == '.npz':
            data = np.load(feature_file)
            # Try common keys
            if 'features' in data:
                features = data['features']
            elif 'patch_feats' in data:
                features = data['patch_feats']
            else:
                # Take first array
                features = data[list(data.keys())[0]]
        elif ext in ['.pt', '.pth']:
            features = torch.load(feature_file, map_location='cpu')
            if isinstance(features, torch.Tensor):
                features = features.numpy()
            elif isinstance(features, dict):
                if 'features' in features:
                    features = features['features']
                    if isinstance(features, torch.Tensor):
                        features = features.numpy()
                else:
                    features = features[list(features.keys())[0]]
                    if isinstance(features, torch.Tensor):
                        features = features.numpy()
        else:
            return None
        
        # Ensure 2D shape
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        return features


def collate_fn(batch):
    """
    Collate function for DataLoader.
    
    Args:
        batch: List of (features, target_ids, image_id) tuples
    
    Returns:
        features_batch: Tensor of shape (batch_size, num_patches, feature_dim)
        targets_batch: Tensor of shape (batch_size, max_len)
        image_ids: List of image identifiers
    """
    features_list = []
    targets_list = []
    image_ids = []
    
    # Find max number of patches in batch
    max_patches = max(f.shape[0] for f, _, _ in batch)
    feature_dim = batch[0][0].shape[1]
    
    for features, targets, img_id in batch:
        # Pad features to max_patches
        if features.shape[0] < max_patches:
            padding = torch.zeros(max_patches - features.shape[0], feature_dim)
            features = torch.cat([features, padding], dim=0)
        
        features_list.append(features)
        targets_list.append(targets)
        image_ids.append(img_id)
    
    features_batch = torch.stack(features_list)
    targets_batch = torch.stack(targets_list)
    
    return features_batch, targets_batch, image_ids
