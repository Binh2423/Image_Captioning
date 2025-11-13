"""
Extract patch features from images using a Vision Transformer.
Saves features as .npz files with optional normalization.
"""
import os
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

try:
    import timm
except ImportError:
    print("Warning: timm not installed. Install with: pip install timm")
    timm = None


class ImageDataset(Dataset):
    """Simple dataset for loading images."""
    
    def __init__(self, image_dir, image_list=None, transform=None):
        """
        Args:
            image_dir: Directory containing images
            image_list: Optional list of image filenames
            transform: Image transformations
        """
        self.image_dir = image_dir
        self.transform = transform
        
        if image_list is not None:
            self.image_files = image_list
        else:
            # Get all image files
            self.image_files = [f for f in os.listdir(image_dir) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        self.image_files.sort()
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Get image ID (filename without extension)
        img_id = os.path.splitext(img_name)[0]
        
        return image, img_id


def extract_features(model, dataloader, device, normalize=False):
    """
    Extract features from images.
    
    Args:
        model: Vision model
        dataloader: DataLoader
        device: Device to run on
        normalize: Whether to L2 normalize features
    
    Returns:
        Dictionary mapping image_id to features
    """
    model.eval()
    features_dict = {}
    
    with torch.no_grad():
        for images, img_ids in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            
            # Forward pass to get patch features
            # For ViT models in timm, use forward_features to get patch tokens
            features = model.forward_features(images)
            
            # Remove CLS token if present (first token)
            if features.dim() == 3 and features.shape[1] > 1:
                # Assume first token is CLS token
                features = features[:, 1:, :]  # (batch_size, num_patches, feature_dim)
            
            # Normalize if requested
            if normalize:
                features = features / (features.norm(dim=-1, keepdim=True) + 1e-8)
            
            # Move to CPU and convert to numpy
            features = features.cpu().numpy()
            
            # Store features
            for i, img_id in enumerate(img_ids):
                features_dict[img_id] = features[i]
    
    return features_dict


def main():
    parser = argparse.ArgumentParser(description='Extract patch features from images')
    
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing images')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for features')
    parser.add_argument('--model_name', type=str, default='vit_base_patch16_224', 
                       help='timm model name')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--image_size', type=int, default=224, help='Input image size')
    parser.add_argument('--normalize', action='store_true', help='L2 normalize features')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing features')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check timm availability
    if timm is None:
        raise ImportError("timm is required. Install with: pip install timm")
    
    # Load model
    print(f"Loading model: {args.model_name}")
    model = timm.create_model(args.model_name, pretrained=True, num_classes=0)
    model = model.to(args.device)
    model.eval()
    
    # Create transform
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset and dataloader
    dataset = ImageDataset(args.image_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"Found {len(dataset)} images")
    
    # Extract features
    features_dict = extract_features(model, dataloader, args.device, normalize=args.normalize)
    
    # Save features
    print(f"Saving features to {args.output_dir}")
    for img_id, features in tqdm(features_dict.items(), desc="Saving"):
        output_path = os.path.join(args.output_dir, f"{img_id}.npz")
        
        # Skip if exists and not overwriting
        if os.path.exists(output_path) and not args.overwrite:
            continue
        
        # Save as npz with 'patch_feats' key
        np.savez_compressed(output_path, patch_feats=features)
    
    print(f"Saved features for {len(features_dict)} images")
    
    # Print feature statistics
    sample_features = next(iter(features_dict.values()))
    print(f"Feature shape: {sample_features.shape}")
    print(f"Feature dim: {sample_features.shape[-1]}")
    print(f"Num patches: {sample_features.shape[0]}")


if __name__ == '__main__':
    main()
