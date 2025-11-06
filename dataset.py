import os
import re
from collections import Counter
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch
import glob

class Vocabulary:
    """
    Lightweight tokenizer-based vocabulary using regex tokenization (no NLTK dependency).
    """
    def __init__(self, min_freq=1, specials=['<pad>', '<start>', '<end>', '<unk>']):
        self.min_freq = min_freq
        self.freq = Counter()
        self.itos = []
        self.stoi = {}
        self.specials = list(specials)

    @staticmethod
    def _tokenize(text):
        # Simple fast tokenizer: words and punctuation
        # lowercase, capture words/numbers and punctuation tokens
        return re.findall(r"[A-Za-z0-9']+|[^\sA-Za-z0-9']", text.lower())

    def build(self, sentences):
        for s in sentences:
            tokens = self._tokenize(s)
            self.freq.update(tokens)
        # start with specials
        self.itos = list(self.specials)
        for word, cnt in self.freq.most_common():
            if cnt >= self.min_freq and word not in self.specials:
                self.itos.append(word)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode_sentence(self, s, max_len=30):
        tokens = self._tokenize(s)
        start_idx = self.stoi.get('<start>')
        end_idx = self.stoi.get('<end>')
        unk_idx = self.stoi.get('<unk>')
        ids = [start_idx]
        for t in tokens[: max_len - 2]:
            ids.append(self.stoi.get(t, unk_idx))
        ids.append(end_idx)
        # pad will be handled by dataset/collate
        return ids

    def decode_ids(self, ids):
        words = []
        for i in ids:
            if i < len(self.itos):
                w = self.itos[i]
            else:
                w = '<unk>'
            if w in ('<start>', '<end>', '<pad>'):
                continue
            words.append(w)
        return ' '.join(words)

class SimpleVocab:
    # Giả định class đã có các thuộc tính sau:
    # self.stoi: dict mapping token -> idx
    # self.pad_idx: int index cho padding
    # self.unk_idx: int index cho unknown token
    # (Nếu chưa có, hãy tạo chúng trong __init__)

    def encode_sentence(self, sentence, max_len=None, add_eos=False, eos_token='<eos>'):
        """
        Chuyển sentence (string) -> tensor chỉ số độ dài max_len.
        - sentence: string
        - max_len: int hoặc None. Nếu None, trả về độ dài token hiện có.
        - add_eos: nếu True, thêm token eos (và tính vào max_len).
        - Trả về: torch.LongTensor shape (L,) với L = max_len nếu max_len được truyền, ngược lại length thực tế.
        """
        # 1) Tokenize (thay bằng tokenizer của bạn nếu cần)
        if isinstance(sentence, (list, tuple)):
            tokens = list(sentence)
        else:
            tokens = sentence.strip().split()

        # 2) Map token -> index, dùng unk nếu không tồn tại
        indices = [self.stoi.get(t, getattr(self, 'unk_idx', 1)) for t in tokens]

        # 3) Thêm eos nếu cần
        if add_eos:
            eos_idx = self.stoi.get(eos_token, None)
            if eos_idx is None:
                # nếu vocab không có eos token, thêm vào bằng unk_idx
                eos_idx = getattr(self, 'unk_idx', 1)
            indices.append(eos_idx)

        # 4) Nếu max_len được truyền, cắt hoặc pad
        if max_len is not None:
            if len(indices) > max_len:
                indices = indices[:max_len]
            else:
                pad_idx = getattr(self, 'pad_idx', 0)
                indices = indices + [pad_idx] * (max_len - len(indices))

        # 5) Trả về tensor long (hoặc list nếu bạn muốn)
        return torch.tensor(indices, dtype=torch.long)
class Flickr8kDataset(Dataset):
    """
    Flickr8k dataset loader.
    Returns:
      image (Tensor), encoded_caption (LongTensor), image_filename (str), list_of_refs (list[str])
    Notes:
      - Keep transforms light and deterministic for better cuDNN kernel selection.
      - This version tries multiple common image-folder names and will search for images if needed.
    """
    def __init__(self, root, split='train', transform=None, vocab=None, max_len=30):
        self.root = root
        text_dir = os.path.join(root, 'Flickr8k_text')
        token_file = os.path.join(text_dir, 'Flickr8k.token.txt')
        if not os.path.exists(token_file):
            raise FileNotFoundError(f"Cannot find {token_file}. Place Flickr8k.token.txt in {text_dir}")
        # read tokens
        self.captions = {}  # img -> list of captions
        with open(token_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                parts = line.strip().split('\t')
                if len(parts) < 2:
                    continue
                key, cap = parts[0], parts[1]
                img = key.split('#')[0]
                self.captions.setdefault(img, []).append(cap)

        # splits
        split_file = os.path.join(text_dir, f'Flickr_8k.{split}Images.txt')
        if not os.path.exists(split_file):
            img_list = list(self.captions.keys())
        else:
            with open(split_file, 'r', encoding='utf-8') as f:
                img_list = [l.strip() for l in f.readlines() if l.strip()]

        # try several common image directory names
        candidate_dirs = [
            os.path.join(root, 'Flickr_8k_Dataset'),
            os.path.join(root, 'Flickr8k_Dataset'),
            os.path.join(root, 'Flickr_8k_dataset'),
            os.path.join(root, 'images'),
            root  # fallback: images might be directly in root
        ]
        img_dir = None
        for d in candidate_dirs:
            if os.path.exists(d) and any(f.lower().endswith('.jpg') or f.lower().endswith('.jpeg') for f in os.listdir(d)):
                img_dir = d
                break

        if img_dir is None:
            # try to find any jpg under root recursively
            found = glob.glob(os.path.join(root, '**', '*.jpg'), recursive=True)
            if found:
                # pick parent folder of first found as img_dir
                img_dir = os.path.dirname(found[0])
                print(f"[dataset] Warning: using discovered image folder: {img_dir}")
            else:
                raise FileNotFoundError(f"Cannot find image directory under {root}. Expected one of {candidate_dirs} or any .jpg file recursively.")

        # build full image paths, but check existence. If missing, try to search for the file anywhere under root.
        resolved_images = []
        missing_imgs = []
        for im in img_list:
            candidate = os.path.join(img_dir, im)
            if os.path.exists(candidate):
                resolved_images.append(candidate)
            else:
                # try to find im anywhere under root
                matches = glob.glob(os.path.join(root, '**', im), recursive=True)
                if matches:
                    resolved_images.append(matches[0])
                else:
                    missing_imgs.append(im)

        if missing_imgs:
            print(f"[dataset] Warning: {len(missing_imgs)} referenced images not found. They will be skipped. Example missing: {missing_imgs[:5]}")

        self.images = resolved_images
        # default transform (ImageNet-style) - keep deterministic resize and normalization
        if transform is None:
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transform

        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img_name = os.path.basename(img_path)
        # Load image (PIL) and apply transforms (fast if num_workers>0)
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        captions = self.captions.get(img_name, [])
        caption = captions[0] if captions else ""
        if self.vocab is not None:
            encoded = self.vocab.encode_sentence(caption, max_len=self.max_len)
            # pad/truncate to max_len
            if len(encoded) < self.max_len:
                pad_idx = self.vocab.stoi.get('<pad>', 0)
                encoded = encoded + [pad_idx] * (self.max_len - len(encoded))
            else:
                encoded = encoded[:self.max_len]
            return image, torch.tensor(encoded, dtype=torch.long), img_name, captions
        else:
            return image, caption, img_name, captions