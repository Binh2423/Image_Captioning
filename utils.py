import os
import torch

def save_checkpoint(state, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)

def load_checkpoint(path, device='cpu'):
    return torch.load(path, map_location=device)

def collate_fn(batch):
    # batch: list of (image, encoded_caption, img_name, captions)
    images = torch.stack([b[0] for b in batch], dim=0)
    captions = torch.stack([b[1] for b in batch], dim=0)
    img_names = [b[2] for b in batch]
    refs = [b[3] for b in batch]
    return images, captions, img_names, refs

def ids_to_sentences(id_seqs, vocab):
    # id_seqs: list of list of token ids (stop at end token), vocab: Vocabulary
    sents = []
    for seq in id_seqs:
        words = []
        for tid in seq:
            if tid < len(vocab.itos):
                w = vocab.itos[tid]
            else:
                w = '<unk>'
            if w == '<end>':
                break
            if w in ('<start>', '<pad>'):
                continue
            words.append(w)
        sents.append(' '.join(words))
    return sents