import os
import argparse
import time
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import ViTEncoder, DecoderLSTM
from dataset import Flickr8kDataset, Vocabulary
from utils import collate_fn, save_checkpoint, ids_to_sentences

# optional cider
try:
    from pycocoevalcap.cider.cider import Cider
    HAVE_CIDER = True
except Exception:
    HAVE_CIDER = False
    from nltk.translate.bleu_score import sentence_bleu

# set cuDNN benchmark for fixed-size inputs (can speed up convs/ViT)
torch.backends.cudnn.benchmark = True

def build_vocab_from_captions(root, min_freq=1):
    text_file = os.path.join(root, 'Flickr8k_text', 'Flickr8k.token.txt')
    caps = []
    with open(text_file, 'r', encoding='utf-8') as f:
        for line in f:
            _, cap = line.strip().split('\t')
            caps.append(cap)
    vocab = Vocabulary(min_freq=min_freq)
    vocab.build(caps)
    return vocab

def train_xe(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # build vocab
    vocab = build_vocab_from_captions(args.data_root, min_freq=1)
    print("Vocab size:", len(vocab))
    # DataLoader optimizations
    train_ds = Flickr8kDataset(args.data_root, split='train', vocab=vocab, max_len=args.max_len)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor
    )

    # models
    encoder = ViTEncoder(freeze=True).to(device)
    decoder = DecoderLSTM(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        enc_dim=encoder.hidden_size,
        dec_dim=args.dec_dim,
        attn_dim=args.attn_dim
    ).to(device)

    # Try to compile decoder (PyTorch >=2.x) for extra speed
    if args.use_compile:
        try:
            decoder = torch.compile(decoder)
            print("Compiled decoder with torch.compile")
        except Exception as e:
            print("torch.compile failed or not available:", e)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi['<pad>'])
    params = list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)

    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    global_step = 0
    os.makedirs('checkpoints', exist_ok=True)
    for epoch in range(args.epochs):
        decoder.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"XE Epoch {epoch+1}")
        for images, captions, img_names, refs in pbar:
            # non_blocking copy from pinned memory
            images = images.to(device, non_blocking=True)
            captions = captions.to(device, non_blocking=True)  # (B, T)
            inputs = captions[:, :-1]
            targets = captions[:, 1:]
            # if encoder is frozen, run it under no_grad to save memory/compute graph
            if all(not p.requires_grad for p in encoder.parameters()):
                with torch.no_grad():
                    enc_feats, cls = encoder(images)
            else:
                enc_feats, cls = encoder(images)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.use_amp):
                outputs, _ = decoder(inputs, cls, enc_feats)
                outputs = outputs.reshape(-1, outputs.size(-1))
                targets_flat = targets.reshape(-1)
                loss = criterion(outputs, targets_flat)

            if args.use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            global_step += 1
            if global_step % args.log_interval == 0:
                pbar.set_postfix({"loss": loss.item()})
        avg = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} avg loss: {avg:.4f}")
        # save checkpoint (reduce io overhead by infrequent saving)
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint({
                'epoch': epoch+1,
                'decoder_state': decoder.state_dict() if not hasattr(decoder, "__wrapped__") else decoder.__wrapped__.state_dict(),
                'vocab': vocab.itos
            }, f'checkpoints/xe_epoch{epoch+1}.pth')


def compute_reward(generated_sents, baseline_sents, refs_dict):
    if HAVE_CIDER:
        cider = Cider()
        G = {k: refs_dict[k] for k in refs_dict}
        score_g, _ = cider.compute_score(G, {k: [generated_sents[k]] for k in generated_sents})
        score_b, _ = cider.compute_score(G, {k: [baseline_sents[k]] for k in baseline_sents})
        advantage = score_g - score_b
        advs = {k: advantage for k in generated_sents}
        return advs
    else:
        advs = {}
        for k in generated_sents:
            g = generated_sents[k].split()
            b = baseline_sents[k].split()
            refs = [r.split() for r in refs_dict[k]]
            rw_g = sentence_bleu(refs, g, weights=(0.5, 0.5))
            rw_b = sentence_bleu(refs, b, weights=(0.5, 0.5))
            advs[k] = rw_g - rw_b
        return advs


def train_scst(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load vocab from XE checkpoint
    assert args.xe_checkpoint is not None
    cp = torch.load(args.xe_checkpoint, map_location='cpu')
    itos = cp['vocab']
    vocab = Vocabulary()
    vocab.itos = itos
    vocab.stoi = {w: i for i, w in enumerate(itos)}

    train_ds = Flickr8kDataset(args.data_root, split='train', vocab=vocab, max_len=args.max_len)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor
    )

    encoder = ViTEncoder(freeze=True).to(device)
    decoder = DecoderLSTM(vocab_size=len(vocab), embed_dim=args.embed_dim,
                          enc_dim=encoder.hidden_size, dec_dim=args.dec_dim, attn_dim=args.attn_dim).to(device)

    if 'decoder_state' in cp:
        decoder.load_state_dict(cp['decoder_state'])
        print("Loaded decoder state from XE checkpoint.")

    if args.use_compile:
        try:
            decoder = torch.compile(decoder)
            print("Compiled decoder with torch.compile")
        except Exception as e:
            print("torch.compile failed or not available:", e)

    optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        decoder.train()
        pbar = tqdm(train_loader, desc=f"SCST Epoch {epoch+1}")
        for images, captions, img_names, refs in pbar:
            images = images.to(device, non_blocking=True)
            # encoder is frozen -> run in no_grad
            with torch.no_grad():
                enc_feats, cls = encoder(images)

            # sampling (no grad) to get baseline and sampled sentences
            sampled_ids = decoder.sample_sequence(cls, enc_feats, start_token=vocab.stoi['<start>'],
                                                  end_token=vocab.stoi['<end>'], max_len=args.max_len, sample=True)
            greedy_ids = decoder.sample_sequence(cls, enc_feats, start_token=vocab.stoi['<start>'],
                                                 end_token=vocab.stoi['<end>'], max_len=args.max_len, sample=False)

            sampled_sents = ids_to_sentences(sampled_ids, vocab)
            greedy_sents = ids_to_sentences(greedy_ids, vocab)

            gen_dict = {}
            base_dict = {}
            refs_dict = {}
            for i, name in enumerate(img_names):
                key = name
                gen_dict[key] = sampled_sents[i]
                base_dict[key] = greedy_sents[i]
                refs_dict[key] = refs[i]

            advs = compute_reward(gen_dict, base_dict, refs_dict)

            # build policy gradient loss: -adv * log p(y^s)
            B = images.size(0)
            device = images.device
            h, c = decoder.init_states(cls)
            optimizer.zero_grad(set_to_none=True)
            policy_loss = 0.0
            # compute per-timestep log-probs with gradient enabled
            for t in range(args.max_len):
                words_t = []
                for i in range(B):
                    if t < len(sampled_ids[i]):
                        words_t.append(sampled_ids[i][t])
                    else:
                        words_t.append(vocab.stoi['<pad>'])
                words_t = torch.tensor(words_t, dtype=torch.long, device=device)
                out, h, c, _ = decoder.forward_step(decoder.embed(words_t), h, c, enc_feats)
                logp = torch.log_softmax(out, dim=-1)
                selected_logp = logp[range(B), words_t]
                adv = torch.tensor([advs[name] for name in img_names], device=device, dtype=torch.float)
                policy_loss = policy_loss - (adv * selected_logp).mean()

            policy_loss = policy_loss / args.max_len
            policy_loss.backward()
            optimizer.step()
            pbar.set_postfix({"pg_loss": policy_loss.item()})
        save_checkpoint({
            'epoch': epoch+1,
            'decoder_state': decoder.state_dict() if not hasattr(decoder, "__wrapped__") else decoder.__wrapped__.state_dict(),
            'vocab': vocab.itos
        }, f'checkpoints/scst_epoch{epoch+1}.pth')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True, help='root folder of Flickr8k dataset')
    parser.add_argument('--phase', choices=['xe', 'scst'], default='xe')
    parser.add_argument('--xe_checkpoint', default=None)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_len', type=int, default=30)
    parser.add_argument('--embed_dim', type=int, default=512)
    parser.add_argument('--dec_dim', type=int, default=512)
    parser.add_argument('--attn_dim', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--log_interval', type=int, default=100)
    # Performance flags
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--prefetch_factor', type=int, default=2)
    parser.add_argument('--use_amp', type=bool, default=True)
    parser.add_argument('--use_compile', type=bool, default=False)
    parser.add_argument('--save_every', type=int, default=1)
    args = parser.parse_args()

    if args.phase == 'xe':
        train_xe(args)
    else:
        train_scst(args)


if __name__ == '__main__':
    main()