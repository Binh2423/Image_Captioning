
import argparse
import json
import os
import re
import sentencepiece as spm

def load_captions_from_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    mapping = {}
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, list):
                mapping[str(k)] = [str(x).strip() for x in v]
            else:
                mapping[str(k)] = [str(v).strip()]
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            caption = None
            img_id = None
            for key in ("caption", "captions", "text"):
                if key in item:
                    caption = item[key]
                    break
            for key in ("image", "img", "image_id", "file_name", "filename"):
                if key in item:
                    img_id = item[key]
                    break
            if img_id is None:
                img_id = str(len(mapping))
            if isinstance(caption, list):
                mapping[str(img_id)] = [str(x).strip() for x in caption]
            elif caption is not None:
                mapping[str(img_id)] = [str(caption).strip()]
    return mapping

def clean_text(s, lowercase=True, remove_punct=True):
    s = s.strip()
    if lowercase:
        s = s.lower()
    if remove_punct:
        s = re.sub(r"[^\w\s'\-]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
    return s

def write_captions_list(mapping, out_txt, lowercase=True, remove_punct=True):
    with open(out_txt, "w", encoding="utf-8") as f:
        for k, caps in mapping.items():
            for c in caps:
                if not isinstance(c, str):
                    continue
                txt = clean_text(c, lowercase=lowercase, remove_punct=remove_punct)
                if txt == "":
                    continue
                f.write(txt + "\n")

def build_vocab_json(sp_model_path, out_dir, add_special=True):
    sp = spm.SentencePieceProcessor(model_file=sp_model_path)
    sp_size = sp.get_piece_size()
    pieces = []
    for i in range(sp_size):
        piece = sp.id_to_piece(i)
        if piece == "<unk>":
            continue
        pieces.append(piece)
    token_to_id = {}
    cur = 4
    if add_special:
        token_to_id["<pad>"] = 0
        token_to_id["<bos>"] = 1
        token_to_id["<eos>"] = 2
        token_to_id["<unk>"] = 3
    for p in pieces:
        if p in token_to_id:
            continue
        token_to_id[p] = cur
        cur += 1
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(token_to_id, f, ensure_ascii=False, indent=2)
    rev = {str(v): k for k, v in token_to_id.items()}
    with open(os.path.join(out_dir, "rev_vocab.json"), "w", encoding="utf-8") as f:
        json.dump(rev, f, ensure_ascii=False, indent=2)
    return token_to_id, rev

def main(args):
    mapping = load_captions_from_json(args.captions)
    if not mapping:
        raise SystemExit("No captions found in " + args.captions)
    os.makedirs(args.output_dir, exist_ok=True)
    captions_txt = os.path.join(args.output_dir, "captions_all.txt")
    write_captions_list(mapping, captions_txt, lowercase=args.lowercase, remove_punct=args.remove_punct)
    model_prefix = os.path.join(args.output_dir, "sp")
    spm.SentencePieceTrainer.Train(input=captions_txt,
                                   model_prefix=model_prefix,
                                   vocab_size=args.vocab_size,
                                   model_type=args.model_type,
                                   character_coverage=args.character_coverage,
                                   bos_id=-1, eos_id=-1, unk_id=0)
    sp_model = model_prefix + ".model"
    token_to_id, rev = build_vocab_json(sp_model, args.output_dir, add_special=True)
    print("Wrote SentencePiece model to", sp_model)
    print("Wrote vocab.json and rev_vocab.json to", args.output_dir)
    print("Vocabulary size (including special tokens):", len(token_to_id))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--captions", default="D:/DACN/dataset/Flickr8k/Flickr8k_text/captions.json", help="path to captions json file")
    p.add_argument("--output_dir", default="D:/DACN/dataset/Flickr8k/tokenizer", help="output directory for tokenizer files")
    p.add_argument("--vocab_size", type=int, default=8000)
    p.add_argument("--model_type", type=str, default="bpe", choices=["unigram", "bpe", "word", "char"])
    p.add_argument("--character_coverage", type=float, default=1.0)
    p.add_argument("--lowercase", action="store_true", help="lowercase captions before training")
    p.add_argument("--remove_punct", action="store_true", help="remove punctuation during training corpus creation")
    args = p.parse_args()
    main(args)