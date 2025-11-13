"""
Tokenizer utilities for SentencePiece integration with custom vocabulary mapping.
Provides text_to_ids and ids_to_text functions that map between SentencePiece pieces
and a custom vocab.json for consistent encoding/decoding.
"""
import json
import re
import sentencepiece as sp


def load_tokenizer(sp_model_path, vocab_path, rev_vocab_path=None):
    """
    Load SentencePiece model and vocabulary mappings.
    
    Args:
        sp_model_path: Path to .model file for SentencePiece
        vocab_path: Path to vocab.json (piece -> id mapping)
        rev_vocab_path: Path to rev_vocab.json (id -> piece mapping), optional
    
    Returns:
        (sp_processor, vocab_dict, rev_vocab_dict)
    """
    sp_processor = sp.SentencePieceProcessor()
    sp_processor.load(sp_model_path)
    
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    
    if rev_vocab_path:
        with open(rev_vocab_path, 'r', encoding='utf-8') as f:
            rev_vocab = json.load(f)
    else:
        # Generate reverse vocab from vocab
        rev_vocab = {str(v): k for k, v in vocab.items()}
    
    return sp_processor, vocab, rev_vocab


def text_to_ids(text, sp_processor, vocab, unk_id=1):
    """
    Encode text to ids using SentencePiece pieces mapped through vocab.json.
    
    Args:
        text: Input text string
        sp_processor: SentencePiece processor
        vocab: Dictionary mapping pieces to ids
        unk_id: ID to use for unknown pieces (default 1)
    
    Returns:
        List of integer ids
    """
    # Encode text to pieces using SentencePiece
    pieces = sp_processor.encode_as_pieces(text)
    
    # Map pieces to ids via vocab.json with fallback
    ids = []
    for piece in pieces:
        if piece in vocab:
            ids.append(vocab[piece])
        else:
            # Fallback: try to find similar piece or use unk_id
            ids.append(unk_id)
    
    return ids


def ids_to_text(ids, sp_processor, rev_vocab, fallback=True):
    """
    Decode ids to text by mapping through rev_vocab.json to pieces,
    then decoding via sp.decode_pieces (NOT sp.decode_ids).
    
    Args:
        ids: List of integer ids
        sp_processor: SentencePiece processor
        rev_vocab: Dictionary mapping ids to pieces
        fallback: If True, use fallback method when decode_pieces fails
    
    Returns:
        Decoded text string
    """
    # Map ids to pieces via rev_vocab.json
    pieces = []
    for id_val in ids:
        id_str = str(id_val)
        if id_str in rev_vocab:
            piece = rev_vocab[id_str]
            # Skip special tokens like <pad>, <sos>, <eos>
            if not (piece.startswith('<') and piece.endswith('>')):
                pieces.append(piece)
        elif fallback:
            # Unknown id, skip it
            continue
    
    # Decode pieces using SentencePiece
    try:
        text = sp_processor.decode_pieces(pieces)
    except Exception:
        if fallback:
            # Fallback: manually replace '▁' with space
            text = ''.join(pieces).replace('▁', ' ').strip()
        else:
            raise
    
    return text


def normalize_detokenized(text):
    """
    Normalize detokenized text by cleaning repeated punctuation and tokenization artifacts.
    
    Args:
        text: Detokenized text string
    
    Returns:
        Normalized text string
    """
    # Remove repeated spaced punctuation like '. . . .'
    text = re.sub(r'(\s+[.!?,;:])+(\s+[.!?,;:])+', r'\1', text)
    
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Fix spaces before punctuation
    text = re.sub(r'\s+([.!?,;:])', r'\1', text)
    
    # Fix spaces after opening brackets/quotes
    text = re.sub(r'([({\["\'])\s+', r'\1', text)
    
    # Fix spaces before closing brackets/quotes
    text = re.sub(r'\s+([])}"\'])', r'\1', text)
    
    # Remove repeated identical tokens (e.g., "the the the" -> "the")
    words = text.split()
    cleaned_words = []
    prev_word = None
    for word in words:
        if word != prev_word:
            cleaned_words.append(word)
            prev_word = word
    
    text = ' '.join(cleaned_words).strip()
    
    return text
