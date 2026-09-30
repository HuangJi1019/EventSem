"""TACoS MSSE semantic feature generation (v2).

Differences from the original (v1) generator:
  1. Regroup CLIP BPE pieces back into whole words via the '</w>' marker before
     querying WordNet/GloVe; the resulting word vector is copied to every token
     position the word occupies, so the output stays aligned with CLIP tokenization.
  2. Eq.(3) OOV fallback is actually implemented: mean of the word's CLIP token
     embeddings, mapped 512 -> 300 by a FIXED, NON-TRAINABLE random projection
     (seed 2024).  NOTE: the 512->300 map is our own design decision, the paper
     does not specify how the CLIP (512d) and GloVe (300d) spaces are joined.
     The fallback is rescaled to the mean norm of the GloVe path so it does not
     silently degenerate to ~0 when --no_norm_tfeat is used.
  3. WordNet lookup optionally expands over n/v/a/r lemma forms.  NOTE: nltk's
     wordnet.synsets() already runs morphy internally, so this does NOT add
     inflection handling -- it adds cross-POS senses.  Kept behind a switch so it
     can be ablated separately.
"""
import os
import sys
import json
import numpy as np
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from transformers import CLIPTokenizer, CLIPTextModel
import nltk

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# --- Configuration ---
OUTPUT_DIR = 'datasets/semantic_embeddings/tacos-token-level-v2'
GLOVE_PATH = 'glove.6B.300d.txt'
GLOVE_DIM = 300
# The stored TACoS clip_text_features were extracted with HF truncation at 32 tokens,
# so 119/13791 long queries are capped at 32 rows there.  The semantic features are mixed
# with them ELEMENTWISE in model.py, so they must be produced the same way -- a bare
# encode() would emit up to 49 rows and blow up (or silently misalign) on those queries.
MAX_Q_L = 32                  # == max_length used for the stored clip_text_features
USE_LEMMA_EXPANSION = False   # fix (3): OFF by default -- measured to add cross-POS
                              # noise, not inflection handling (nltk's synsets() already
                              # runs morphy).  Set True only to ablate it.

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- GloVe ---
print(f"Loading GloVe embeddings from {GLOVE_PATH}...")
glove_vocab = {'<unk>': 0}
glove_embeddings = [np.zeros(GLOVE_DIM, dtype=np.float32)]
with open(GLOVE_PATH, 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f):
        parts = line.rstrip().split(' ')
        word = parts[0]
        vector = np.asarray(parts[1:], dtype=np.float32)
        if len(vector) != GLOVE_DIM:
            print(f"Warning: Line {line_num+1} has incorrect dimension. Skipping.")
            continue
        glove_vocab[word] = len(glove_embeddings)
        glove_embeddings.append(vector)
glove_embeddings = np.asarray(glove_embeddings, dtype=np.float32)


class DummyWordVectors:
    def __init__(self, vocab, embeddings, vector_size):
        self.vocab, self.vectors, self.vector_size = vocab, embeddings, vector_size

    def __contains__(self, word):
        return word in self.vocab

    def __getitem__(self, word):
        return self.vectors[self.vocab.get(word, 0)]


word_vectors = DummyWordVectors(glove_vocab, glove_embeddings, GLOVE_DIM)
print(f"Finished loading GloVe. Vocab size: {len(glove_vocab)}, shape: {glove_embeddings.shape}")

clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# --- CLIP token embedding table, for the Eq.(3)-style OOV fallback ---
_clip_text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True)
CLIP_TOKEN_EMBED = _clip_text_model.text_model.embeddings.token_embedding.weight.detach().numpy()
del _clip_text_model
_rng = np.random.RandomState(2024)
_CLIP_TO_GLOVE_PROJ = _rng.normal(0, 1 / np.sqrt(CLIP_TOKEN_EMBED.shape[1]),
                                  size=(CLIP_TOKEN_EMBED.shape[1], GLOVE_DIM)).astype(np.float32)
_GLOVE_PATH_NORM = 3.30  # measured mean ||.|| of the GloVe/WordNet path (charades train)

lemmatizer = WordNetLemmatizer()
_SEM_CACHE = {}


def get_all_synsets(word):
    candidates = {word}
    if USE_LEMMA_EXPANSION:
        for pos in ('n', 'v', 'a', 'r'):
            try:
                candidates.add(lemmatizer.lemmatize(word, pos=pos))
            except Exception:
                pass
    synsets, seen = [], set()
    for cand in candidates:
        for syn in wordnet.synsets(cand):
            if syn.name() not in seen:
                seen.add(syn.name())
                synsets.append(syn)
    return synsets


def extract_word_semantic_relations(word):
    related_words = {word}
    for syn in get_all_synsets(word):
        for lemma in syn.lemmas():
            related_words.add(lemma.name().replace('_', ' ').lower())
        for hypernym in syn.hypernyms():
            for lemma in hypernym.lemmas():
                related_words.add(lemma.name().replace('_', ' ').lower())
        for hyponym in sorted(syn.hyponyms(), key=lambda x: x.name())[:3]:
            for lemma in hyponym.lemmas():
                related_words.add(lemma.name().replace('_', ' ').lower())
        for meronym in syn.part_meronyms():
            for lemma in meronym.lemmas():
                related_words.add(lemma.name().replace('_', ' ').lower())
    # sorted(): a set iterates in hash order, which changes the float32 summation order
    # of the mean below. Sorting makes generation bit-for-bit reproducible.
    return sorted(related_words)


def clip_subword_fallback_embedding(clip_token_ids_for_word):
    v = CLIP_TOKEN_EMBED[clip_token_ids_for_word].mean(axis=0) @ _CLIP_TO_GLOVE_PROJ
    n = np.linalg.norm(v)
    if n < 1e-8:
        return v.astype(np.float32)
    return (v / n * _GLOVE_PATH_NORM).astype(np.float32)


def word_to_semantic_embedding(word, clip_token_ids_for_word):
    if word in _SEM_CACHE:
        return _SEM_CACHE[word]
    related_words = extract_word_semantic_relations(word)
    embeddings = [word_vectors[w] for w in related_words if w in word_vectors]
    emb = (np.mean(embeddings, axis=0).astype(np.float32) if embeddings
           else clip_subword_fallback_embedding(clip_token_ids_for_word))
    _SEM_CACHE[word] = emb
    return emb


def group_into_words(token_ids):
    """-> list of (token_positions, is_special); consecutive BPE pieces of one word share a span."""
    raw_tokens = clip_tokenizer.convert_ids_to_tokens(token_ids)
    special_ids = {clip_tokenizer.bos_token_id, clip_tokenizer.eos_token_id, clip_tokenizer.pad_token_id}
    spans, current = [], []
    for i, (tid, raw) in enumerate(zip(token_ids, raw_tokens)):
        if tid in special_ids:
            if current:
                spans.append((current, False))
                current = []
            spans.append(([i], True))
            continue
        current.append(i)
        if raw.endswith('</w>'):
            spans.append((current, False))
            current = []
    if current:
        spans.append((current, False))
    return spans


def clip_word_level_semantic_embedding(sentence):
    if MAX_Q_L > 0:
        # HF truncation keeps the first MAX_Q_L-1 tokens plus <|endoftext|>; a bare
        # [:MAX_Q_L] slice would end mid-word and not match the stored text features.
        token_ids = clip_tokenizer(sentence, truncation=True, max_length=MAX_Q_L)['input_ids']
    else:
        token_ids = clip_tokenizer.encode(sentence)
    if not token_ids:
        return np.zeros((1, GLOVE_DIM), dtype=np.float32)

    embeddings = [None] * len(token_ids)
    for positions, is_special in group_into_words(token_ids):
        if is_special:
            emb = word_vectors['<unk>']
        else:
            piece_ids = [token_ids[p] for p in positions]
            raw_pieces = clip_tokenizer.convert_ids_to_tokens(piece_ids)
            word = ''.join(p.replace('</w>', '') for p in raw_pieces).lower()
            emb = word_to_semantic_embedding(word, piece_ids)
        for p in positions:
            embeddings[p] = emb
    return np.stack(embeddings).astype(np.float32)


def process_jsonl_clip_aligned(jsonl_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    n = 0
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            emb = clip_word_level_semantic_embedding(data['query'])
            np.save(os.path.join(output_dir, f"{data['qid']}.npy"), emb)
            n += 1
    print(f"Done: {jsonl_file} -> {output_dir}  ({n} queries)")


if __name__ == '__main__':
    # python linguistic_knowledge_tacos_v2.py                      -> train + test splits into OUTPUT_DIR
    # python linguistic_knowledge_tacos_v2.py <query.jsonl> <out_dir> -> one query file (e.g. SRE)
    # The SRE files reuse the SAME qids as the test split with different query text, so
    # always write them to a separate out_dir.
    if len(sys.argv) == 3:
        process_jsonl_clip_aligned(sys.argv[1], sys.argv[2])
    else:
        for split_file in ['data/tacos/train.jsonl',
                       'data/tacos/test.jsonl']:
            process_jsonl_clip_aligned(split_file, OUTPUT_DIR)
    print(f"Done (word-level v2). cache={len(_SEM_CACHE)} word types")
