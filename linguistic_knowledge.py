import os
import json
import numpy as np
from nltk.corpus import wordnet
from transformers import CLIPTokenizer
import nltk 

# Download WordNet data if you haven't already
nltk.download('wordnet')

# --- Configuration ---

OUTPUT_DIR = 'datasets/semantic_embeddings/charades-sta_token_level_wordnet' # Your desired output directory
# Ensure GloVe word vector file exists. Download from https://nlp.stanford.edu/projects/glove/
# For example, glove.6B.zip, then extract glove.6B.100d.txt
GLOVE_PATH = 'glove.6B.300d.txt' # !!! REPLACE WITH YOUR ACTUAL GLOVE FILE PATH !!!
GLOVE_DIM = 300 # Dimension of your GloVe vectors (e.g., 50d, 100d, 200d, 300d)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Initialize GloVe Vectors ---
print(f"Loading GloVe embeddings from {GLOVE_PATH}...")
glove_vocab = {}
glove_embeddings = []
# Add an Unknown (UNK) vector, typically all zeros
glove_embeddings.append(np.zeros(GLOVE_DIM, dtype=np.float32)) # Index 0 for UNK
glove_vocab['<unk>'] = 0 # Map '<unk>' to index 0

with open(GLOVE_PATH, 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f):
        if line_num % 100000 == 0:
            print(f"  Processed {line_num} lines...")
        parts = line.strip().split()
        word = parts[0]
        vector = np.array(parts[1:], dtype=np.float32)
        if len(vector) != GLOVE_DIM:
            print(f"Warning: Line {line_num+1} in GloVe file has incorrect dimension. Skipping.")
            continue
        glove_vocab[word] = len(glove_embeddings)
        glove_embeddings.append(vector)

glove_embeddings = np.array(glove_embeddings)
# Create a dummy word_vectors object to mimic gensim's KeyedVectors for compatibility
# This assumes the GLOVE_DIM is the vector_size
class DummyWordVectors:
    def __init__(self, vocab, embeddings, vector_size):
        self.vocab = vocab
        self.vectors = embeddings
        self.vector_size = vector_size
    def __contains__(self, word):
        return word in self.vocab
    def __getitem__(self, word):
        return self.vectors[self.vocab.get(word, self.vocab['<unk>'])]

word_vectors = DummyWordVectors(glove_vocab, glove_embeddings, GLOVE_DIM)

print(f"Finished loading GloVe. Vocab size: {len(glove_vocab)}, Embedding shape: {glove_embeddings.shape}")

clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")


def extract_token_semantic_relations(word, max_depth=1):
    """
    Extract synonyms and hierarchical relationships for a given word
    max_depth: how deep to go in the hierarchy (1 = direct relations only)
    """
    related_words = set([word])
    
    for syn in wordnet.synsets(word):
        # 1. Synonyms (existing)
        for lemma in syn.lemmas():
            related_words.add(lemma.name().replace('_', ' ').lower())
        
        # 2. Hypernyms (more general concepts)
        for hypernym in syn.hypernyms():
            if max_depth >= 1:
                for lemma in hypernym.lemmas():
                    related_words.add(lemma.name().replace('_', ' ').lower())
        
        # 3. Hyponyms (more specific concepts) - optional, might add noise
        for hyponym in syn.hyponyms()[:3]:  # Limit to avoid too many
            if max_depth >= 1:
                for lemma in hyponym.lemmas():
                    related_words.add(lemma.name().replace('_', ' ').lower())
        
        # 4. Meronyms (part-of relationships) - optional
        for meronym in syn.part_meronyms():
            for lemma in meronym.lemmas():
                related_words.add(lemma.name().replace('_', ' ').lower())
    
    return list(related_words)

def token_to_semantic_embedding(token, word_vectors):
    cleaned_token = token.replace('</w>', '')
    # Use the enhanced function
    related_words = extract_token_semantic_relations(cleaned_token, max_depth=1)
    
    embeddings = [word_vectors[w] for w in related_words if w in word_vectors]
    
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return word_vectors['<unk>']

# Generate CLIP-aligned token-level semantic embedding for a sentence
# Generate CLIP-aligned token-level semantic embedding for a sentence
def clip_token_level_semantic_embedding(sentence, word_vectors):
    """
    Generates token-level GloVe embeddings (averaged with synonyms) aligned with CLIP subwords.
    This version includes CLIP's start/end of text tokens.
    Returns: np.ndarray of shape [seq_len, D_sem]
    """
    # 1) Use CLIP's tokenizer to get token IDs and then decode back to strings.
    # This automatically includes <|startoftext|> and <|endoftext|> tokens.
    # We use encode() to get the full sequence with special tokens.
    token_ids = clip_tokenizer.encode(sentence)

    token_strings = [clip_tokenizer.decode([_id], skip_special_tokens=False) for _id in token_ids]


    embeddings = []
    for tok_str in token_strings:

        is_special_token = (tok_str == clip_tokenizer.bos_token) or \
                           (tok_str == clip_tokenizer.eos_token) or \
                           (tok_str == clip_tokenizer.pad_token) 

        if is_special_token:

            emb = word_vectors['<unk>']
        else:

            processed_token = tok_str.lower().replace('Ġ', '')
            emb = token_to_semantic_embedding(processed_token, word_vectors)
        embeddings.append(emb)


    if not embeddings:
        return np.array([word_vectors['<unk>']], dtype=np.float32)


    return np.stack(embeddings) 

def process_jsonl_clip_aligned(jsonl_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                qid = data['qid']
                query_text = data['query']

                print(f"Processing query {qid}...") 

                emb = clip_token_level_semantic_embedding(query_text, word_vectors)

                output_path = os.path.join(output_dir, f"{qid}.npy")
                np.save(output_path, emb)
    print("Done: all queries processed and saved.")


jsonl_file = 'data/highlight_train_release.jsonl'  # Replace with your input file path
output_dir = 'datasets/semantic_embeddings/charades-sta_token_level_wordnet'  # Output directory
process_jsonl_clip_aligned(jsonl_file, output_dir)