import json
import random
import requests
import time
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from transformers import pipeline
import streamlit as st
from bs4 import BeautifulSoup
import urllib.parse
from wiki_urls_scraping import generate_random_wiki_urls_scraping
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import uuid
import pickle
from pathlib import Path


# ---------------------------------------------------------------
# Step 1: Load Wikipedia URLs (200 Fixed + 300 randomly scraped)
# ---------------------------------------------------------------
def load_urls():

    # Load fixed URLs from 200_fixed_urls.json file
    with open(r'D:\Bits-MTech\Assignments\hybrid-rag-system\data\200_fixed_urls.json', "r") as f:
        fixed_urls = json.load(f)["fixed_wiki_urls"]
    
    # Randomly sample 300 URLs (replace with real Wikipedia scraping). For demo, we use fixed URLs only.
    #random_urls = generate_random_wiki_urls_scraping(count=300)
    return fixed_urls

# -----------------------------
# Step 2: Extract and Chunk Text
# -----------------------------
def chunk_text(text, chunk_size=300, overlap=50):
    """Return list of text chunks (strings) split by tokens (whitespace).
    Defaults: 300-token chunks with 50-token overlap to satisfy 200-400 range.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks


def chunk_text_with_metadata(text, url, title, chunk_size=300, overlap=50, start_id=0):
    """Split `text` into token chunks and return list of dicts with metadata.

    Each chunk dict: {'id': <uuid4>, 'url': url, 'title': title, 'text': chunk_text}
    """
    chunks = []
    token_chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    for part in token_chunks:
        chunk_id = str(uuid.uuid4())
        chunks.append({
            'id': chunk_id,
            'url': url,
            'title': title,
            'text': part
        })
    return chunks


def fetch_text_from_url(url, max_chars=20000):
    """Fetch a page and return (title, cleaned_text).

    On errors returns ("", "").
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; MSIE 10.0; Windows NT)"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        # Get title (Wikipedia uses h1#firstHeading)
        title_tag = soup.find('h1', id='firstHeading')
        if title_tag:
            title = title_tag.get_text(strip=True)
        else:
            title = soup.title.string if soup.title else ''

        content_div = soup.find('div', {'class': 'mw-parser-output'})
        if content_div:
            paragraphs = content_div.find_all('p')
        else:
            paragraphs = soup.find_all('p')
        text = " ".join([p.get_text(separator=" ", strip=True) for p in paragraphs])
        # Trim excessive whitespace and limit size
        text = " ".join(text.split())
        return title, text[:max_chars]
    except Exception:
        return "", ""

# -----------------------------
# Step 3: Dense Vector Index (FAISS)
# -----------------------------
def build_dense_index(chunks, model_name="all-MiniLM-L6-v2"):
    """Build FAISS index from list of chunk dicts. Uses the chunk['text'] field."""
    model = SentenceTransformer(model_name)
    texts = [c['text'] for c in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index, embeddings, model

def dense_retrieve(query, index, model, chunks, top_k=10):
    q_emb = model.encode([query], convert_to_numpy=True)
    scores, ids = index.search(q_emb, top_k)
    return [(chunks[i], float(scores[0][j])) for j, i in enumerate(ids[0])]

# -----------------------------
# Step 4: Sparse Retrieval (BM25)
# -----------------------------
def build_sparse_index(chunks):
    """Build BM25 index from list of chunk dicts."""
    texts = [c['text'] for c in chunks]
    tokenized = [text.split() for text in texts]
    bm25 = BM25Okapi(tokenized)
    return bm25, tokenized

def sparse_retrieve(query, bm25, chunks, top_k=10):
    scores = bm25.get_scores(query.split())
    ranked = np.argsort(scores)[::-1][:top_k]
    return [(chunks[i], float(scores[i])) for i in ranked]

# -----------------------------
# Step 5: Reciprocal Rank Fusion
# -----------------------------
def reciprocal_rank_fusion(dense_results, sparse_results, k=60, top_n=10):
    """Fuse dense and sparse ranked lists. Inputs are lists of (chunk_dict, score).

    Returns list of (chunk_dict, fused_score).
    """
    scores = {}
    id_to_chunk = {}
    # Assign ranks
    for rank, (chunk, _) in enumerate(dense_results):
        cid = chunk['id']
        id_to_chunk[cid] = chunk
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
    for rank, (chunk, _) in enumerate(sparse_results):
        cid = chunk['id']
        id_to_chunk[cid] = chunk
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
    # Sort by fused score
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [(id_to_chunk[cid], score) for cid, score in fused]

# -----------------------------
# Step 6: Response Generation
# -----------------------------
def generate_answer(query, context_chunks, model_name="google/flan-t5-base"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    # context_chunks is list of (chunk_dict, score)
    context = "\n".join([c['text'] for c, _ in context_chunks])
    prompt = f"Answer the question based on context:\n{context}\n\nQuestion: {query}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
    outputs = model.generate(**inputs, max_length=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)




# ---------------------------------------------------------------
# Corpus Persistence Functions
# ---------------------------------------------------------------
def save_corpus_and_vectors(corpus_dir: str, chunks: list, embeddings: np.ndarray, 
                           dense_index: faiss.IndexFlatIP, tokenized: list, 
                           model_name: str = "all-MiniLM-L6-v2"):
    """Save preprocessed corpus and vector database to disk.
    
    Creates the following files in corpus_dir:
    - corpus_chunks.json: Chunk metadata (id, url, title, text)
    - embeddings.npy: Dense embeddings (NumPy array)
    - dense_index.faiss: FAISS dense index file
    - tokenized_data.pkl: Tokenized texts for BM25
    - corpus_metadata.json: Metadata (model name, chunk count, etc.)
    """
    Path(corpus_dir).mkdir(parents=True, exist_ok=True)
    
    # Save chunks metadata
    chunks_path = Path(corpus_dir) / 'corpus_chunks.json'
    with open(chunks_path, 'w', encoding='utf-8') as fh:
        json.dump(chunks, fh, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved {len(chunks)} chunks to {chunks_path}")
    
    # Save embeddings
    embeddings_path = Path(corpus_dir) / 'embeddings.npy'
    np.save(embeddings_path, embeddings)
    print(f"[INFO] Saved embeddings ({embeddings.shape}) to {embeddings_path}")
    
    # Save FAISS index
    index_path = Path(corpus_dir) / 'dense_index.faiss'
    try:
        faiss.write_index(dense_index, str(index_path))
        print(f"[INFO] Saved FAISS index to {index_path}")
    except Exception as e:
        print(f"[WARNING] Failed to save FAISS index: {e}. Index can be rebuilt from embeddings.")
    
    # Save tokenized data for BM25 recreation
    tokenized_path = Path(corpus_dir) / 'tokenized_data.pkl'
    with open(tokenized_path, 'wb') as fh:
        pickle.dump(tokenized, fh)
    print(f"[INFO] Saved tokenized data ({len(tokenized)} tokens) to {tokenized_path}")
    
    # Save metadata
    metadata = {
        'model_name': model_name,
        'chunk_count': len(chunks),
        'embedding_dim': embeddings.shape[1],
        'timestamp': time.time()
    }
    metadata_path = Path(corpus_dir) / 'corpus_metadata.json'
    with open(metadata_path, 'w', encoding='utf-8') as fh:
        json.dump(metadata, fh, indent=2)
    print(f"[INFO] Saved corpus metadata to {metadata_path}")


def load_corpus_and_vectors(corpus_dir: str, model_name: str = "all-MiniLM-L6-v2"):
    """Load preprocessed corpus and vector database from disk.
    
    Returns: (chunks, embeddings, dense_index, tokenized) or None if files missing
    """
    corpus_path = Path(corpus_dir)
    chunks_path = corpus_path / 'corpus_chunks.json'
    embeddings_path = corpus_path / 'embeddings.npy'
    index_path = corpus_path / 'dense_index.faiss'
    tokenized_path = corpus_path / 'tokenized_data.pkl'
    
    # Check if core files exist
    if not chunks_path.exists() or not embeddings_path.exists():
        print(f"[WARNING] Corpus cache not found at {corpus_dir}")
        return None
    
    print(f"[INFO] Loading corpus from {corpus_dir}...")
    
    # Load chunks
    with open(chunks_path, 'r', encoding='utf-8') as fh:
        chunks = json.load(fh)
    print(f"[INFO] Loaded {len(chunks)} chunks")
    
    # Load embeddings
    embeddings = np.load(str(embeddings_path))
    print(f"[INFO] Loaded embeddings with shape {embeddings.shape}")
    
    # Load or rebuild FAISS index
    if index_path.exists():
        try:
            dense_index = faiss.read_index(str(index_path))
            print(f"[INFO] Loaded FAISS index")
        except Exception as e:
            print(f"[WARNING] Failed to load FAISS index: {e}. Rebuilding from embeddings...")
            dim = embeddings.shape[1]
            dense_index = faiss.IndexFlatIP(dim)
            dense_index.add(embeddings)
    else:
        print(f"[INFO] FAISS index not found. Rebuilding from embeddings...")
        dim = embeddings.shape[1]
        dense_index = faiss.IndexFlatIP(dim)
        dense_index.add(embeddings)
    
    # Load tokenized data or recreate
    if tokenized_path.exists():
        with open(tokenized_path, 'rb') as fh:
            tokenized = pickle.load(fh)
        print(f"[INFO] Loaded {len(tokenized)} tokenized texts")
    else:
        print(f"[INFO] Tokenized data not found. Recreating from chunks...")
        tokenized = [c['text'].split() for c in chunks]
    
    return chunks, embeddings, dense_index, tokenized


def load_or_build_corpus(corpus_dir: str, urls: list = None, chunk_size: int = 300, 
                        overlap: int = 50, force_rebuild: bool = False,
                        model_name: str = "all-MiniLM-L6-v2"):
    """Load corpus from cache or build from URLs if missing.
    
    First checks for saved corpus. If found and force_rebuild=False, loads it.
    Otherwise builds from URLs, saves to cache, and returns.
    
    Returns: (chunks, embeddings, dense_index, model, bm25, tokenized)
    """
    # Try to load from cache first
    if not force_rebuild:
        loaded = load_corpus_and_vectors(corpus_dir, model_name)
        if loaded:
            chunks, embeddings, dense_index, tokenized = loaded
            model = SentenceTransformer(model_name)
            bm25, _ = build_sparse_index(chunks)
            print(f"[INFO] Using cached corpus from {corpus_dir}")
            return chunks, embeddings, dense_index, model, bm25, tokenized
    
    # Build corpus from URLs
    if urls is None or len(urls) == 0:
        raise ValueError("No URLs provided and corpus cache not found. Please provide URLs to build corpus.")
    
    print(f"[INFO] Building corpus from {len(urls)} URLs...")
    all_chunks = []
    successful_urls = 0
    
    for idx, url in enumerate(urls):
        if idx % 10 == 0:
            print(f"[INFO] Processing URL {idx+1}/{len(urls)}...")
        title, text = fetch_text_from_url(url)
        if text:
            parts = chunk_text_with_metadata(text, url, title, chunk_size=chunk_size, overlap=overlap)
            all_chunks.extend(parts)
            successful_urls += 1
    
    print(f"[INFO] Built corpus with {len(all_chunks)} chunks from {successful_urls} URLs")
    
    # Build indices
    dense_index, embeddings, model = build_dense_index(all_chunks, model_name=model_name)
    bm25, tokenized = build_sparse_index(all_chunks)
    
    # Save to cache
    save_corpus_and_vectors(corpus_dir, all_chunks, embeddings, dense_index, tokenized, model_name)
    
    return all_chunks, embeddings, dense_index, model, bm25, tokenized