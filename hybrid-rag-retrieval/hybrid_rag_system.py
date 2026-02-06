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


# ---------------------------------------------------------------
# Step 1: Load Wikipedia URLs (200 Fixed + 300 randomly scraped)
# ---------------------------------------------------------------
def load_urls():

    # Load fixed URLs from 200_fixed_urls.json file
    with open(r'D:\Bits-MTech\Assignments\hybrid-rag-system\hybrid-rag-retrieval\data\200_fixed_urls.json', "r") as f:
        fixed_urls = json.load(f)["fixed_wiki_urls"][:50]
    
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

# # -----------------------------
# # Step 7: Streamlit UI
# # -----------------------------
# def run_ui(chunks, dense_index, dense_model, bm25):
#     st.title("Hybrid RAG System (Dense + BM25 + RRF)")
#     query = st.text_input("Enter your question:")
#     if query:
#         start = time.time()
#         dense_results = dense_retrieve(query, dense_index, dense_model, chunks)
#         sparse_results = sparse_retrieve(query, bm25, chunks)
#         rrf_results = reciprocal_rank_fusion(dense_results, sparse_results)
#         answer = generate_answer(query, rrf_results)
#         end = time.time()

#         st.subheader("Generated Answer")
#         st.write(answer)
#         st.subheader("Top Retrieved Chunks")
#         for chunk, score in rrf_results:
#             title = chunk.get('title', '')
#             url = chunk.get('url', '')
#             text_snip = chunk.get('text', '')[:200]
#             st.write(f"RRF Score: {score:.4f} | Title: {title} | URL: {url}")
#             st.write(text_snip + "...")
#         st.write(f"Response Time: {end-start:.2f} seconds")

# -----------------------------
# Main Execution
# -----------------------------
# if __name__ == "__main__":
#     urls = load_urls()
#     # Build corpus with real text and metadata
#     all_chunks = []
#     for url in urls:
#         title, text = fetch_text_from_url(url)
#         if not text:
#             continue
#         parts = chunk_text_with_metadata(text, url, title, chunk_size=300, overlap=50)
#         all_chunks.extend(parts)

#     # Build indices using chunk dicts
#     dense_index, embeddings, dense_model = build_dense_index(all_chunks)
#     bm25, tokenized = build_sparse_index(all_chunks)

#     run_ui(all_chunks, dense_index, dense_model, bm25)