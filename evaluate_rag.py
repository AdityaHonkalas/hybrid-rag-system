import json
import os
import time
import uuid
from typing import List, Dict, Tuple, Optional
import requests
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from hybrid_rag_system import load_or_build_corpus, load_urls


# -----------------------------
# Utilities and Defaults
# -----------------------------
DEFAULT_MODEL = 'all-MiniLM-L6-v2'
CORPUS_CACHE_DIR = os.path.join(os.path.dirname(__file__), r'data\corpus_cache')
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50


def initialize_corpus(corpus_cache_dir: str = CORPUS_CACHE_DIR, 
                     model_name: str = DEFAULT_MODEL,
                     force_rebuild: bool = False) -> Tuple[List[Dict], SentenceTransformer]:
    """Initialize/load corpus for evaluation.
    
    Loads from cache if available, otherwise builds from URLs.
    Returns: (chunks, model) for use in evaluation
    """
    print(f"[INFO] Initializing corpus for evaluation...")
    print(f"[INFO] Cache directory: {corpus_cache_dir}")
    
    # Load URLs
    urls = load_urls()
    print(f"[INFO] Loaded {len(urls)} URLs")
    
    # Load or build corpus
    chunks, embeddings, dense_idx, model, bm25, tokenized = load_or_build_corpus(
        corpus_dir=corpus_cache_dir,
        urls=urls,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
        force_rebuild=force_rebuild,
        model_name=model_name
    )
    
    print(f"[INFO] Corpus initialized with {len(chunks)} chunks")
    return chunks, model


def load_dataset(path: str) -> List[Dict]:
    """Load dataset JSON. Expect format {"data": [ {question, answer, source_ids}, ... ]}
    Returns list of items.
    """
    with open(path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    return payload.get('data', [])


def source_match(retrieved_doc: Dict, true_sources: List[str]) -> bool:
    """Match dataset source_ids with retrieved chunk by title or URL (normalized)."""
    title = (retrieved_doc.get('title') or '').lower().replace(' ', '_')
    url = (retrieved_doc.get('url') or '').lower()

    for src in true_sources:
        src_norm = src.lower().replace(' ', '_')
        if src_norm in title or src_norm in url:
            return True
    return False


def compute_semantic_similarity(predicted_answer: str, reference_answer: str, model: SentenceTransformer) -> float:
    emb_pred = model.encode([predicted_answer])
    emb_ref = model.encode([reference_answer])
    score = cosine_similarity(emb_pred, emb_ref)[0][0]
    return float(score)


def token_overlap_ratio(predicted: str, reference: str) -> float:
    """Custom Metric 1: Token Overlap Ratio (TO)
    TO = (# shared tokens) / (# tokens in reference)
    Measures how much of the reference's content is present in the generated answer.
    """
    pred_tokens = set(predicted.lower().split())
    ref_tokens = reference.lower().split()
    if not ref_tokens:
        return 0.0
    shared = sum(1 for t in ref_tokens if t in pred_tokens)
    return shared / len(ref_tokens)


def source_coverage_score(retrieved_docs: List[Dict], true_sources: List[str], top_k: int) -> float:
    """Custom Metric 2: Source Coverage Score (SCS)
    SCS = (# unique true sources present in top_k retrieved docs) / (# unique true sources)
    Measures whether the retrieval stage covers the ground-truth sources.
    """
    if not true_sources:
        return 0.0
    found = set()
    for doc in retrieved_docs[:top_k]:
        title = (doc.get('title') or '').lower().replace(' ', '_')
        url = (doc.get('url') or '').lower()
        for src in true_sources:
            src_norm = src.lower().replace(' ', '_')
            if src_norm in title or src_norm in url:
                found.add(src_norm)
    return len(found) / len(set([s.lower().replace(' ', '_') for s in true_sources]))


def evaluate_dataset(
    dataset: List[Dict],
    api_url: str = None,
    init_url: Optional[str] = None,
    top_k: int = 5,
    model_name: str = DEFAULT_MODEL,
    progress: bool = True,
    query_fn: Optional[callable] = None
) -> Tuple[Dict, List[Dict]]:
    """Evaluate dataset against RAG API.
    Returns summary dict and list of per-question results.
    """
    # Optionally initialize remote system --> No need to call initialize API as the corpus has been checked already before evaluation. 
    # if init_url:
    #     try:
    #         requests.post(init_url, timeout=30)
    #     except Exception:
    #         pass

    total_questions = len(dataset)

    # Accumulators
    mrr_total = 0.0
    precision_total = 0.0
    hit_total = 0
    dense_mrr_total = 0.0
    sparse_mrr_total = 0.0
    source_coverage_total = 0.0
    latency_total = 0.0

    results_log = []

    iterator = tqdm(dataset, desc='Evaluating RAG System') if progress else dataset
    for item in iterator:
        question_id = item.get('id') or str(uuid.uuid4())
        question = item.get('question', '')
        true_answer = (item.get('answer') or '').strip()
        true_sources = item.get('source_ids', []) or []

        start = time.time()
        try:
            if query_fn is not None:
                # call provided local query function; it should return a dict like the API response
                result = query_fn(question)
            else:
                resp = requests.post(api_url, json={'query': question}, timeout=30)
                result = resp.json()
        except Exception:
            # On API failure, record a placeholder
            result = {'answer': '', 'retrieved_chunks': [], 'response_time_seconds': 0}
        end = time.time()

        latency = result.get('response_time_seconds', end - start)
        latency_total += latency

        generated_answer = (result.get('answer') or '').strip()
        retrieved_docs = result.get('retrieved_chunks', []) or []

        # MRR
        rank = None
        for idx, doc in enumerate(retrieved_docs, start=1):
            if source_match(doc, true_sources):
                rank = idx
                break
        mrr_val = (1 / rank) if rank else 0.0
        mrr_total += mrr_val

        # Precision@K
        top_k_docs = retrieved_docs[:top_k]
        relevant_count = sum(source_match(doc, true_sources) for doc in top_k_docs)
        precision_total += relevant_count / top_k

        # Hit@K
        hit_total += 1 if relevant_count > 0 else 0

        # Dense/Sparse MRR (if available in response)
        dense_retrieved_docs = result.get('dense_retrieved_chunks', []) or []
        sparse_retrieved_docs = result.get('sparse_retrieved_chunks', []) or []

        dense_rank = None
        for idx, doc in enumerate(dense_retrieved_docs, start=1):
            if source_match(doc, true_sources):
                dense_rank = idx
                break
        dense_mrr = (1 / dense_rank) if dense_rank else 0.0
        dense_mrr_total += dense_mrr

        sparse_rank = None
        for idx, doc in enumerate(sparse_retrieved_docs, start=1):
            if source_match(doc, true_sources):
                sparse_rank = idx
                break
        sparse_mrr = (1 / sparse_rank) if sparse_rank else 0.0
        sparse_mrr_total += sparse_mrr

        scs = source_coverage_score(retrieved_docs, true_sources, top_k)
        source_coverage_total += scs

        results_log.append({
            'question_id': question_id,
            'question': question,
            'expected_answer': true_answer,
            'generated_answer': generated_answer,
            'correct_source': true_sources,
            'rank': rank,
            'mrr': mrr_val,
            'hit': 1 if relevant_count > 0 else 0,
            'precision_at_k': relevant_count / top_k,
            'dense_mrr': dense_mrr,
            'sparse_mrr': sparse_mrr,
            'source_coverage_score': scs,
            'latency': latency
        })

    # Aggregate
    summary = {
        'MRR': mrr_total / total_questions if total_questions else 0.0,
        f'Precision@{top_k}': precision_total / total_questions if total_questions else 0.0,
        f'Hit@{top_k}': hit_total / total_questions if total_questions else 0.0,
        'Avg_Dense_MRR': dense_mrr_total / total_questions if total_questions else 0.0,
        'Avg_Sparse_MRR': sparse_mrr_total / total_questions if total_questions else 0.0,
        'Avg_Source_Coverage_Score': source_coverage_total / total_questions if total_questions else 0.0,
        'Avg_Latency': latency_total / total_questions if total_questions else 0.0,
        'total_questions': total_questions
    }

    return summary, results_log


def generate_reports(summary: Dict, detailed: List[Dict], out_dir: str, top_k: int = 5, ablation_results: Optional[List[Dict]] = None) -> Dict:
    """Generate CSV, JSON, HTML and PDF reports with visualizations.
    Returns dict of generated file paths.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    # Save JSON
    json_path = Path(out_dir) / 'evaluation_results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({'summary': summary, 'detailed': detailed}, f, indent=2)

    # Save CSV
    df = pd.DataFrame(detailed)
    csv_path = Path(out_dir) / 'evaluation_results.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8')

    # Visualizations
    images = {}
    # Metric comparisons (bar)
    metrics_for_plot = {
        'MRR': summary.get('MRR', 0),
        f'Precision@{top_k}': summary.get(f'Precision@{top_k}', 0),
        f'Hit@{top_k}': summary.get(f'Hit@{top_k}', 0),
        'Avg_Dense_MRR': summary.get('Avg_Dense_MRR', 0),
        'Avg_Sparse_MRR': summary.get('Avg_Sparse_MRR', 0)
    }
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(metrics_for_plot.keys()), y=list(metrics_for_plot.values()))
    plt.title('Overall Metric Comparison')
    plt.xticks(rotation=15)
    m_comp_path = Path(out_dir) / 'metric_comparison.png'
    plt.tight_layout()
    plt.savefig(m_comp_path)
    images['metric_comparison'] = str(m_comp_path)
    plt.close()

    # Distribution plots
    plt.figure(figsize=(8, 5))
    sns.histplot(df['dense_mrr'].dropna(), kde=True, bins=25)
    plt.title('Dense MRR Distribution')
    sim_path = Path(out_dir) / 'dense_mrr_dist.png'
    plt.tight_layout()
    plt.savefig(sim_path)
    images['dense_mrr_dist'] = str(sim_path)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.histplot(df['latency'].dropna(), kde=True, bins=25)
    plt.title('Response Time Distribution (s)')
    lat_path = Path(out_dir) / 'latency_dist.png'
    plt.tight_layout()
    plt.savefig(lat_path)
    images['latency_dist'] = str(lat_path)
    plt.close()

    # Retrieval heatmap: correlation between metrics
    corr = df[['precision_at_k', 'dense_mrr', 'sparse_mrr', 'source_coverage_score', 'latency']].corr()
    plt.figure(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Metric Correlation Heatmap')
    heat_path = Path(out_dir) / 'retrieval_heatmap.png'
    plt.tight_layout()
    plt.savefig(heat_path)
    images['retrieval_heatmap'] = str(heat_path)
    plt.close()

    # Error analysis: top failures (zero hit or low semantic similarity)
    failures = df[(df['hit'] == 0) | ((df['dense_mrr'] == 0) & (df['sparse_mrr'] == 0))].sort_values(by='dense_mrr')
    failures_path = Path(out_dir) / 'failures.csv'
    failures.to_csv(failures_path, index=False, encoding='utf-8')

    # Failure pattern analysis (simple heuristics)
    pattern_counts = {
        'zero_hit': int(((df['hit'] == 0)).sum()),
        'zero_dense_and_sparse_mrr': int(((df['dense_mrr'] == 0) & (df['sparse_mrr'] == 0)).sum()),
        'low_source_coverage': int((df['source_coverage_score'] < 0.5).sum())
    }

    # Generate simple HTML summary
    html_path = Path(out_dir) / 'evaluation_report.html'
    with open(html_path, 'w', encoding='utf-8') as fh:
        fh.write('<html><head><meta charset="utf-8"><title>RAG Evaluation Report</title></head><body>')
        fh.write(f'<h1>RAG Evaluation Report</h1>')
        fh.write('<h2>Overall Summary</h2>')
        fh.write('<ul>')
        for k, v in summary.items():
            fh.write(f'<li><strong>{k}:</strong> {v}</li>')
        fh.write('</ul>')

        fh.write('<h2>Visualizations</h2>')
        for key, img in images.items():
            fh.write(f'<h3>{key.replace("_", " ").title()}</h3>')
            fh.write(f'<img src="{Path(img).name}" style="max-width:800px;">')

        fh.write('<h2>Failure Examples</h2>')
        fh.write('<p>Simple failure pattern counts:</p>')
        fh.write('<ul>')
        for k, v in pattern_counts.items():
            fh.write(f'<li><strong>{k}:</strong> {v}</li>')
        fh.write('</ul>')

        fh.write('<table border="1" cellpadding="5"><tr><th>Question ID</th><th>Question</th><th>Expected</th><th>Generated</th><th>MRR</th><th>Dense MRR</th><th>Sparse MRR</th><th>Source Coverage</th><th>Latency</th></tr>')
        for _, row in failures.head(20).iterrows():
            fh.write('<tr>')
            fh.write(f'<td>{row.get("question_id")}</td>')
            fh.write(f'<td>{row.get("question")}</td>')
            fh.write(f'<td>{row.get("expected_answer")}</td>')
            fh.write(f'<td>{row.get("generated_answer")}</td>')
            fh.write(f'<td>{row.get("mrr"):.3f}</td>')
            fh.write(f'<td>{row.get("dense_mrr"):.3f}</td>')
            fh.write(f'<td>{row.get("sparse_mrr"):.3f}</td>')
            fh.write(f'<td>{row.get("source_coverage_score"):.3f}</td>')
            fh.write(f'<td>{row.get("latency"):.2f}</td>')
            fh.write('</tr>')
        fh.write('</table>')

        # Detailed results table (all questions) --- > Not needed in case of detailed reports generated
        '''fh.write('<h2>Detailed Results Table</h2>')
        fh.write('<table border="1" cellpadding="5"><tr><th>Question ID</th><th>Question</th><th>Ground Truth</th><th>Generated Answer</th><th>MRR</th><th>Token Overlap</th><th>Source Coverage</th><th>Time (s)</th></tr>')
        for _, row in df.iterrows():
            fh.write('<tr>')
            fh.write(f'<td>{row.get("question_id")}</td>')
            fh.write(f'<td>{row.get("question")}</td>')
            fh.write(f'<td>{row.get("expected_answer")}</td>')
            fh.write(f'<td>{row.get("generated_answer")}</td>')
            fh.write(f'<td>{row.get("mrr"):.3f}</td>')
            fh.write(f'<td>{row.get("token_overlap_ratio"):.3f}</td>')
            fh.write(f'<td>{row.get("source_coverage_score"):.3f}</td>')
            fh.write(f'<td>{row.get("latency"):.2f}</td>')
            fh.write('</tr>')
        fh.write('</table>')'''

        # Detailed justification for custom metrics for Precision and Hit
        fh.write('<h2>Custom Metrics: Justification & Methodology</h2>')
        fh.write('<h3>1. Precision@K</h3>')
        fh.write('<ul><strong>Why chosen:</strong> This directly measures the relevance density within the limited top‑K chunks that the generator actually uses. In a hybrid RAG setup, retrieval noise dilutes context; Precision@K quantifies that dilution and helps tune retrieval to maximize relevant evidence in the prompt.</ul>')
        fh.write('<ul><strong>Calculation:</strong> For each question, count how many of the top K retrieved chunks match any ground‑truth source (source_match). Then divide by K.</ul>')
        fh.write('<ul><strong>Formula:</strong> Precision@K = (# relevant retrieved in top K) / K </ul>')

        fh.write('<h3>2. Hit@K</h3>')
        fh.write('<ul><strong>Why chosen:</strong>  This captures the minimum viable retrieval success: whether at least one relevant chunk was found in the top‑K. It’s a strong gating signal for generation quality—if Hit@K is 0, the answer is likely ungrounded regardless of rank distribution. </ul>')
        fh.write('<ul><strong>Calculation:</strong> If there is at least one relevant chunk in the top K, the score is 1, otherwise 0. Aggregate by averaging over questions. </ul>')
        fh.write('<ul><strong>Interpretation:</strong> Hit@K = 1 if (# relevant retrieved in top K) > 0 else 0 </ul>')

        fh.write('</body></html>')

    # Copy images to out_dir (they are already in out_dir) so HTML references work

    # Create PDF - assemble key images and summary
    pdf_path = Path(out_dir) / 'evaluation_report.pdf'
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    width, height = letter
    c.setFont('Helvetica-Bold', 16)
    c.drawString(50, height - 50, 'RAG Evaluation Report')
    c.setFont('Helvetica', 10)
    y = height - 80
    for k, v in summary.items():
        c.drawString(50, y, f'{k}: {v}')
        y -= 14
        if y < 120:
            c.showPage()
            y = height - 50

    # Add metric comparison image
    for img_key in ['metric_comparison', 'dense_mrr_dist', 'latency_dist', 'retrieval_heatmap']:
        img_file = images.get(img_key)
        if img_file and Path(img_file).exists():
            try:
                c.showPage()
                c.drawImage(str(img_file), 40, 150, width=520, preserveAspectRatio=True)
            except Exception:
                pass

    c.save()

    # Optional: ablation plot if ablation_results provided
    if ablation_results:
        try:
            labels = [a.get('label') for a in ablation_results]
            mrrs = [a.get('MRR', 0) for a in ablation_results]
            plt.figure(figsize=(8, 4))
            sns.barplot(x=labels, y=mrrs)
            plt.title('Ablation Study: MRR by Setting')
            plt.xticks(rotation=30)
            ablation_path = Path(out_dir) / 'ablation_mrr.png'
            plt.tight_layout()
            plt.savefig(ablation_path)
            images['ablation_mrr'] = str(ablation_path)
            plt.close()
        except Exception:
            pass
    return {
        'json': str(json_path),
        'csv': str(csv_path),
        'html': str(html_path),
        'pdf': str(pdf_path),
        'images': images,
        'failures_csv': str(failures_path)
    }


if __name__ == '__main__':
    # CLI convenience for running locally
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate RAG system on dataset')
    parser.add_argument('--dataset', type=str, default='data/wikipedia_qa_100.json')
    parser.add_argument('--api', type=str, default='http://127.0.0.1:5000/api/query')
    parser.add_argument('--init', type=str, default='http://127.0.0.1:5000/api/initialize')
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--out', type=str, default='evaluations/last')
    parser.add_argument('--force-rebuild-corpus', action='store_true', 
                       help='Force rebuild corpus cache from URLs')
    args = parser.parse_args()

    # Initialize corpus (loads from cache or builds from URLs)
    print(f"\n[EVAL] Initializing corpus (cache: {CORPUS_CACHE_DIR})...")
    try:
        chunks, model = initialize_corpus(
            corpus_cache_dir=CORPUS_CACHE_DIR,
            model_name=DEFAULT_MODEL,
            force_rebuild=args.force_rebuild_corpus
        )
        print(f"[EVAL] ✓ Corpus ready with {len(chunks)} chunks\n")
    except Exception as e:
        print(f"[EVAL] ✗ Failed to initialize corpus: {e}")
        import sys
        sys.exit(1)

    # Run evaluation
    data = load_dataset(args.dataset)
    summary, detailed = evaluate_dataset(data, api_url=args.api, init_url=args.init, top_k=args.topk)
    paths = generate_reports(summary, detailed, args.out, top_k=args.topk)
    print(f"\n[EVAL] Reports generated in {args.out}")
