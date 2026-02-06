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


# -----------------------------
# Utilities and Defaults
# -----------------------------
DEFAULT_MODEL = 'all-MiniLM-L6-v2'


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
    api_url: str,
    init_url: Optional[str] = None,
    top_k: int = 5,
    model_name: str = DEFAULT_MODEL,
    progress: bool = True
) -> Tuple[Dict, List[Dict]]:
    """Evaluate dataset against RAG API.
    Returns summary dict and list of per-question results.
    """
    # Optionally initialize remote system
    if init_url:
        try:
            requests.post(init_url, timeout=30)
        except Exception:
            pass

    model = SentenceTransformer(model_name)

    total_questions = len(dataset)

    # Accumulators
    mrr_total = 0.0
    precision_total = 0.0
    hit_total = 0
    semantic_similarity_total = 0.0
    token_overlap_total = 0.0
    source_coverage_total = 0.0
    latency_total = 0.0

    results_log = []

    iterator = tqdm(dataset, desc='Evaluating RAG System') if progress else dataset
    for item in iterator:
        question = item.get('question', '')
        true_answer = (item.get('answer') or '').strip()
        true_sources = item.get('source_ids', []) or []

        start = time.time()
        try:
            resp = requests.post(api_url, json={'query': question}, timeout=30)
            result = resp.json()
        except Exception as e:
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
        mrr_total += (1 / rank) if rank else 0.0

        # Precision@K
        top_k_docs = retrieved_docs[:top_k]
        relevant_count = sum(source_match(doc, true_sources) for doc in top_k_docs)
        precision_total += relevant_count / top_k

        # Hit@K
        hit_total += 1 if relevant_count > 0 else 0

        # Semantic similarity
        sem_sim = compute_semantic_similarity(generated_answer, true_answer, model)
        semantic_similarity_total += sem_sim

        # Custom metrics
        to_ratio = token_overlap_ratio(generated_answer, true_answer)
        token_overlap_total += to_ratio

        scs = source_coverage_score(retrieved_docs, true_sources, top_k)
        source_coverage_total += scs

        results_log.append({
            'question': question,
            'expected_answer': true_answer,
            'generated_answer': generated_answer,
            'correct_source': true_sources,
            'rank': rank,
            'hit': 1 if relevant_count > 0 else 0,
            'precision_at_k': relevant_count / top_k,
            'semantic_similarity': sem_sim,
            'token_overlap_ratio': to_ratio,
            'source_coverage_score': scs,
            'latency': latency
        })

    # Aggregate
    summary = {
        'MRR': mrr_total / total_questions if total_questions else 0.0,
        f'Precision@{top_k}': precision_total / total_questions if total_questions else 0.0,
        f'Hit@{top_k}': hit_total / total_questions if total_questions else 0.0,
        'Avg_Semantic_Similarity': semantic_similarity_total / total_questions if total_questions else 0.0,
        'Avg_Token_Overlap_Ratio': token_overlap_total / total_questions if total_questions else 0.0,
        'Avg_Source_Coverage_Score': source_coverage_total / total_questions if total_questions else 0.0,
        'Avg_Latency': latency_total / total_questions if total_questions else 0.0,
        'total_questions': total_questions
    }

    return summary, results_log


def generate_reports(summary: Dict, detailed: List[Dict], out_dir: str, top_k: int = 5) -> Dict:
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
        'Avg_Semantic_Similarity': summary.get('Avg_Semantic_Similarity', 0)
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
    sns.histplot(df['semantic_similarity'].dropna(), kde=True, bins=25)
    plt.title('Semantic Similarity Distribution')
    sim_path = Path(out_dir) / 'semantic_similarity_dist.png'
    plt.tight_layout()
    plt.savefig(sim_path)
    images['semantic_similarity_dist'] = str(sim_path)
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
    corr = df[['precision_at_k', 'semantic_similarity', 'token_overlap_ratio', 'source_coverage_score', 'latency']].corr()
    plt.figure(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Metric Correlation Heatmap')
    heat_path = Path(out_dir) / 'retrieval_heatmap.png'
    plt.tight_layout()
    plt.savefig(heat_path)
    images['retrieval_heatmap'] = str(heat_path)
    plt.close()

    # Error analysis: top failures (zero hit or low semantic similarity)
    failures = df[(df['hit'] == 0) | (df['semantic_similarity'] < 0.3)].sort_values(by='semantic_similarity')
    failures_path = Path(out_dir) / 'failures.csv'
    failures.to_csv(failures_path, index=False, encoding='utf-8')

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
        fh.write('<table border="1" cellpadding="5"><tr><th>Question</th><th>Expected</th><th>Generated</th><th>Semantic Similarity</th><th>Latency</th></tr>')
        for _, row in failures.head(20).iterrows():
            fh.write('<tr>')
            fh.write(f'<td>{row.get("question")}</td>')
            fh.write(f'<td>{row.get("expected_answer")}</td>')
            fh.write(f'<td>{row.get("generated_answer")}</td>')
            fh.write(f'<td>{row.get("semantic_similarity"):.3f}</td>')
            fh.write(f'<td>{row.get("latency"):.2f}</td>')
            fh.write('</tr>')
        fh.write('</table>')

        # Detailed justification for custom metrics
        fh.write('<h2>Custom Metrics: Justification & Methodology</h2>')
        fh.write('<h3>Token Overlap Ratio (TO)</h3>')
        fh.write('<p><strong>Why chosen:</strong> Simple proxy for content overlap between generated and reference answers; useful when exact phrasing differs but content words overlap.</p>')
        fh.write('<p><strong>Calculation:</strong> TO = (# shared tokens) / (# tokens in reference). Tokens lowercased and split on whitespace.</p>')
        fh.write('<p><strong>Interpretation:</strong> Values near 1.0 indicate high overlap; near 0 indicate little overlap. Use with semantic similarity for robust interpretation.</p>')

        fh.write('<h3>Source Coverage Score (SCS)</h3>')
        fh.write('<p><strong>Why chosen:</strong> Measures whether retrieval fetches the ground-truth sources, indicating retrieval coverage.</p>')
        fh.write('<p><strong>Calculation:</strong> SCS = (# unique ground-truth sources found in top-K retrieved docs) / (# unique ground-truth sources).</p>')
        fh.write('<p><strong>Interpretation:</strong> Values near 1 indicate retrieval covered ground-truth sources; low values indicate missing evidence even if generation looks good.</p>')

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
    for img_key in ['metric_comparison', 'semantic_similarity_dist', 'latency_dist', 'retrieval_heatmap']:
        img_file = images.get(img_key)
        if img_file and Path(img_file).exists():
            try:
                c.showPage()
                c.drawImage(str(img_file), 40, 150, width=520, preserveAspectRatio=True)
            except Exception:
                pass

    c.save()

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
    args = parser.parse_args()

    data = load_dataset(args.dataset)
    summary, detailed = evaluate_dataset(data, api_url=args.api, init_url=args.init, top_k=args.topk)
    paths = generate_reports(summary, detailed, args.out, top_k=args.topk)
    print('Reports generated in', args.out)