import json
import time
import re
from flask import Flask, render_template, request, jsonify, Response
import requests
import uuid
from threading import Lock
import os
import evaluate_rag
from hybrid_rag_system import *


# ---------------------------------------------------------------
# Initialize Flask App
# ---------------------------------------------------------------
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Global variables for indices and models
chunks = []
dense_index = None
dense_model = None
bm25 = None
embeddings = None
tokenized = None
#tokenizer = None
#gen_model = None
data_lock = Lock()  # Thread-safe access to shared resources

# Corpus cache configuration
CORPUS_CACHE_DIR = os.path.join(os.path.dirname(__file__), r'data\corpus_cache')
MODEL_NAME = 'all-MiniLM-L6-v2'
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

print("[INFO] Flask app initialized. Indices not yet loaded.")
print(f"[INFO] Corpus cache directory: {CORPUS_CACHE_DIR}")

# Evaluation job tracking
evaluation_jobs = {}
EVAL_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'evaluations')
os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------
# Initialization Route
# ---------------------------------------------------------------
@app.route('/api/initialize', methods=['POST'])
def initialize_system():
    """Initialize/load the RAG system. Loads from cache if available, otherwise builds from URLs."""
    global chunks, dense_index, dense_model, bm25, embeddings, tokenized
    
    data = request.get_json() or {}
    force_rebuild = bool(data.get('force_rebuild', False))
    
    with data_lock:
        try:
            print("[INFO] Starting system initialization...")
            print(f"[INFO] Force rebuild: {force_rebuild}")
            
            # Load URLs
            print("[INFO] Loading URLs...")
            urls = load_urls()
            print(f"[INFO] Loaded {len(urls)} URLs.")
            
            # Load or build corpus (with caching)
            print(f"[INFO] Checking corpus cache at {CORPUS_CACHE_DIR}...")
            all_chunks, emb, dense_idx, model, bm25_idx, tokenized_texts = load_or_build_corpus(
                corpus_dir=CORPUS_CACHE_DIR,
                urls=urls,
                chunk_size=CHUNK_SIZE,
                overlap=CHUNK_OVERLAP,
                force_rebuild=force_rebuild,
                model_name=MODEL_NAME
            )
            
            print(f"[INFO] System initialized with {len(all_chunks)} chunks.")
            
            # Update globals
            chunks = all_chunks
            dense_index = dense_idx
            dense_model = model
            bm25 = bm25_idx
            tokenized = tokenized_texts
            embeddings = emb
            
            return jsonify({
                'status': 'success',
                'message': f'System initialized with {len(chunks)} chunks',
                'cache_dir': CORPUS_CACHE_DIR,
                'force_rebuild': force_rebuild,
                'embedding_shape': str(embeddings.shape) if embeddings is not None else 'None'
            })
        except Exception as e:
            print(f"[ERROR] Initialization failed: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/rebuild-cache', methods=['POST'])
def rebuild_cache():
    """Force rebuild and save corpus cache."""
    request_data = request.get_json() or {}
    request_data['force_rebuild'] = True
    request.json = request_data
    return initialize_system()


@app.route('/api/system/status', methods=['GET'])
def system_status():
    """Check if corpus is initialized and ready for queries."""
    is_ready = bool(chunks and dense_index and dense_model and bm25)
    return jsonify({
        'status': 'ready' if is_ready else 'not_initialized',
        'system_initialized': is_ready,
        'chunks_loaded': len(chunks),
        'has_embeddings': embeddings is not None,
        'has_dense_index': dense_index is not None,
        'has_bm25': bm25 is not None,
        'has_model': dense_model is not None
    })


# ---------------------------------------------------------------
# Query Endpoint
# ---------------------------------------------------------------
@app.route('/api/query', methods=['POST'])
def query_rag():
    """Accept a query and return generated answer + retrieved chunks."""
    global chunks, dense_index, dense_model, bm25
    
    if not chunks:
        return jsonify({'status': 'error', 'message': 'System not initialized. Call /api/initialize first.'}), 400
    
    try:
        data = request.get_json()
        query_text = data.get('query', '').strip()
        
        if not query_text:
            return jsonify({'status': 'error', 'message': 'Query cannot be empty'}), 400
        
        top_k = data.get('top_k', 10)
        
        start_time = time.time()
        
        with data_lock:
            # Retrieve
            dense_results = dense_retrieve(query_text, dense_index, dense_model, chunks, top_k=top_k)
            sparse_results = sparse_retrieve(query_text, bm25, chunks, top_k=top_k)
            rrf_results = reciprocal_rank_fusion(dense_results, sparse_results, top_n=5)
            
            # Generate
            answer = generate_answer(query_text, rrf_results)
        
        elapsed = time.time() - start_time
        
        # Format results
        retrieved_chunks = []
        for chunk, score in rrf_results:
            retrieved_chunks.append({
                'id': chunk['id'],
                'title': chunk['title'],
                'url': chunk['url'],
                'text': chunk['text'],
                'score': round(score, 4)
            })
        
        return jsonify({
            'status': 'success',
            'query': query_text,
            'answer': answer,
            'retrieved_chunks': retrieved_chunks,
            'response_time_seconds': round(elapsed, 2)
        })
    except Exception as e:
        print(f"[ERROR] Query failed: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ---------------------------------------------------------------
# Evaluation API
# ---------------------------------------------------------------
def _run_evaluation_job(job_id: str, dataset_path: str, top_k: int, api_url: str, init_url: str):
    """Background job runner for evaluation."""
    global chunks, dense_index, dense_model, bm25, embeddings, tokenized
    
    try:
        evaluation_jobs[job_id]['status'] = 'running'
        evaluation_jobs[job_id]['started_at'] = time.time()

        # Ensure corpus is loaded before evaluation
        print(f"[EVAL-{job_id[:8]}] Checking corpus status...")
        if not chunks or not dense_index or not dense_model or not bm25:
            print(f"[EVAL-{job_id[:8]}] Corpus not initialized. Loading now...")
            with data_lock:
                urls = load_urls()
                all_chunks, emb, dense_idx, model, bm25_idx, tokenized_texts = load_or_build_corpus(
                    corpus_dir=CORPUS_CACHE_DIR,
                    urls=urls,
                    chunk_size=CHUNK_SIZE,
                    overlap=CHUNK_OVERLAP,
                    force_rebuild=False,
                    model_name=MODEL_NAME
                )
                chunks = all_chunks
                dense_index = dense_idx
                dense_model = model
                bm25 = bm25_idx
                tokenized = tokenized_texts
                embeddings = emb
            print(f"[EVAL-{job_id[:8]}] Corpus ready with {len(chunks)} chunks")
        else:
            print(f"[EVAL-{job_id[:8]}] Corpus already initialized")

        data = evaluate_rag.load_dataset(dataset_path)
        # Provide a local query function to avoid HTTP calls to the same Flask server
        def local_query_fn(question_text: str):
            start_q = time.time()
            try:
                with data_lock:
                    dense_results = dense_retrieve(question_text, dense_index, dense_model, chunks, top_k=top_k)
                    sparse_results = sparse_retrieve(question_text, bm25, chunks, top_k=top_k)
                    rrf_results = reciprocal_rank_fusion(dense_results, sparse_results, top_n=top_k)
                    answer = generate_answer(question_text, rrf_results)

                elapsed_q = time.time() - start_q

                retrieved_chunks = []
                for chunk, score in rrf_results:
                    retrieved_chunks.append({
                        'id': chunk.get('id'),
                        'title': chunk.get('title'),
                        'url': chunk.get('url'),
                        'text': chunk.get('text'),
                        'score': round(score, 4)
                    })

                return {'answer': answer, 'retrieved_chunks': retrieved_chunks, 'response_time_seconds': elapsed_q}
            except Exception as e:
                return {'answer': '', 'retrieved_chunks': [], 'response_time_seconds': 0}

        summary, detailed = evaluate_rag.evaluate_dataset(data, api_url=api_url, init_url=init_url, top_k=top_k, progress=False, query_fn=local_query_fn)

        out_dir = os.path.join(EVAL_OUTPUT_DIR, job_id)
        paths = evaluate_rag.generate_reports(summary, detailed, out_dir, top_k=top_k)

        evaluation_jobs[job_id]['status'] = 'finished'
        evaluation_jobs[job_id]['finished_at'] = time.time()
        evaluation_jobs[job_id]['summary'] = summary
        evaluation_jobs[job_id]['paths'] = paths
    except Exception as e:
        evaluation_jobs[job_id]['status'] = 'error'
        evaluation_jobs[job_id]['error'] = str(e)
        evaluation_jobs[job_id]['finished_at'] = time.time()


@app.route('/api/evaluate', methods=['POST'])
def start_evaluation():
    """Start an evaluation job. Accepts JSON: {dataset_path, top_k, api_url, init_url}
    Runs synchronously (blocking).
    """
    data = request.get_json() or {}
    dataset_path = data.get('dataset_path', r'D:\Bits-MTech\Assignments\hybrid-rag-system\data\wikipedia_qa_100.json')
    top_k = int(data.get('top_k', 5))
    api_url = data.get('api_url', 'http://127.0.0.1:5000/api/query')
    init_url = data.get('init_url', 'http://127.0.0.1:5000/api/initialize')

    job_id = str(uuid.uuid4())
    evaluation_jobs[job_id] = {
        'status': 'pending',
        'dataset_path': dataset_path,
        'top_k': top_k,
        'api_url': api_url,
        'init_url': init_url,
        'created_at': time.time()
    }

    # Run evaluation job synchronously (blocking)
    _run_evaluation_job(job_id, dataset_path, top_k, api_url, init_url)

    return jsonify({'status': 'started', 'job_id': job_id})


@app.route('/api/evaluate/status/<job_id>', methods=['GET'])
def evaluation_status(job_id):
    job = evaluation_jobs.get(job_id)
    if not job:
        return jsonify({'status': 'error', 'message': 'job_id not found'}), 404
    return jsonify(job)


@app.route('/api/evaluate/report/<job_id>', methods=['GET'])
def evaluation_report(job_id):
    job = evaluation_jobs.get(job_id)
    if not job:
        # Try to recover from filesystem if job not present in-memory
        out_dir = os.path.join(EVAL_OUTPUT_DIR, job_id)
        if os.path.isdir(out_dir):
            # attempt to load JSON summary if present
            json_path = os.path.join(out_dir, 'evaluation_results.json')
            paths = {
                'json': os.path.join(out_dir, 'evaluation_results.json'),
                'csv': os.path.join(out_dir, 'evaluation_results.csv'),
                'html': os.path.join(out_dir, 'evaluation_report.html'),
                'pdf': os.path.join(out_dir, 'evaluation_report.pdf')
            }
            summary = None
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r', encoding='utf-8') as fh:
                        payload = json.load(fh)
                    summary = payload.get('summary')
                except Exception:
                    summary = None

            return jsonify({'status': 'success', 'paths': paths, 'summary': summary})
        return jsonify({'status': 'error', 'message': 'job_id not found'}), 404

    if job.get('status') != 'finished':
        return jsonify({'status': job.get('status'), 'message': 'report not ready'}), 400

    # If finished, ensure paths are available; if not, attempt to resolve from disk
    paths = job.get('paths') or {}
    if not paths:
        out_dir = os.path.join(EVAL_OUTPUT_DIR, job_id)
        paths = {
            'json': os.path.join(out_dir, 'evaluation_results.json'),
            'csv': os.path.join(out_dir, 'evaluation_results.csv'),
            'html': os.path.join(out_dir, 'evaluation_report.html'),
            'pdf': os.path.join(out_dir, 'evaluation_report.pdf')
        }

    return jsonify({'status': 'success', 'paths': paths, 'summary': job.get('summary')})


@app.route('/api/evaluate/jobs', methods=['GET'])
def evaluation_jobs_list():
    """Return a list of existing evaluation jobs by merging in-memory tracking
    with any persisted reports on disk (evaluations/<job_id>/).
    """
    jobs = {}

    # Start with in-memory jobs
    for jid, meta in evaluation_jobs.items():
        jobs[jid] = meta.copy()

    # Scan evaluations directory for persisted job outputs
    if os.path.exists(EVAL_OUTPUT_DIR):
        for name in os.listdir(EVAL_OUTPUT_DIR):
            job_dir = os.path.join(EVAL_OUTPUT_DIR, name)
            if not os.path.isdir(job_dir):
                continue

            # If already tracked in-memory, ensure paths/summary filled
            if name in jobs:
                if not jobs[name].get('paths'):
                    json_path = os.path.join(job_dir, 'evaluation_results.json')
                    if os.path.exists(json_path):
                        try:
                            with open(json_path, 'r', encoding='utf-8') as fh:
                                payload = json.load(fh)
                            jobs[name]['summary'] = payload.get('summary')
                            jobs[name]['paths'] = {
                                'json': os.path.join(job_dir, 'evaluation_results.json'),
                                'csv': os.path.join(job_dir, 'evaluation_results.csv'),
                                'html': os.path.join(job_dir, 'evaluation_report.html'),
                                'pdf': os.path.join(job_dir, 'evaluation_report.pdf')
                            }
                        except Exception:
                            pass
                continue

            # Build record from filesystem
            record = {
                'status': 'finished' if (os.path.exists(os.path.join(job_dir, 'evaluation_report.html')) or os.path.exists(os.path.join(job_dir, 'evaluation_report.pdf'))) else 'unknown',
                'dataset_path': None,
                'top_k': None,
                'created_at': os.path.getmtime(job_dir),
                'paths': {},
                'summary': None
            }
            json_path = os.path.join(job_dir, 'evaluation_results.json')
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r', encoding='utf-8') as fh:
                        payload = json.load(fh)
                    record['summary'] = payload.get('summary')
                    record['paths'] = {
                        'json': os.path.join(job_dir, 'evaluation_results.json'),
                        'csv': os.path.join(job_dir, 'evaluation_results.csv'),
                        'html': os.path.join(job_dir, 'evaluation_report.html'),
                        'pdf': os.path.join(job_dir, 'evaluation_report.pdf')
                    }
                except Exception:
                    pass

            jobs[name] = record

    # Ensure each job includes job_id
    for jid in list(jobs.keys()):
        try:
            jobs[jid]['job_id'] = jid
        except Exception:
            pass

    return jsonify({'status': 'success', 'jobs': list(jobs.values())})


@app.route('/api/evaluate/img/<job_id>/<filename>', methods=['GET'])
def evaluation_image(job_id, filename):
    """Serve image files (PNG, JPG, etc.) from evaluation job directory."""
    out_dir = os.path.join(EVAL_OUTPUT_DIR, job_id)
    if not os.path.isdir(out_dir):
        return jsonify({'status': 'error', 'message': 'job_id not found'}), 404

    # Restrict to image files only
    allowed_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.svg'}
    ext = os.path.splitext(filename)[1].lower()
    if ext not in allowed_extensions:
        return jsonify({'status': 'error', 'message': 'file type not allowed'}), 400

    safe_name = os.path.basename(filename)
    file_path = os.path.join(out_dir, safe_name)

    # Prevent path traversal
    if not os.path.exists(file_path) or not os.path.commonpath([file_path, out_dir]) == out_dir:
        return jsonify({'status': 'error', 'message': 'image not found'}), 404

    from flask import send_file
    mime_types = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.svg': 'image/svg+xml'
    }
    mimetype = mime_types.get(ext, 'image/png')
    return send_file(file_path, mimetype=mimetype)


@app.route('/api/evaluate/download/<job_id>/<path:filename>', methods=['GET'])
def evaluation_download(job_id, filename):
    # Allow downloads for jobs tracked in-memory or only present on disk
    out_dir = os.path.join(EVAL_OUTPUT_DIR, job_id)
    if not os.path.isdir(out_dir):
        return jsonify({'status': 'error', 'message': 'job_id not found'}), 404

    # Prevent path traversal by restricting to known filenames
    allowed_files = {
        'evaluation_report.html',
        'evaluation_report.pdf',
        'evaluation_results.json',
        'evaluation_results.csv',
        'failures.csv'
    }
    safe_name = os.path.basename(filename)
    if safe_name not in allowed_files:
        return jsonify({'status': 'error', 'message': 'file not allowed'}), 400

    file_path = os.path.join(out_dir, safe_name)
    if not os.path.exists(file_path):
        return jsonify({'status': 'error', 'message': 'file not found'}), 404

    from flask import send_file
    ext = os.path.splitext(safe_name)[1].lower()
    
    # Serve HTML inline with rewritten image paths
    if ext == '.html':
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Replace relative image paths with API endpoints
        # This handles: src="image.png" or src='image.png'
        html_content = re.sub(
            r'src=["\']([^"\']+\.(?:png|jpg|jpeg|gif|svg))["\']',
            rf'src="/api/evaluate/img/{job_id}/\1"',
            html_content,
            flags=re.IGNORECASE
        )
        
        return Response(html_content, mimetype='text/html')

    # PDFs should be served inline when possible, but we send as attachment for reliable download
    if ext == '.pdf':
        return send_file(file_path, mimetype='application/pdf', as_attachment=True)

    # JSON/CSV/other text files - send as attachment so browser downloads them
    if ext in ('.json', '.csv'):
        # choose mimetype
        mtype = 'application/json' if ext == '.json' else 'text/csv'
        return send_file(file_path, mimetype=mtype, as_attachment=True)

    # Fallback
    return send_file(file_path, as_attachment=True)


# ---------------------------------------------------------------
# Web UI Routes
# ---------------------------------------------------------------
@app.route('/', methods=['GET'])
def index():
    """Serve the main UI page."""
    return render_template('index.html')


@app.route('/evaluation', methods=['GET'])
def evaluation_dashboard():
    """Serve the evaluation dashboard page."""
    return render_template('evaluation_dashboard.html')


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    is_ready = len(chunks) > 0
    return jsonify({
        'status': 'healthy',
        'system_initialized': is_ready,
        'chunks_loaded': len(chunks)
    })


# ---------------------------------------------------------------
# Error Handlers
# ---------------------------------------------------------------
@app.errorhandler(404)
def not_found(error):
    return jsonify({'status': 'error', 'message': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'status': 'error', 'message': 'Internal server error'}), 500


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
if __name__ == '__main__':
    print("[INFO] Starting Flask app on http://127.0.0.1:5000")
    print("[INFO] Use POST /api/initialize to load data and build indices")
    print("[INFO] Use POST /api/query with {'query': '...'} to perform queries")
    app.run(debug=True, host='127.0.0.1', port=5000)
