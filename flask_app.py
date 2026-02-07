import json
import time
from flask import Flask, render_template, request, jsonify
import requests
import uuid
from threading import Lock, Thread
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
#tokenizer = None
#gen_model = None
data_lock = Lock()  # Thread-safe access to shared resources

print("[INFO] Flask app initialized. Indices not yet loaded.")

# Evaluation job tracking
evaluation_jobs = {}
EVAL_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'evaluations')
os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------
# Initialization Route
# ---------------------------------------------------------------
@app.route('/api/initialize', methods=['POST'])
def initialize_system():
    """Initialize/load the RAG system with URLs and build indices."""
    global chunks, dense_index, dense_model, bm25, tokenizer, gen_model
    
    with data_lock:
        try:
            print("[INFO] Starting system initialization...")
            
            # Load models
            # print("[INFO] Loading tokenizer and model...")
            # tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
            # gen_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
            # print("[INFO] Models loaded.")
            
            # Load and chunk data
            print("[INFO] Loading URLs...")
            urls = load_urls()
            print(f"[INFO] Loaded {len(urls)} URLs.")
            
            all_chunks = []
            successful_urls = 0
            for idx, url in enumerate(urls):
                if idx % 10 == 0:
                    print(f"[INFO] Processing URL {idx+1}/{len(urls)}...")
                title, text = fetch_text_from_url(url)
                if text:
                    parts = chunk_text_with_metadata(text, url, title, chunk_size=300, overlap=50)
                    all_chunks.extend(parts)
                    successful_urls += 1
            
            print(f"[INFO] Processed {successful_urls} URLs successfully. Total chunks: {len(all_chunks)}")
            
            if not all_chunks:
                return jsonify({'status': 'error', 'message': 'No chunks created from URLs'}), 400
            
            chunks = all_chunks
            
            # Build indices
            dense_index, _, dense_model = build_dense_index(chunks)
            bm25, _ = build_sparse_index(chunks)
            
            return jsonify({
                'status': 'success',
                'message': f'System initialized with {len(chunks)} chunks from {successful_urls} URLs'
            })
        except Exception as e:
            print(f"[ERROR] Initialization failed: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 500


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
    try:
        evaluation_jobs[job_id]['status'] = 'running'
        evaluation_jobs[job_id]['started_at'] = time.time()

        data = evaluate_rag.load_dataset(dataset_path)
        summary, detailed = evaluate_rag.evaluate_dataset(data, api_url=api_url, init_url=init_url, top_k=top_k, progress=False)

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
    """Start an evaluation job. Accepts JSON: {dataset_path, top_k, api_url, init_url, run_sync}
    Returns a job_id for status tracking.
    """
    data = request.get_json() or {}
    dataset_path = data.get('dataset_path', r'path\to\hybrid-rag-system\data\wikipedia_qa_100_updated.json')
    top_k = int(data.get('top_k', 5))
    api_url = data.get('api_url', 'http://127.0.0.1:5000/api/query')
    init_url = data.get('init_url', 'http://127.0.0.1:5000/api/initialize')
    run_sync = bool(data.get('run_sync', False))

    job_id = str(uuid.uuid4())
    evaluation_jobs[job_id] = {
        'status': 'pending',
        'dataset_path': dataset_path,
        'top_k': top_k,
        'api_url': api_url,
        'init_url': init_url,
        'created_at': time.time()
    }

    if run_sync:
        # Run inline (blocking)
        _run_evaluation_job(job_id, dataset_path, top_k, api_url, init_url)
    else:
        t = Thread(target=_run_evaluation_job, args=(job_id, dataset_path, top_k, api_url, init_url), daemon=True)
        t.start()

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
        return jsonify({'status': 'error', 'message': 'job_id not found'}), 404
    if job.get('status') != 'finished':
        return jsonify({'status': job.get('status'), 'message': 'report not ready'}), 400
    return jsonify({'status': 'success', 'paths': job.get('paths'), 'summary': job.get('summary')})


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


@app.route('/api/evaluate/download/<job_id>/<path:filename>', methods=['GET'])
def evaluation_download(job_id, filename):
    job = evaluation_jobs.get(job_id)
    if not job:
        return jsonify({'status': 'error', 'message': 'job_id not found'}), 404
    out_dir = os.path.join(EVAL_OUTPUT_DIR, job_id)
    file_path = os.path.join(out_dir, filename)
    if not os.path.exists(file_path):
        return jsonify({'status': 'error', 'message': 'file not found'}), 404
    from flask import send_file
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
