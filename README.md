# Hybrid RAG System - Flask App

This is a Flask-based web application for a Hybrid Retrieval-Augmented Generation (RAG) system combining dense embeddings (FAISS), sparse retrieval (BM25), and reciprocal rank fusion (RRF).

## Features

- **Dense Retrieval**: Uses sentence transformers with FAISS for semantic search
- **Sparse Retrieval**: Uses BM25 for keyword-based search
- **Reciprocal Rank Fusion**: Combines both methods for improved results
- **Text Generation**: Uses Google's FLAN-T5 model for answer generation
- **Web UI**: Clean, modern Flask interface with real-time results
- **REST API**: JSON endpoints for programmatic access
- **Metadata Tracking**: Each chunk stores URL, title, and unique ID

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python flask_app.py
```

## Installation

1. Create and activate a virtual environment:
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Optional GPU FAISS (only if you have CUDA installed):
```bash
pip uninstall -y faiss-cpu
pip install faiss-gpu
```

## Dependencies

Key packages from `requirements.txt`:
- `flask`
- `requests`
- `beautifulsoup4`
- `numpy`, `scipy`, `scikit-learn`
- `faiss-cpu` (or `faiss-gpu`)
- `sentence-transformers`
- `rank-bm25`
- `transformers`
- `torch`
- `pandas`, `matplotlib`, `seaborn`, `reportlab`, `tqdm`

## Project Structure

```
hybrid-rag-system/
??? flask_app.py                      # Main Flask application
??? hybrid_rag_system.py              # Core RAG logic (also used by evaluation)
??? evaluate_rag.py                   # Evaluation script + report generation
??? data/
?   ??? 200_fixed_urls.json           # Fixed Wikipedia URL list (200 URLs)
?   ??? wikipedia_qa_100.json          # Sample QA dataset (if generated)
??? requirements.txt                  # Python dependencies
??? templates/
?   ??? index.html                    # Web UI template
??? README.md                         # This file
```

## Usage

### Run the System (Web UI)

1. Start the Flask server:
```bash
python flask_app.py
```

2. Open `http://127.0.0.1:5000`
3. Click "Initialize System" to load URLs and build indices (first run can take 5-15 minutes).
4. Enter a question and click "Search".

### Run the System (REST API)

1. Initialize the system:
```bash
curl -X POST http://127.0.0.1:5000/api/initialize   -H "Content-Type: application/json"
```

Response:
```json
{
  "status": "success",
  "message": "System initialized with 5000 chunks from 200 URLs"
}
```

2. Check health:
```bash
curl http://127.0.0.1:5000/health
```

Response:
```json
{
  "status": "healthy",
  "system_initialized": true,
  "chunks_loaded": 5000
}
```

3. Query the system:
```bash
curl -X POST http://127.0.0.1:5000/api/query   -H "Content-Type: application/json"   -d '{"query": "What is machine learning?", "top_k": 10}'
```

Response:
```json
{
  "status": "success",
  "query": "What is machine learning?",
  "answer": "Machine learning is a subset of artificial intelligence...",
  "retrieved_chunks": [
    {
      "id": "uuid-1",
      "title": "Machine Learning",
      "url": "https://en.wikipedia.org/wiki/Machine_learning",
      "text": "Machine learning chunk text...",
      "score": 0.8234
    }
  ],
  "response_time_seconds": 2.45
}
```

## Evaluation

The evaluation script uses the same RAG pipeline and computes retrieval quality metrics, plus generates CSV/JSON/HTML/PDF reports.

1. Start the Flask app (required for API-based evaluation):
```bash
python flask_app.py
```

2. Run evaluation:
```bash
python evaluate_rag.py --dataset data/wikipedia_qa_100.json --out evaluations/last
```

3. Outputs (saved to the `--out` directory):
- `evaluation_results.json`
- `evaluation_results.csv`
- `evaluation_report.html`
- `evaluation_report.pdf`
- `metric_comparison.png`, `dense_mrr_dist.png`, `latency_dist.png`, `retrieval_heatmap.png`

Notes:
- First evaluation run will be slower due to model downloads and corpus caching.
- Typical runtime depends on CPU/GPU and dataset size; plan for several minutes on CPU for 100 QA pairs.

## Data Source (Fixed 200 Wikipedia URLs)

The system uses a fixed list of 200 Wikipedia URLs stored in `data/200_fixed_urls.json`. The RAG corpus is built from this list and remains deterministic unless the list is changed.

Snippet:
```json
{
  "fixed_wiki_urls": [
    "https://en.wikipedia.org/wiki/Physics",
    "https://en.wikipedia.org/wiki/Chemistry",
    "https://en.wikipedia.org/wiki/Biology",
    "https://en.wikipedia.org/wiki/Astronomy",
    "https://en.wikipedia.org/wiki/Geology"
  ]
}
```

## Wikipedia Q&A Pair Generator

### Features
- Fetches content directly from Wikipedia URLs
- Generates factual question-answer pairs
- Outputs in clean JSON format
- Configurable number of Q&A pairs
- Respectful to Wikipedia servers (includes delays)

### Input Format

The script expects a JSON file with Wikipedia URLs in the following format:

```json
{
  "fixed_wiki_urls": [
    "https://en.wikipedia.org/wiki/Physics",
    "https://en.wikipedia.org/wiki/Chemistry",
    ...
  ]
}
```

### Output Format

The generated Q&A dataset follows this structure:

```json
{
  "dataset": "wikipedia_qa_100",
  "data": [
    {
      "id": 1,
      "question": "What is Physics?",
      "answer": "Physics is the natural science of matter, involving the study of matter...",
      "source_ids": ["Physics"],
      "category": "Factual"
    },
    ...
  ]
}
```
### Usage

### Option 1: Python Script

```bash
python wikipedia_qa_generator.py
```

The script will:
1. Load URLs from `200_fixed_urls.json`
2. Fetch Wikipedia articles (with progress updates)
3. Generate 100 Q&A pairs
4. Save to `wikipedia_qa_100.json`

### Option 2: Jupyter Notebook

Open `wikipedia_qa_generator.ipynb` in Jupyter or Google Colab and run the cells in order.

### Configuration

Edit these variables in the script or notebook:

```python
urls_file = "200_fixed_urls.json"      # Input file with URLs
output_file = "wikipedia_qa_100.json"   # Output file
num_qa_pairs = 100                      # Number of Q&A pairs to generate
max_articles_to_fetch = 150             # How many articles to fetch
```

### How It Works

1. **URL Loading**: Reads Wikipedia URLs from the input JSON file
2. **Content Fetching**: Downloads and parses Wikipedia articles using BeautifulSoup
3. **Text Extraction**: Extracts the main content paragraphs from each article
4. **Q&A Generation**: Creates question-answer pairs based on:
   - Article title
   - First meaningful sentence/paragraph
   - Content patterns (location, time, definition, etc.)
5. **Output**: Saves in the specified JSON format with IDs and metadata


## Configuration

### Chunking Parameters
Edit `flask_app.py` to modify chunking behavior:

```python
parts = chunk_text_with_metadata(text, url, title, chunk_size=300, overlap=50)
```

- `chunk_size`: Number of tokens per chunk (default: 300)
- `overlap`: Token overlap between consecutive chunks (default: 50)
- Satisfies requirement: 200-400 tokens with 50-token overlap

### Model Selection
To use a different model, update `flask_app.py`:

```python
# For dense embeddings:
dense_index, _, dense_model = build_dense_index(chunks, model_name="sentence-transformers/all-mpnet-base-v2")

# For text generation:
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
gen_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
```

## Performance Notes

- **First initialization**: 5-15 minutes (downloads models, scrapes URLs, builds indices)
- **Subsequent queries**: 2-5 seconds (after initialization)
- **Memory**: ~4-8 GB for full system with 200 URLs
- **GPU acceleration**: Speeds up embeddings generation by 5-10x

## Metadata Structure

Each chunk includes:

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",  // Unique UUID
  "url": "https://en.wikipedia.org/wiki/Example",
  "title": "Example Article",
  "text": "Chunk text content..."
}
```

## Troubleshooting

### Issue: "System not initialized" on query
**Solution**: Call `/api/initialize` first to load data and build indices.

### Issue: Slow initialization
**Solution**: This is normal. Models are large (~2GB+). Use a GPU if available, or reduce the number of URLs in `200_fixed_urls.json`.

### Issue: Out of memory
**Solution**: 
- Reduce chunk_size or max_chars in `fetch_text_from_url()`
- Use fewer URLs
- Use `faiss-gpu` for GPU acceleration

### Issue: Model download fails
**Solution**: Ensure internet connection is stable. Models are cached in `~/.cache/huggingface/`. You can manually download:
```bash
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2
huggingface-cli download google/flan-t5-base
```

## Switching from Streamlit to Flask

If you were using the Streamlit version (`hybrid_rag_system.py`), here's what changed:

| Aspect | Streamlit | Flask |
|--------|-----------|-------|
| **Interface** | Widget-based (st.text_input, st.button) | HTML/JavaScript UI + REST API |
| **Initialization** | Automatic on startup | Explicit `/api/initialize` call |
| **Performance** | Recompute on every interaction | Cached indices, persistent memory |
| **Scalability** | Single user | Multiple concurrent users |
| **API** | Built-in sharing (ngrok) | REST endpoints |
| **State Management** | Automatic | Manual with locks |

## API Reference

### POST /api/initialize
Initialize the RAG system with Wikipedia URLs and build indices.
- **Parameters**: None
- **Returns**: `{status, message}`

### POST /api/query
Submit a query to the system.
- **Parameters**: 
  - `query` (string): Your question
  - `top_k` (int, optional): Number of chunks to retrieve (default: 10)
- **Returns**: `{status, query, answer, retrieved_chunks, response_time_seconds}`

### GET /health
Check system health and initialization status.
- **Parameters**: None
- **Returns**: `{status, system_initialized, chunks_loaded}`

### GET /
Serve the web UI.

## License

This project is for educational purposes.

## References

- [Sentence Transformers](https://www.sbert.net/)
- [FAISS Documentation](https://faiss.ai/)
- [BM25 (Rank-BM25)](https://github.com/dorianbrown/rank_bm25)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [Flask Documentation](https://flask.palletsprojects.com/)
