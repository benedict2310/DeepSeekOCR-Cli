# DeepSeek-OCR Hybrid Search Integration

**Version:** 1.0
**Target Platform:** macOS Apple Silicon (M1–M4), Python ≥3.10

## Overview

This extension adds **visual and text-based hybrid search** capabilities to DeepSeek-OCR Mac CLI. Every OCR run can now:

1. Extract **vision embeddings** from DeepSeek-OCR (page-level)
2. Maintain a **visual HNSW index** for layout/content similarity search
3. Build/update a **text index** from OCR'd Markdown
4. Expose an optional **FastAPI web API** for demo and search

All processing is **local**, **offline**, and **Metal-accelerated** (MPS) with no CUDA/flash-attention requirements.

## Features

- 🖼️ **Visual Search**: Find pages by visual similarity (layout-aware)
- 📝 **Text Search**: Semantic text search using sentence transformers
- 🔀 **Hybrid Search**: Combine visual and text scores with configurable weights
- 🚀 **Incremental Updates**: Append to indexes without full rebuilds
- 🌐 **Web Demo**: FastAPI-based UI for testing searches
- 📊 **HNSW Indexes**: Fast approximate nearest neighbor search

## Installation

### Prerequisites

```bash
# Ensure base DeepSeek-OCR dependencies are installed
pip install -r requirements.txt

# Install hybrid search dependencies
pip install hnswlib sentence-transformers

# Optional: Install FastAPI for web demo
pip install fastapi uvicorn
```

### Verify Installation

```bash
python -c "import hnswlib, sentence_transformers; print('✓ Dependencies ready')"
```

## Quick Start

### 1. OCR with Index Updates

Process documents and update both visual and text indexes:

```bash
./deepseek_ocr_mac.py docs/invoices.pdf \
  --update-index \
  --visual-index ./vi_index \
  --text-index ./ti_index
```

**What happens:**
- PDF is OCR'd to Markdown (as usual)
- Each page image is embedded using DeepSeek-OCR's vision encoder
- Visual embeddings are added to `./vi_index/`
- Text embeddings are generated from OCR text
- Text embeddings are added to `./ti_index/`

### 2. Search by Image (Visual Search)

Find pages visually similar to a screenshot or image:

```python
from PIL import Image
from visual_index import VisualIndex, DeepSeekVisionEmbedder
from pathlib import Path

# Load visual index
vi = VisualIndex(space="cosine")
vi.load(Path("./vi_index"))

# Load embedder and query
embedder = DeepSeekVisionEmbedder("deepseek-ai/DeepSeek-OCR")
img = Image.open("query_screenshot.png").convert("RGB")
qvec = embedder.embed_image(img)

# Search
results = vi.query(qvec, topk=5)
for meta, score in results:
    print(f"{score:.4f} :: {meta['display']}")
```

### 3. Search by Text (Semantic Search)

Find pages containing relevant content:

```python
from hybrid_search import TextIndex, load_st_model
from pathlib import Path
import numpy as np

# Load text index
tidx = TextIndex(space="cosine")
tidx.load(Path("./ti_index"))

# Encode query
st = load_st_model()
query = "quarterly revenue table with VAT breakdown"
qvec = st.encode([query], normalize_embeddings=True)[0].astype(np.float32)

# Search
results = tidx.query(qvec, k=5)
for doc, score in results:
    print(f"{score:.4f} :: {doc['name']} ({doc['path']})")
```

### 4. Web Demo

Launch the FastAPI web interface:

```bash
# Start server
uvicorn app:app --reload --port 8000

# Open browser
open http://127.0.0.1:8000
```

**Features:**
- Text search form
- Image upload for visual search
- Real-time results with similarity scores
- Interactive UI with score visualization

## CLI Reference

### New Arguments

```
--update-index
    Update visual (and optional text) index after OCR

--visual-index PATH
    Folder for visual HNSW index (created if missing)
    Example: --visual-index ./vi_index

--text-index PATH
    Folder for text HNSW index (created if missing)
    Example: --text-index ./ti_index

--text-embed-model MODEL
    Text embedding model for text index
    Default: sentence-transformers/all-MiniLM-L6-v2

--index-batch SIZE
    Batch size when appending to indexes (default: 128)
```

### Example Commands

```bash
# OCR only (no indexing)
./deepseek_ocr_mac.py document.pdf

# OCR + visual index only
./deepseek_ocr_mac.py document.pdf --update-index --visual-index ./vi_index

# OCR + both indexes
./deepseek_ocr_mac.py document.pdf \
  --update-index \
  --visual-index ./vi_index \
  --text-index ./ti_index

# Process directory with custom text model
./deepseek_ocr_mac.py ./scans/ \
  --update-index \
  --visual-index ./vi_index \
  --text-index ./ti_index \
  --text-embed-model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

## File Structure

After running with `--update-index`, your directory will look like:

```
project/
├── deepseek_ocr_mac.py        # Main CLI
├── visual_index.py            # Vision embedding & HNSW index
├── hybrid_search.py           # Text index & hybrid search
├── app.py                     # FastAPI web demo (optional)
├── vi_index/                  # Visual index
│   ├── hnsw.bin              # HNSW index data
│   ├── meta.json             # Page metadata
│   └── dim.txt               # Embedding dimension
├── ti_index/                  # Text index
│   ├── hnsw.bin              # HNSW index data
│   ├── docs.json             # Document metadata
│   └── dim.txt               # Embedding dimension
└── outputs/                   # OCR outputs (as before)
    └── merged_output.md
```

## Architecture

### Visual Index Pipeline

1. **Image Input** → PIL Image (page screenshot or rendered PDF page)
2. **Preprocessing** → Resize to 1024px max dimension, normalize
3. **Vision Encoding** → DeepSeek-OCR model extracts features
4. **Pooling** → Mean-pool vision tokens to single vector
5. **Normalization** → L2 normalize for cosine similarity
6. **Indexing** → Add to HNSW index with metadata

### Text Index Pipeline

1. **OCR Text** → Markdown output from DeepSeek-OCR
2. **Text Encoding** → Sentence transformer encodes text
3. **Normalization** → L2 normalize embeddings
4. **Indexing** → Add to HNSW index with document metadata

### Hybrid Search

Combines visual and text scores using weighted fusion:

```
final_score = α × text_score + (1 - α) × visual_score
```

Default: α = 0.6 (60% text, 40% visual)

## API Reference

### `VisualIndex`

```python
class VisualIndex:
    def __init__(self, space="cosine", dim=None)
    def build(self, X: np.ndarray, meta: list, M=32, efC=200)
    def add(self, X: np.ndarray, meta: list)
    def query(self, vec: np.ndarray, topk=5) -> List[Tuple[dict, float]]
    def save(self, folder: Path)
    def load(self, folder: Path)
    def resize(self, new_max: int)
```

### `TextIndex`

```python
class TextIndex:
    def __init__(self, space="cosine", dim=None)
    def build(self, X: np.ndarray, docs: list, M=32, efC=200)
    def add(self, X: np.ndarray, docs: list)
    def query(self, vec: np.ndarray, k=5) -> List[Tuple[dict, float]]
    def save(self, folder: Path)
    def load(self, folder: Path)
    def resize(self, new_max: int)
```

### `DeepSeekVisionEmbedder`

```python
class DeepSeekVisionEmbedder:
    def __init__(self, model_id="deepseek-ai/DeepSeek-OCR", device=None, dtype=torch.float32)
    def embed_image(self, pil_image) -> np.ndarray
```

## FastAPI Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI homepage |
| `/search_text` | POST | Text search (form: `q`, `topk`) |
| `/search_image` | POST | Image search (multipart: `file`, `topk`) |
| `/health` | GET | Health check and index status |
| `/docs` | GET | OpenAPI documentation |

## Performance Notes

### Index Size

- **Visual embeddings**: ~2-4 KB per page (depends on model)
- **Text embeddings**: ~1.5 KB per page (all-MiniLM-L6-v2, 384d)
- **HNSW overhead**: ~50-100 bytes per vector (M=32)

**Example:** 1000 pages ≈ 3-5 MB total index size

### Query Speed

- **HNSW search**: < 1ms for k=5 on 10K pages (Apple Silicon)
- **Vision embedding**: ~200-500ms per image (M1/M2)
- **Text embedding**: ~10-50ms per query (sentence transformer)

### Scaling

- **Incremental updates**: Use `resize()` + `add()` for append-only growth
- **Batch processing**: Process pages in batches to reduce memory
- **Periodic rebuild**: Rebuild index after 5-10× growth for optimal quality

## Troubleshooting

### Mac/MPS Specific Issues

**Problem:** Repeating/gibberish text during embedding extraction

**Solution:**
- Ensure `attn_implementation="eager"` in model loading
- Use `torch_dtype=torch.float32` (not bfloat16)
- Set `tokenizer.padding_side = "right"`
- Avoid autocast on MPS

**Problem:** HNSW errors on `add_items()`

**Solution:**
- Call `resize_index(new_max)` before adding new items
- Ensure new max >= current size + new items

**Problem:** Out of memory during indexing

**Solution:**
- Process in smaller batches
- Clear MPS cache: `torch.mps.empty_cache()`
- Reduce DPI for PDF rendering (e.g., 200 instead of 300)

### Index Quality

**Problem:** Poor visual search results

**Solution:**
- Ensure consistent image preprocessing (same resolution policy)
- Use same DPI for query images as indexed pages
- Check that normalization is applied consistently

**Problem:** Poor text search results

**Solution:**
- Try a different sentence transformer model
- Increase `k` to retrieve more candidates
- Ensure OCR quality is good (check `--strict --min-words`)

## Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run hybrid search tests only
pytest tests/test_hybrid_search.py

# Run with coverage
pytest --cov=. --cov-report=html tests/test_hybrid_search.py

# Skip slow tests (model downloads)
pytest -m "not slow" tests/test_hybrid_search.py
```

## Advanced Usage

### Custom Hybrid Search

```python
from hybrid_search import hybrid_search

results = hybrid_search(
    query="quarterly revenue",
    text_index=tidx,
    visual_index=vi,
    query_image=screenshot,  # Optional PIL image
    text_model=st,
    alpha=0.7,  # 70% text, 30% visual
    k=10
)

for doc, score in results:
    print(f"{score:.4f} :: {doc}")
```

### Multi-Index Management

```python
# Separate indexes for different document types
invoices_vi = VisualIndex()
invoices_vi.load(Path("./indexes/invoices_vi"))

reports_vi = VisualIndex()
reports_vi.load(Path("./indexes/reports_vi"))

# Search both and merge results
results1 = invoices_vi.query(qvec, topk=5)
results2 = reports_vi.query(qvec, topk=5)
merged = sorted(results1 + results2, key=lambda x: x[1], reverse=True)[:5]
```

### Batch Processing

```python
# Process multiple documents and update index once
page_vectors = []
page_metas = []

for pdf_file in pdf_files:
    pages = process_pdf(pdf_file)  # Your processing logic
    for i, (img, text, name) in enumerate(pages):
        vec = embedder.embed_image(img)
        page_vectors.append(vec)
        page_metas.append({"display": name, "doc": pdf_file.name, "page": i})

# Bulk add
X = np.stack(page_vectors).astype(np.float32)
vi.resize(len(vi.meta) + len(X))
vi.add(X, page_metas)
vi.save(index_dir)
```

## Future Enhancements

Potential upgrades for higher accuracy:

- **Reranking**: Add cross-encoder reranking on top-K results
- **Region-based search**: Crop pages into blocks for finer-grained search
- **Document-level fusion**: Aggregate page scores per document
- **Multi-modal queries**: Support combined text + image queries
- **Query expansion**: Automatically expand queries with synonyms

## License

See main project LICENSE file.

## Credits

Built on:
- [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR)
- [hnswlib](https://github.com/nmslib/hnswlib)
- [sentence-transformers](https://www.sbert.net/)
- [FastAPI](https://fastapi.tiangolo.com/)

---

**Questions?** Open an issue on GitHub or consult the main README.
