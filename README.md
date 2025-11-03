# DeepSeek-OCR Mac CLI

[![CI/CD Pipeline](https://github.com/benedict2310/DeepSeekOCR-Cli/actions/workflows/ci.yml/badge.svg)](https://github.com/benedict2310/DeepSeekOCR-Cli/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Offline OCR for images and PDFs using DeepSeek-OCR on macOS (Apple Silicon).

## Overview

This CLI tool provides simple, local OCR capabilities leveraging the DeepSeek-OCR model for high-quality text and layout extraction. Output is generated in Markdown format with automatic page separation.

**Key Features:**
- Runs natively on Apple Silicon (M1-M4) using PyTorch MPS acceleration
- Supports both images and PDFs with auto-conversion
- Outputs clean Markdown-structured text
- Fully offline after initial model download
- No CUDA or external binaries required

## Requirements

- macOS 13+ (Ventura or later)
- Python 3.9+
- Apple Silicon (M1, M2, M3, M4) recommended
- ~10GB disk space for model cache

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/benedict2310/DeepSeekOCR-Cli.git
cd DeepSeekOCR-Cli
```

### 2. Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Make executable

```bash
chmod +x deepseek_ocr_mac.py
```

### 5. (Optional) Install globally

```bash
ln -s "$(pwd)/deepseek_ocr_mac.py" /usr/local/bin/deepseek-ocr
```

## Usage

### Basic Examples

**Process a single image:**
```bash
./deepseek_ocr_mac.py sample.png
```

**Process a PDF file:**
```bash
./deepseek_ocr_mac.py document.pdf
```

**Process a directory:**
```bash
./deepseek_ocr_mac.py ./scans
```

### Advanced Options

**High-quality OCR with custom DPI:**
```bash
./deepseek_ocr_mac.py mypaper.pdf --dpi 360 --base-size 1280 --image-size 768
```

**Disable cropping and compression:**
```bash
./deepseek_ocr_mac.py file.pdf --no-crop --no-compress
```

**Custom output directory:**
```bash
./deepseek_ocr_mac.py document.pdf -o results
```

### Extension Features

**Extract tables to CSV:**
```bash
./deepseek_ocr_mac.py document.pdf --emit-csv
# Or TSV format
./deepseek_ocr_mac.py document.pdf --emit-csv=tsv
```

**Extract LaTeX equations:**
```bash
./deepseek_ocr_mac.py paper.pdf --math-latex
```

**Auto-tag code blocks:**
```bash
./deepseek_ocr_mac.py docs.pdf --code-lang
```

**Generate RAG chunks for vector databases:**
```bash
./deepseek_ocr_mac.py document.pdf --emit-chunks
```

**Compression presets for speed/quality trade-off:**
```bash
./deepseek_ocr_mac.py file.pdf --compression high  # Fastest
./deepseek_ocr_mac.py file.pdf --compression med   # Balanced
./deepseek_ocr_mac.py file.pdf --compression low   # Best quality (default)
```

**Quality gate for CI/CD:**
```bash
./deepseek_ocr_mac.py file.pdf --strict --min-words 50
# Exit code 1 if any page has < 50 words
```

**Extract bounding boxes:**
```bash
./deepseek_ocr_mac.py document.pdf --emit-boxes --overlay
```

**Combine multiple features:**
```bash
./deepseek_ocr_mac.py research.pdf \
  --emit-csv \
  --math-latex \
  --code-lang \
  --emit-chunks \
  --strict
```

### Command-Line Options

```
usage: deepseek_ocr_mac.py [-h] [-o OUT] [--model MODEL]
                           [--base-size BASE_SIZE] [--image-size IMAGE_SIZE]
                           [--no-crop] [--no-compress] [--dpi DPI]
                           [--emit-csv [{csv,tsv}]] [--math-latex]
                           [--code-lang] [--chart-to-csv] [--emit-boxes]
                           [--overlay] [--compression {low,med,high}]
                           [--workers WORKERS] [--emit-chunks] [--strict]
                           [--min-words MIN_WORDS]
                           path

positional arguments:
  path                  File or folder (image/pdf or dir)

Core options:
  -h, --help            Show this help message
  -o OUT, --out OUT     Output directory (default: outputs)
  --model MODEL         Model name (default: deepseek-ai/DeepSeek-OCR)
  --base-size SIZE      Base resolution (default: from preset)
  --image-size SIZE     Target image size (default: from preset)
  --no-crop             Disable intelligent cropping
  --no-compress         Disable compression
  --dpi DPI             DPI for PDF rendering (default: 288)

Extension features:
  --emit-csv [{csv,tsv}]
                        Extract Markdown tables to CSV/TSV
  --math-latex          Export equations as LaTeX files
  --code-lang           Tag code blocks with detected languages
  --chart-to-csv        (Experimental) Extract chart data to CSV
  --emit-boxes          Emit per-page bounding-box JSON
  --overlay             Render bbox overlays to PNG (requires --emit-boxes)
  --emit-chunks         Emit chunks.jsonl for RAG/vector databases

Performance & Quality:
  --compression {low,med,high}
                        Compression preset (default: low = best quality)
  --workers WORKERS     Parallel workers (default: 1, future use)
  --strict              Fail if quality checks fail
  --min-words MIN_WORDS
                        Min words per page for --strict (default: 20)
```

## Supported Formats

**Images:**
- PNG (.png)
- JPEG (.jpg, .jpeg)
- WebP (.webp)
- BMP (.bmp)
- TIFF (.tif, .tiff)

**Documents:**
- PDF (.pdf)

## Extension Features Reference

### Table Extraction (`--emit-csv`)

Automatically detects and extracts Markdown tables to structured CSV/TSV files.

**Supported formats:**
- `--emit-csv` or `--emit-csv=csv` → CSV format
- `--emit-csv=tsv` → Tab-separated values

**Use cases:**
- Import tables into Excel/Google Sheets
- Data analysis workflows
- Structured data extraction

**Example:** A document with 3 tables will generate:
- `outputs/tables/page_0001_table_1.csv`
- `outputs/tables/page_0003_table_1.csv`
- `outputs/tables/page_0005_table_1.csv`

### Math Extraction (`--math-latex`)

Extracts LaTeX mathematical expressions from OCR output to individual `.tex` files.

**Detects:**
- Inline math: `$x^2 + y^2 = z^2$`
- Display math: `$$\int_0^\infty e^{-x} dx$$`

**Use cases:**
- Re-rendering equations in papers
- Equation databases
- Academic document processing

**Output:** `outputs/equations/page_0001_eq_001.tex`

### Code Language Detection (`--code-lang`)

Automatically detects programming languages in code blocks and adds language tags.

**Supported languages (10):**
python, javascript, typescript, java, cpp, go, rust, sql, bash, json

**Before:**
```
```
def hello():
    print("world")
```
```

**After:**
```
```python
def hello():
    print("world")
```
```

**Use cases:**
- Improved syntax highlighting
- Code documentation
- Developer tutorials

### Chart Data Extraction (`--chart-to-csv`)

**Experimental feature** that extracts chart/graph data from specially formatted code blocks.

**Detects blocks tagged as:**
- ` ```csv`
- ` ```chart-data`

**Output:** `outputs/charts/page_0005_chart_1.csv`

### RAG Chunks (`--emit-chunks`)

Generates fixed-size text chunks optimized for Retrieval-Augmented Generation (RAG) pipelines.

**Chunk specifications:**
- **Size:** 1200 characters per chunk
- **Format:** JSONL (one JSON object per line)
- **Metadata:** Page number, start/end positions

**Use cases:**
- Vector database ingestion (Pinecone, Weaviate, Chroma)
- LangChain/LlamaIndex integration
- Semantic search systems
- Question-answering bots

**Example chunk:**
```json
{
  "page": 1,
  "start": 0,
  "end": 1200,
  "text": "Introduction\n\nThis document describes..."
}
```

### Bounding Boxes (`--emit-boxes`, `--overlay`)

Generates approximate text region bounding boxes using heuristic image processing.

**How it works:**
1. Edge detection on rendered page
2. Grid-based region segmentation (10×5 grid)
3. Content detection via pixel analysis

**Outputs:**
- `--emit-boxes`: JSON file with box coordinates
- `--overlay`: PNG with red rectangles drawn on original image

**JSON format:**
```json
{
  "page": 1,
  "bbox_provider": "heuristic",
  "boxes": [
    {"x": 100, "y": 50, "w": 400, "h": 60, "hint": "text-block"}
  ]
}
```

**Use cases:**
- Layout analysis
- Clickable PDF regions
- Accessibility enhancement
- Visual debugging

**Note:** This is a heuristic approximation. For precise OCR-based bounding boxes, consider using the model's native output (if available in future versions).

### Quality Gates (`--strict`, `--min-words`)

Enforces quality thresholds for automated workflows.

**Behavior:**
- Counts words per page
- If any page < `--min-words`, mark as failed
- Exit with code 1 if `--strict` is enabled

**Use cases:**
- CI/CD pipelines
- Batch processing validation
- Quality assurance

**Example:**
```bash
./deepseek_ocr_mac.py scan.pdf --strict --min-words 50
# Exit code 0 = all pages ≥ 50 words
# Exit code 1 = one or more pages < 50 words
```

### Compression Presets (`--compression`)

Pre-configured quality/speed trade-offs for common scenarios.

| Preset | Base Size | Image Size | Test Compress | Speed | Quality |
|--------|-----------|------------|---------------|-------|---------|
| `low` (default) | 1280 | 768 | No | Slow | Best |
| `med` | 1024 | 640 | Yes | Medium | Good |
| `high` | 896 | 512 | Yes | Fast | Standard |

**Override presets:**
```bash
# Use 'high' preset but with custom base size
./deepseek_ocr_mac.py file.pdf --compression high --base-size 1536
```

## Hybrid Search (NEW!)

The CLI now includes a powerful hybrid search system that combines visual and semantic text search over your OCR'd documents.

### Quick Start

**1. Build indexes while processing documents:**
```bash
./deepseek_ocr_mac.py docs/*.pdf \
  --update-index \
  --visual-index ./vi_index \
  --text-index ./ti_index \
  -o ./outputs
```

**2. Launch the web demo:**
```bash
uvicorn app:app --reload --port 8000
```

**3. Open browser:** http://127.0.0.1:8000

### Features

**Visual Search:**
- Search by image similarity using CLIP embeddings (OpenAI's vision-language model)
- Find visually similar pages based on layout, structure, and content
- Perfect for finding screenshots, diagrams, or documents with similar formatting
- Example: Upload a product photo to find similar product images

**Text Search:**
- Semantic text search using sentence transformers
- Understands meaning, not just keywords
- Search for "coding agents" to find discussions about AI assistants
- Works on OCR-extracted text content

**Hybrid Search:**
- Combines both visual and text search with score fusion
- Best of both worlds for multi-modal document retrieval
- Automatically balances visual similarity and semantic relevance

**Concurrent Processing Safety:**
- Built-in file locking prevents index corruption
- Safe to process multiple documents simultaneously
- Uses `filelock` library with 60-second timeout
- Lock files stored as `.index.lock` in index directories

**Live Index Reload:**
- `/reload` endpoint refreshes indexes without server restart
- Add new documents and reload instantly
- No downtime required for index updates

### Example Use Cases

**Find similar screenshots:**
```bash
# Upload a Twitter screenshot to find other social media captures
curl -X POST -F "file=@screenshot.png" -F "topk=5" http://localhost:8000/search_image
```

**Semantic text search:**
```bash
# Search for discussions about Vulkan graphics
curl -X POST -F "q=Vulkan workstation" -F "topk=3" http://localhost:8000/search_text
```

**Process documents in parallel (safe with file locking):**
```bash
# Process multiple PDFs - file locking prevents race conditions
./deepseek_ocr_mac.py paper1.pdf --update-index --visual-index ./vi --text-index ./ti &
./deepseek_ocr_mac.py paper2.pdf --update-index --visual-index ./vi --text-index ./ti &
wait
```

**Reload indexes after updates:**
```bash
curl -X POST http://localhost:8000/reload
```

### Architecture

- **Visual Index**: CLIP-ViT-B-32 embeddings + HNSW approximate nearest neighbor search
- **Text Index**: all-MiniLM-L6-v2 sentence embeddings + HNSW
- **Storage**: Memory-mapped HNSW indexes with JSON metadata
- **Concurrency**: File locking with 60s timeout prevents corruption
- **Performance**: Sub-second queries even with thousands of pages

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI homepage |
| `/search_text` | POST | Text search (form: `q`, `topk`) |
| `/search_image` | POST | Image search (multipart: `file`, `topk`) |
| `/health` | GET | Health check and index status |
| `/reload` | POST | Reload indexes from disk without restarting server |
| `/docs` | GET | OpenAPI documentation |

### Index Management

**Incremental Updates:**
```bash
# Add new documents to existing indexes
./deepseek_ocr_mac.py new_doc.pdf --update-index --visual-index ./vi --text-index ./ti
```

**Rebuild from Scratch:**
```bash
# Delete old indexes
rm -rf ./vi_index ./ti_index

# Reprocess all documents
./deepseek_ocr_mac.py docs/*.pdf --update-index --visual-index ./vi_index --text-index ./ti_index
```

**Memory Management:**
- Avoid running multiple OCR processes in parallel (high RAM usage)
- Each process loads ~8GB model + MPS memory
- File locking ensures safety, but sequential processing recommended
- Indexes can be safely updated concurrently once models are loaded

For complete documentation, see [HYBRID_SEARCH.md](HYBRID_SEARCH.md).

## Output

Results are saved to the `outputs/` directory (configurable with `-o`):

### Basic Output Structure

```
outputs/
├── merged_output.md          # Combined Markdown output with quality stats
├── sample-p0001.png.text     # Individual page results
└── sample-p0002.png.text
```

- `merged_output.md` - All pages combined with quality summary in YAML frontmatter
- `*.png.text` - Per-page OCR results

### Extended Output Structure (with all features enabled)

```
outputs/
├── merged_output.md          # Combined output with quality stats
├── chunks.jsonl              # RAG chunks (--emit-chunks)
├── tables/
│   ├── page_0001_table_1.csv
│   └── page_0002_table_1.csv
├── equations/
│   ├── page_0001_eq_001.tex
│   ├── page_0001_eq_002.tex
│   └── page_0003_eq_001.tex
├── charts/
│   └── page_0005_chart_1.csv
├── boxes/
│   ├── page_0001.json
│   └── page_0002.json
└── overlays/
    ├── page_0001_overlay.png
    └── page_0002_overlay.png
```

### Quality Summary

The `merged_output.md` includes a YAML frontmatter with processing statistics:

```yaml
---
quality:
  pages: 10
  failed_pages: 0
  success_rate: 100.0%
  total_words: 5847
  min_words: 20
  compression: low
  workers: 1
  processing_time: 45.32s
  model: deepseek-ai/DeepSeek-OCR
---
```

### RAG Chunks Format

The `chunks.jsonl` file (generated with `--emit-chunks`) contains fixed-size chunks optimized for vector databases:

```json
{"page": 1, "start": 0, "end": 1200, "text": "..."}
{"page": 1, "start": 1200, "end": 2400, "text": "..."}
{"page": 2, "start": 0, "end": 1200, "text": "..."}
```

Perfect for LangChain, LlamaIndex, or custom RAG pipelines.

## Performance

Typical performance on Apple Silicon (M3 Pro):

| Compression Preset | Resolution | Pages/min | Quality | Use Case |
|-------------------|------------|-----------|---------|----------|
| `--compression high` | 896×512 | 25-35 | Standard | Quick scans, drafts |
| `--compression med` | 1024×640 | 10-15 | High | General documents |
| `--compression low` | 1280×768 | 5-10 | Very High | Academic papers, archival |

**Compression Presets:**
- **high**: Fastest processing, good for initial scans
- **med**: Balanced quality/speed (default OCR settings)
- **low**: Best quality, recommended for final output (default)

Performance scales with DPI and resolution settings. Higher values = better quality but slower processing.

**Manual Override:** You can override preset values:
```bash
# Start with 'high' preset but use custom resolution
./deepseek_ocr_mac.py file.pdf --compression high --base-size 1536
```

## Architecture

```
User File (PDF/Image)
        │
        ▼
[PyMuPDF] → render pages → temp PNGs
        │
        ▼
[DeepSeek-OCR Model]
        │
        ▼
Markdown Output → outputs/merged_output.md
```

- **Model:** deepseek-ai/DeepSeek-OCR (via Hugging Face Transformers)
- **Device:** MPS (Metal) if available, else CPU
- **Core Dependencies:** torch, transformers, pillow, pymupdf

## Apple Silicon (MPS) Compatibility

This project includes patches for full Apple Silicon MPS (Metal Performance Shaders) support. The original DeepSeek-OCR model was designed for CUDA, but we've implemented the following fixes to enable native macOS acceleration:

### MPS Patches Applied

1. **Attention Implementation**: Uses `eager` attention instead of `flash_attention_2` (MPS doesn't support Flash Attention)
2. **Data Type Handling**: Converts model to `float32` for MPS compatibility (bfloat16 has partial support)
3. **Autocast Disabled**: Removes `torch.autocast` on MPS to prevent numerical instability
4. **Scatter Operation Fix**: Replaces `masked_scatter_` with direct boolean indexing to fix generation loops
5. **Tokenizer Padding**: Sets right-side padding to avoid known MPS bugs

These patches are based on [HuggingFace Discussion #20](https://huggingface.co/deepseek-ai/DeepSeek-OCR/discussions/20) and enable full functionality on Apple Silicon without requiring CUDA.

### Version Requirements

The following specific versions are required for MPS compatibility:

```
transformers==4.46.3
tokenizers==0.20.3
```

These are pinned in `requirements.txt` and will be installed automatically.

## Troubleshooting

### Model Download Issues

First run will download ~8GB model from Hugging Face. Ensure stable internet connection.

```bash
# Check model cache location
ls ~/.cache/huggingface/hub/
```

### MPS Not Available

If MPS acceleration fails, CLI will automatically fall back to CPU:

```
Using device: cpu
```

This is normal on Intel Macs. Performance will be slower but functional.

### Incompatible Transformers Version

If you see errors like `cannot import name 'LlamaFlashAttention2'`, you may have the wrong transformers version:

```bash
# Reinstall with correct versions
pip install transformers==4.46.3 tokenizers==0.20.3
```

### Repetitive Output ("Background Background...")

This was a known issue with MPS that has been fixed in this repository. If you still see it:

1. Ensure you're using the latest code from this repo
2. Verify transformers version is exactly 4.46.3
3. Check that the model is loading with `attn_implementation="eager"`

### PyMuPDF Installation Issues

If PDF processing fails:

```bash
pip install --upgrade pymupdf
```

### Memory Issues

For large PDFs, process in smaller batches or reduce resolution:

```bash
./deepseek_ocr_mac.py large.pdf --image-size 480 --dpi 200
```

## Development

### Project Structure

```
DeepSeekOCR-Cli/
├── deepseek_ocr_mac.py       # Main CLI script
├── requirements.txt           # Python dependencies
├── requirements-dev.txt       # Development dependencies
├── pyproject.toml            # Project configuration
├── pytest.ini                # Pytest configuration
├── setup.sh                  # Automated setup script
├── README.md                 # This file
├── PRD.md                    # Product requirements document
├── .gitignore               # Git ignore rules
├── .github/
│   └── workflows/
│       └── ci.yml           # GitHub Actions CI/CD
├── tests/
│   ├── __init__.py
│   ├── fixtures.py          # Test fixtures and utilities
│   ├── test_unit.py         # Unit tests
│   └── test_integration.py  # Integration tests
└── outputs/                 # Default output directory (created at runtime)
```

### Setting Up Development Environment

```bash
# Clone and enter repository
git clone https://github.com/benedict2310/DeepSeekOCR-Cli.git
cd DeepSeekOCR-Cli

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install development dependencies
pip install -r requirements-dev.txt
```

### Running Tests

The project includes comprehensive unit and integration tests with mocked model dependencies to avoid downloading the large DeepSeek-OCR model during testing.

**Run all tests:**
```bash
pytest
```

**Run specific test files:**
```bash
# Unit tests only
pytest tests/test_unit.py -v

# Integration tests only
pytest tests/test_integration.py -v
```

**Run tests with coverage:**
```bash
pytest --cov=. --cov-report=html --cov-report=term
```

**Run tests by marker:**
```bash
pytest -m unit        # Unit tests only
pytest -m integration # Integration tests only
```

### Code Quality Checks

**Format code with Black:**
```bash
black .
```

**Check formatting:**
```bash
black --check --diff .
```

**Lint with Ruff:**
```bash
ruff check .
```

**Auto-fix linting issues:**
```bash
ruff check --fix .
```

**Type checking with MyPy:**
```bash
mypy deepseek_ocr_mac.py --ignore-missing-imports
```

### Continuous Integration

The project uses GitHub Actions for automated testing and quality checks on every push and pull request.

**CI Pipeline includes:**
- ✅ Code formatting checks (Black)
- ✅ Linting (Ruff)
- ✅ Type checking (MyPy)
- ✅ Unit tests (Python 3.9-3.12)
- ✅ Integration tests (Ubuntu & macOS)
- ✅ Coverage reporting (Codecov)
- ✅ Security scanning (Bandit, Safety)
- ✅ CLI integration tests

**View CI status:**
Check the Actions tab on GitHub after pushing changes.

### Manual Testing

```bash
# Test with sample image
./deepseek_ocr_mac.py test_image.png

# Test with sample PDF
./deepseek_ocr_mac.py test_document.pdf

# Verify MPS acceleration
python3 -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"

# Create test fixtures
python3 -c "
from tests.fixtures import create_test_image, create_test_pdf
from pathlib import Path
create_test_image(Path('test.png'), 'Sample Text')
create_test_pdf(Path('test.pdf'), num_pages=3)
"
```

## Future Enhancements

Potential improvements (not yet implemented):

- [ ] Parallel page processing for multi-core utilization
- [ ] Additional output formats (HTML, JSON)
- [ ] Streaming progress feedback
- [ ] Homebrew tap for `brew install`
- [ ] LangChain integration
- [ ] Batch processing optimizations

## References

- [DeepSeek-OCR Official Repo](https://github.com/deepseek-ai/DeepSeek-OCR)
- [Hugging Face Model Hub](https://huggingface.co/deepseek-ai/DeepSeek-OCR)
- [MPS Backend Support Discussion](https://huggingface.co/deepseek-ai/DeepSeek-OCR/discussions/20) - Critical patches for Apple Silicon
- [PyTorch MPS Documentation](https://pytorch.org/docs/stable/notes/mps.html)
- [PyMuPDF Documentation](https://pymupdf.readthedocs.io)

## License

This CLI tool is provided as-is for use with the DeepSeek-OCR model. Please refer to the [DeepSeek-OCR license](https://github.com/deepseek-ai/DeepSeek-OCR) for model usage terms.

## Contributing

Contributions welcome! Please open an issue or PR for:

- Bug fixes
- Performance improvements
- New features
- Documentation improvements

## Acknowledgments

- DeepSeek AI for the excellent OCR model
- PyMuPDF team for PDF rendering capabilities
- Hugging Face for model hosting infrastructure
