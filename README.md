# DeepSeek-OCR Mac CLI

[![CI/CD Pipeline](https://github.com/benedict2310/DeepSeekOCR-Cli/actions/workflows/ci.yml/badge.svg)](https://github.com/benedict2310/DeepSeekOCR-Cli/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Offline OCR and hybrid search for images and PDFs on macOS (Apple Silicon).

## Features

**🚀 High-Quality OCR**
- Runs natively on Apple Silicon (M1-M4) using PyTorch MPS acceleration
- Supports images and PDFs with automatic conversion
- Outputs clean Markdown-structured text
- Fully offline after initial model download (~8GB)

**🔍 Hybrid Search** ⭐ NEW
- **Visual search**: Find images by visual similarity using CLIP embeddings
- **Semantic search**: Search text by meaning using sentence transformers
- **Web UI**: Interactive demo at http://localhost:8000
- **Concurrent-safe**: File locking prevents index corruption
- See [HYBRID_SEARCH.md](HYBRID_SEARCH.md) for full documentation

**📦 Post-Processing Extensions**
- Extract tables to CSV/TSV
- Export LaTeX equations
- Auto-tag code blocks with languages
- Generate RAG-ready chunks
- Extract bounding boxes with visual overlays
- Quality gates for CI/CD workflows

## Quick Start

### Installation

```bash
# Clone and setup
git clone https://github.com/benedict2310/DeepSeekOCR-Cli.git
cd DeepSeekOCR-Cli
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
chmod +x deepseek_ocr_mac.py
```

### Basic OCR

```bash
# Process a single image
./deepseek_ocr_mac.py sample.png

# Process a PDF
./deepseek_ocr_mac.py document.pdf

# Process a directory
./deepseek_ocr_mac.py ./scans

# High-quality OCR
./deepseek_ocr_mac.py paper.pdf --dpi 360
```

### Hybrid Search

```bash
# 1. Build indexes while processing
./deepseek_ocr_mac.py docs/*.pdf \
  --update-index \
  --visual-index ./vi_index \
  --text-index ./ti_index

# 2. Launch web demo
uvicorn app:app --reload --port 8000

# 3. Open http://localhost:8000 in your browser
```

**Search by image similarity:**
```bash
curl -X POST -F "file=@screenshot.png" -F "topk=5" http://localhost:8000/search_image
```

**Search by text:**
```bash
curl -X POST -F "q=coding agents" -F "topk=5" http://localhost:8000/search_text
```

## Common Options

| Option | Description | Example |
|--------|-------------|---------|
| `-o DIR` | Output directory | `-o results` |
| `--dpi N` | PDF rendering DPI (default: 288) | `--dpi 360` |
| `--base-size N` | Vision model base size (default: 1008) | `--base-size 1280` |
| `--image-size N` | Vision model image patches (default: 768) | `--image-size 1024` |
| `--compression LEVEL` | Speed/quality preset (low/med/high) | `--compression high` |
| `--emit-csv [tsv]` | Extract tables to CSV/TSV | `--emit-csv tsv` |
| `--math-latex` | Export LaTeX equations | `--math-latex` |
| `--code-lang` | Auto-detect code languages | `--code-lang` |
| `--emit-chunks` | Generate RAG chunks (JSONL) | `--emit-chunks` |
| `--emit-boxes` | Extract bounding boxes | `--emit-boxes` |
| `--overlay` | Create visual overlays | `--overlay` |
| `--strict` | Exit non-zero on quality failures | `--strict --min-words 50` |

Run `./deepseek_ocr_mac.py --help` for all options.

## Output Structure

```
outputs/
├── merged_output.md          # All pages in one Markdown file
├── sample-p0001.png.text     # Per-page text files
├── sample-p0002.png.text
├── chunks.jsonl              # RAG chunks (--emit-chunks)
├── tables/                   # Extracted tables (--emit-csv)
│   └── page_0001_table_1.csv
├── equations/                # LaTeX equations (--math-latex)
│   └── page_0001_eq_001.tex
└── boxes/                    # Bounding boxes (--emit-boxes)
    └── page_0001.json
```

## Extension Features

The CLI includes 11 post-processing flags that add ~550 lines of functionality. All features are **opt-in** and fully backward compatible.

**Data Extraction:**
- `--emit-csv` / `--emit-csv=tsv` - Extract Markdown tables to CSV/TSV
- `--math-latex` - Export LaTeX equations to individual `.tex` files
- `--chart-to-csv` - Extract chart data from code blocks (experimental)
- `--emit-boxes` / `--overlay` - Bounding box JSON and visual overlays

**Content Enhancement:**
- `--code-lang` - Auto-detect and tag 10+ programming languages
- `--emit-chunks` - RAG-ready JSONL (1200 chars, with metadata)

**Quality & Performance:**
- `--compression {low,med,high}` - Preset quality/speed profiles
- `--strict` / `--min-words N` - Quality gates (exit code 1 on failure)
- `--workers N` - Parallel processing framework (coming soon)

See test files in `tests/test_extensions.py` for usage examples.

## Hybrid Search Architecture

**Visual Index:**
- CLIP-ViT-B-32 embeddings (OpenAI's vision-language model)
- HNSW approximate nearest neighbor search
- Finds visually similar images (layout, structure, content)

**Text Index:**
- all-MiniLM-L6-v2 sentence embeddings
- HNSW for fast semantic search
- Understands meaning, not just keywords

**Hybrid Fusion:**
- Combines visual + text scores with weighted fusion
- File locking for concurrent processing safety
- Sub-second queries even with thousands of pages

See [HYBRID_SEARCH.md](HYBRID_SEARCH.md) for complete documentation, API reference, and troubleshooting.

## Requirements

- macOS 13+ (Ventura or later)
- Python 3.9+
- Apple Silicon (M1, M2, M3, M4) recommended
- ~10GB disk space for model cache

## Supported Formats

**Images:** PNG, JPG, JPEG, WebP, BMP, TIF, TIFF
**PDFs:** Automatically converted to images for processing

## Testing

```bash
# Run all tests
pytest

# Run specific suites
pytest tests/test_unit.py          # Unit tests
pytest tests/test_extensions.py    # Extension features
pytest tests/test_hybrid_search.py # Hybrid search
pytest tests/test_integration.py   # Integration tests

# With coverage
pytest --cov=. --cov-report=html
```

## Development

```bash
# Format code
black .

# Lint
ruff check .
ruff check --fix .

# Type checking
mypy deepseek_ocr_mac.py --ignore-missing-imports
```

## Troubleshooting

**MPS acceleration issues:**
```bash
# Verify MPS is available
python3 -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

**Model download fails:**
```bash
# Clear cache and retry
rm -rf ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-OCR
```

**PDF rendering requires PyMuPDF:**
```bash
pip install pymupdf
```

## Architecture Notes

This is a **single-file CLI design** (`deepseek_ocr_mac.py`, ~900 lines) with hybrid search modules (`visual_index.py`, `hybrid_search.py`, `app.py`).

**MPS Compatibility:**
- Uses `attn_implementation="eager"` (Flash Attention incompatible with MPS)
- Converts to `float32` (bfloat16 causes dtype mismatches)
- Sets `padding_side="right"` (avoids MPS generation bugs)
- Requires `transformers==4.46.3` and `tokenizers==0.20.3`

**Security:**
- `trust_remote_code=True` required for DeepSeek-OCR custom architecture
- Safe for official `deepseek-ai/DeepSeek-OCR` repository
- Flagged by Bandit B615 (expected, with `# nosec` comments)

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- DeepSeek-OCR model by [DeepSeek AI](https://github.com/deepseek-ai/DeepSeek-OCR)
- CLIP embeddings by [OpenAI](https://github.com/openai/CLIP)
- Sentence transformers by [UKPLab](https://www.sbert.net/)
