# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DeepSeek-OCR Mac CLI is an offline OCR tool for images and PDFs using the DeepSeek-OCR model on macOS (Apple Silicon). The project consists of a single-file CLI script (`deepseek_ocr_mac.py`) that leverages PyTorch MPS acceleration for native Apple Silicon performance.

## Architecture

**Single-File CLI Design**: The entire application logic is contained in `deepseek_ocr_mac.py` (~255 lines). This architectural choice prioritizes simplicity and portability.

**Processing Pipeline**:
1. Input validation and file collection (`collect_targets`)
2. PDF-to-image conversion using PyMuPDF (`render_pdf_to_images`)
3. OCR inference via DeepSeek-OCR model (`run_infer`)
4. Markdown output generation (per-page `.text` files + `merged_output.md`)

**MPS Compatibility Layer**: Critical patches for Apple Silicon support are embedded in the model loading logic (lines 182-201):
- Uses `attn_implementation="eager"` instead of Flash Attention (MPS incompatible)
- Converts model to `float32` (bfloat16 causes dtype mismatches on MPS)
- Sets `tok.padding_side = "right"` to avoid MPS generation bugs
- Device detection: `mps` if available, else `cpu`

**Dependency Versions**: `transformers==4.46.3` and `tokenizers==0.20.3` are pinned for MPS stability. Do not upgrade without testing.

## Extension Features (New in v2.0)

The project now includes comprehensive post-processing extensions that add 11 new CLI flags and ~550 lines of functionality. All extensions are **opt-in** via command-line flags and fully backward compatible.

### Feature Categories

**Data Extraction Extensions:**
- `--emit-csv` / `--emit-csv=tsv` - Extract Markdown tables to CSV/TSV files
- `--math-latex` - Export LaTeX equations to individual `.tex` files
- `--chart-to-csv` - Extract chart data from code blocks (experimental)
- `--emit-boxes` / `--overlay` - Generate bounding box JSON and visual overlays

**Content Enhancement:**
- `--code-lang` - Auto-detect and tag code blocks with language identifiers
- `--emit-chunks` - Generate RAG-ready JSONL chunks (1200 chars, with metadata)

**Performance & Quality:**
- `--compression {low,med,high}` - Preset quality/speed profiles
- `--strict` / `--min-words N` - Quality gates for CI/CD (exit code 1 on failure)
- `--workers N` - Parallel processing support (framework added, not yet implemented)

### Post-Processing Pipeline Architecture

All extensions run **after** OCR inference completes, making them MPS-safe (CPU-only operations):

```
OCR Inference (MPS/GPU)
    → Word Count & Quality Check
    → Table Extraction (regex + CSV writer)
    → Math Extraction (regex + file I/O)
    → Code Language Tagging (pattern matching + replacement)
    → Chart Data Extraction (regex + CSV writer)
    → RAG Chunks (text chunking + JSONL)
    → Bounding Boxes (PIL image processing + JSON)
    → Quality Summary (YAML frontmatter)
```

### Key Implementation Files

**Post-Processing Functions** (`deepseek_ocr_mac.py:97-429`):
- `build_prompt()` - Generates prompts with optional extras
- `word_count()` - Regex-based word counting
- `md_table_to_rows()` + `extract_tables_to_csv()` - Table extraction
- `extract_math()` - LaTeX equation extraction
- `detect_code_language()` + `tag_code_languages()` - 10-language detection
- `extract_chart_data()` - Chart data parsing
- `emit_chunks()` - RAG chunk generation
- `extract_bounding_boxes_heuristic()` + `create_overlay()` - Visual analysis

**Processing Stats** (`deepseek_ocr_mac.py:56-78`):
- `ProcessingStats` dataclass tracks success_rate, processing_time, total_words

**Main Integration** (`deepseek_ocr_mac.py:686-783`):
- Processing loop with conditional post-processing
- Quality summary generation with YAML frontmatter
- Strict mode with exit code handling

### Testing

**Unit Tests:** `tests/test_extensions.py` contains 44 tests covering all post-processing functions:
- ProcessingStats calculations (5 tests)
- Table extraction (6 tests)
- Math extraction (4 tests)
- Code language detection (8 tests)
- Chart extraction (3 tests)
- RAG chunks (3 tests)
- Bounding boxes (4 tests)
- Full pipeline integration (1 test)

**Running Extension Tests:**
```bash
pytest tests/test_extensions.py -v  # All extension tests
pytest tests/ -v                     # Full suite (66 tests)
```

### Output Structure

With all extensions enabled, the output directory structure is:

```
outputs/
├── merged_output.md          # Markdown + YAML quality summary
├── chunks.jsonl              # RAG chunks
├── tables/                   # CSV/TSV tables
│   └── page_NNNN_table_N.csv
├── equations/                # LaTeX equations
│   └── page_NNNN_eq_NNN.tex
├── charts/                   # Chart data
│   └── page_NNNN_chart_N.csv
├── boxes/                    # Bounding box JSON
│   └── page_NNNN.json
└── overlays/                 # Visual overlays
    └── page_NNNN_overlay.png
```

## Common Development Commands

### Setup and Installation
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Development dependencies (includes testing tools)
pip install -r requirements-dev.txt

# Make executable
chmod +x deepseek_ocr_mac.py
```

### Running the CLI
```bash
# Basic usage
./deepseek_ocr_mac.py <image.png|document.pdf|directory/>

# High-quality OCR
./deepseek_ocr_mac.py file.pdf --dpi 360 --base-size 1280 --image-size 768

# Custom output directory
./deepseek_ocr_mac.py file.pdf -o results/
```

### Testing
```bash
# Run all tests
pytest

# Run specific test suites
pytest tests/test_unit.py -v       # Unit tests only
pytest tests/test_integration.py -v # Integration tests only

# Run with coverage
pytest --cov=. --cov-report=html --cov-report=term

# Run by marker
pytest -m unit        # Unit tests
pytest -m integration # Integration tests
```

**Test Strategy**: Tests use mocked models to avoid downloading the 8GB DeepSeek-OCR model. See `tests/fixtures.py` for test image/PDF generation utilities.

### Code Quality
```bash
# Format code (line length: 100)
black .

# Check formatting
black --check --diff .

# Lint
ruff check .
ruff check --fix .  # Auto-fix issues

# Type checking
mypy deepseek_ocr_mac.py --ignore-missing-imports
```

### Manual Testing

**Extension Features Testing:**
```bash
# Test all extensions on a single document
./deepseek_ocr_mac.py test.pdf \
  --emit-csv \
  --math-latex \
  --code-lang \
  --emit-chunks \
  --emit-boxes --overlay \
  --strict --min-words 10

# Test compression presets
./deepseek_ocr_mac.py test.pdf --compression high  # Fast
./deepseek_ocr_mac.py test.pdf --compression low   # Quality

# Test quality gate (should fail if < 1000 words/page)
./deepseek_ocr_mac.py test.pdf --strict --min-words 1000
echo $?  # Check exit code (1 = failed quality check)

# Verify output structure
ls -R outputs/
cat outputs/chunks.jsonl | head -5
cat outputs/tables/page_0001_table_1.csv
cat outputs/equations/page_0001_eq_001.tex
```

**MPS Verification:**
```bash
# Verify MPS acceleration
python3 -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"

# Create test fixtures
python3 -c "
from tests.fixtures import create_test_image, create_test_pdf
from pathlib import Path
create_test_image(Path('test.png'), 'Sample Text')
create_test_pdf(Path('test.pdf'), num_pages=3)
"

# Test with fixtures
./deepseek_ocr_mac.py test.png
./deepseek_ocr_mac.py test.pdf
```

## Key Implementation Details

### MPS Scatter Operation Fix
The most critical MPS patch is the scatter operation fix. Original DeepSeek-OCR uses `masked_scatter_` which causes infinite generation loops on MPS. This was fixed upstream but requires specific transformers versions. If you see repetitive output ("Background Background..."), verify:
1. `transformers==4.46.3` is installed
2. Model loads with `attn_implementation="eager"`
3. Model is converted to `float32` for MPS

### PDF Rendering
PDFs are converted to PNGs via PyMuPDF with DPI-based scaling:
- Default DPI: 288 (balance of quality/speed)
- Scale factor: `max(1.0, dpi / 72.0)`
- Rendered images saved to temporary directory (cleaned up on exit)

### Output Format
```
outputs/
├── merged_output.md          # All pages concatenated with "# filename" headers
├── sample-p0001.png.text     # Per-page OCR results (raw markdown)
└── sample-p0002.png.text
```

### Error Handling
- Missing PyMuPDF: Exit code 2
- Invalid path: Exit code 1
- Model loading failure: Exit code 5
- No input files: Exit code 4
- General errors: Exit code 6
- Keyboard interrupt: Exit code 130

## Git Workflow

**Feature Branch Strategy**: This repository follows a feature branch workflow. When starting new features:

1. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/feature-name
   ```

2. **Make commits** with descriptive messages:
   ```bash
   git commit -m "Add feature X with Y capability

   - Implementation details
   - Testing coverage
   - Documentation updates

   🤖 Generated with [Claude Code](https://claude.com/claude-code)

   Co-Authored-By: Claude <noreply@anthropic.com>"
   ```

3. **Push and create Pull Request**:
   ```bash
   git push -u origin feature/feature-name
   gh pr create --title "Feature: Description" --body "Details..."
   ```

4. **Wait for CI/CD checks** to pass before merging

**Branch Naming Convention**:
- Features: `feature/descriptive-name`
- Bug fixes: `fix/issue-description`
- Documentation: `docs/what-changed`

**Never commit directly to `main`**. Always use feature branches and pull requests for code review and CI/CD validation.

## CI/CD Pipeline

GitHub Actions runs on every push and PR:
- **Code Quality**: Black formatting, Ruff linting, MyPy type checking
- **Tests**: Unit and integration tests across Python 3.9-3.12 on Ubuntu/macOS
- **Security**: Bandit security scanning, Safety dependency checks
- **CLI Integration**: Executable permissions, help text, MPS availability check

## Important Constraints

1. **Trust Remote Code**: `trust_remote_code=True` is required for DeepSeek-OCR (custom model architecture). This is safe for the official `deepseek-ai/DeepSeek-OCR` repository but flagged by security scanners (Bandit B615). Do not remove `# nosec B615` comments.

2. **MPS Precision**: Do not use bfloat16 on MPS. Stick to float32 for numerical stability.

3. **Transformers Version**: Do not upgrade transformers beyond 4.46.3 without extensive MPS testing.

4. **Single File Design**: Keep all application logic in `deepseek_ocr_mac.py`. Avoid creating modules unless absolutely necessary.

## Troubleshooting Model Issues

If model downloads fail or OCR produces garbage output:
```bash
# Check Hugging Face cache
ls ~/.cache/huggingface/hub/

# Force re-download (delete cache)
rm -rf ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-OCR

# Verify transformers version
pip show transformers tokenizers
```

## References

Critical external resources:
- [MPS Compatibility Discussion](https://huggingface.co/deepseek-ai/DeepSeek-OCR/discussions/20) - Source of MPS patches
- [DeepSeek-OCR Official Repo](https://github.com/deepseek-ai/DeepSeek-OCR)
- [PyTorch MPS Backend Docs](https://pytorch.org/docs/stable/notes/mps.html)
