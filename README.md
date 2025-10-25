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

### Command-Line Options

```
usage: deepseek_ocr_mac.py [-h] [-o OUT] [--model MODEL] [--base-size BASE_SIZE]
                           [--image-size IMAGE_SIZE] [--no-crop] [--no-compress]
                           [--dpi DPI]
                           path

positional arguments:
  path                  File or folder (image/pdf or dir)

options:
  -h, --help           Show this help message
  -o, --out OUT        Output directory (default: outputs)
  --model MODEL        Model name (default: deepseek-ai/DeepSeek-OCR)
  --base-size SIZE     Base resolution (default: 1024)
  --image-size SIZE    Target image size (default: 640)
  --no-crop           Disable intelligent cropping
  --no-compress       Disable compression
  --dpi DPI           DPI for PDF rendering (default: 288)
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

## Output

Results are saved to the `outputs/` directory (configurable with `-o`):

```
outputs/
├── merged_output.md          # Combined Markdown output
├── sample-p0001.png.text     # Individual page results
├── sample-p0002.png.text
└── ...
```

- `merged_output.md` - All pages combined with headers
- `*.png.text` - Per-page OCR results

## Performance

Typical performance on Apple Silicon (M3 Pro):

| Resolution | Pages/min | Quality |
|-----------|-----------|---------|
| 640px     | 20-30     | Standard |
| 1024px    | 10-15     | High |
| 1280px    | 5-10      | Very High |

Performance scales with DPI and resolution settings. Higher values = better quality but slower processing.

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
