# PRD: DeepSeek-OCR Mac CLI

**Version:** 1.0
**Owner:** Benedict
**Date:** 2025-10-24
**Goal:** Enable simple local OCR for images and PDFs using DeepSeek-OCR on macOS (Apple Silicon) — no CUDA or external binaries, fully offline.

---

## 1. Objective

Provide an offline, Mac-native OCR CLI that leverages DeepSeek-OCR for high-quality text and layout extraction (Markdown format).

The solution should:
- Run on Apple Silicon (M1–M4) using PyTorch + MPS acceleration
- Support both images and PDFs (auto-conversion)
- Output Markdown-structured text (with page separation)
- Be minimal, local, and easily extensible

---

## 2. Problem Statement

The DeepSeek-OCR model is optimized for Linux + CUDA (FlashAttention/vLLM).

macOS users face hurdles:
- Flash-Attention fails to compile on ARM
- CUDA dependencies break pip install
- PDF conversion often requires external binaries (Poppler)

**Goal:** Offer a "one-file" Mac CLI that hides all these complexities and just works.

---

## 3. Key Requirements

| Category | Requirement |
|----------|-------------|
| **Compatibility** | macOS 13+ (M1–M4) |
| **Acceleration** | PyTorch MPS (Metal backend) |
| **Inputs** | .pdf, .png, .jpg, .jpeg, .webp, .bmp, .tif, .tiff |
| **Outputs** | Markdown (`outputs/merged_output.md`), per-page text files |
| **Performance** | 1–3 s/page on M3 Pro @ 640 px |
| **Offline** | No internet required after model is cached |
| **Usability** | One-line CLI command: `deepseek-ocr mydoc.pdf` |

---

## 4. Technical Overview

### 4.1 Architecture

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
- **Device:** "mps" if available, else CPU
- **Core dependencies:** torch, transformers, pillow, pymupdf

---

## 5. Installation Guide

### 5.1 Create environment

```bash
mkdir ~/deepseek-ocr-mac && cd ~/deepseek-ocr-mac
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### 5.2 Install dependencies

```bash
pip install torch torchvision torchaudio
pip install "transformers>=4.46" einops addict easydict pillow tqdm
pip install pymupdf
```

No CUDA or flash-attn required.

---

## 6. Usage Examples

### Single image

```bash
./deepseek_ocr_mac.py sample.png
```

### PDF file

```bash
./deepseek_ocr_mac.py document.pdf
```

### Directory with multiple PDFs/images

```bash
./deepseek_ocr_mac.py ./scans
```

### Adjust quality or performance

```bash
./deepseek_ocr_mac.py mypaper.pdf --dpi 360 --base-size 1280 --image-size 768
```

### Disable cropping or compression

```bash
./deepseek_ocr_mac.py file.pdf --no-crop --no-compress
```

---

## 7. Output

```
outputs/
 ├── merged_output.md
 ├── page-0001.png.text
 ├── page-0002.png.text
 └── …
```

- `merged_output.md` — combined Markdown output
- `outputs/*.png.text` — per-page OCR result files (from model.infer)
- Images are auto-deleted if temporary (PDF renderings)

---

## 8. Future Enhancements (Optional)

| Feature | Description |
|---------|-------------|
| **Parallel page processing** | Run OCR across multiple cores on M-series (safe parallel batching) |
| **Output formats** | `--format html/json` |
| **Whisper-style streaming** | Integrate progress feedback |
| **Homebrew tap** | `brew install deepseek-ocr-mac` |
| **LangChain tool integration** | Expose as a local OCR tool in your agentic workflows |

---

## 9. Validation Checklist

- [x] Runs on macOS (M1–M4)
- [x] Works without CUDA or FlashAttention
- [x] Accepts both PDFs and image folders
- [x] Outputs Markdown to outputs/merged_output.md
- [x] Uses MPS acceleration automatically
- [x] Fully offline after model download

---

## 10. References

- [DeepSeek-OCR official repo](https://github.com/deepseek-ai/DeepSeek-OCR) (2025-10 release)
- [Hugging Face model hub](https://huggingface.co/deepseek-ai/DeepSeek-OCR)
- [PyTorch MPS docs](https://pytorch.org/docs/stable/notes/mps.html)
- [PyMuPDF docs](https://pymupdf.readthedocs.io)
