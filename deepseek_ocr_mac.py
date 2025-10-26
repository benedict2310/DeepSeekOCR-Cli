#!/usr/bin/env python3
"""
DeepSeek-OCR Mac CLI
Offline OCR for PDFs and images using DeepSeek-OCR on macOS (Apple Silicon)
"""
import argparse
import csv
import gc
import json
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image, ImageDraw, ImageFilter
from transformers import AutoModel, AutoTokenizer

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

# Compression presets for different quality/speed trade-offs
# NOTE: base_size values above 1024 can trigger shape mismatch bugs in DeepSeek-OCR model
COMPRESSION_PRESETS = {
    "low": dict(base_size=1024, image_size=640, crop_mode=True, test_compress=False),  # Baseline quality
    "med": dict(base_size=1024, image_size=640, crop_mode=True, test_compress=True),   # Balanced
    "high": dict(base_size=896, image_size=512, crop_mode=True, test_compress=True),  # Fast/compressed
}

# Regular expressions for post-processing
TABLE_RE = re.compile(r"(?:^\|.*\|\s*$\n)+", re.MULTILINE)
MATH_RE = re.compile(r"(\$\$.*?\$\$|\$[^$\n]+\$)", re.DOTALL)
CODE_FENCE_RE = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)

# Code language detection heuristics
CODE_LANG_PATTERNS = {
    "python": [r"\bdef\b", r"\bclass\b", r"\bimport\b", r"\bfrom\b.*\bimport\b"],
    "javascript": [r"\bfunction\b", r"\bconst\b", r"\blet\b", r"\bvar\b", r"=>"],
    "typescript": [r"\binterface\b", r"\btype\b.*=", r":\s*\w+\s*="],
    "java": [r"\bpublic\s+class\b", r"\bprivate\b", r"\bprotected\b"],
    "cpp": [r"#include\b", r"\bstd::", r"\bnamespace\b"],
    "go": [r"\bfunc\b", r"\bpackage\b", r"import\s*\("],
    "rust": [r"\bfn\b", r"\blet\s+mut\b", r"\bimpl\b", r"\buse\b"],
    "sql": [r"\bSELECT\b", r"\bFROM\b", r"\bWHERE\b", r"\bJOIN\b"],
    "bash": [r"#!/bin/bash", r"\bif\s*\[", r"\becho\b", r"\bexport\b"],
    "json": [r"^\s*\{", r"^\s*\[", r'"\w+":\s*["\[\{]'],
}


@dataclass
class ProcessingStats:
    """Statistics for a processing run."""

    total_pages: int = 0
    failed_pages: List[int] = field(default_factory=list)
    total_words: int = 0
    compression: str = "low"
    workers: int = 1
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_pages == 0:
            return 0.0
        return (self.total_pages - len(self.failed_pages)) / self.total_pages * 100

    @property
    def processing_time(self) -> float:
        """Calculate total processing time in seconds."""
        return self.end_time - self.start_time


def is_image(path: Path):
    """Check if file is a supported image format."""
    return path.suffix.lower() in IMG_EXTS


def is_pdf(path: Path):
    """Check if file is a PDF."""
    return path.suffix.lower() == ".pdf"


def ensure_dir(p: Path):
    """Create directory if it doesn't exist."""
    p.mkdir(parents=True, exist_ok=True)
    return p


def build_prompt(mode: str = "markdown", extras: Optional[List[str]] = None) -> str:
    """
    Build a prompt for the OCR model with optional extras.

    Args:
        mode: Base mode (currently only "markdown" supported)
        extras: Additional prompt instructions

    Returns:
        Complete prompt string
    """
    base = "<image>\n"
    if mode == "markdown":
        base += "<|grounding|>Convert the document to markdown."
    if extras:
        base += "\n" + "\n".join(extras)
    return base


def word_count(text: str) -> int:
    """Count words in text."""
    return len(re.findall(r"\w+", text))


def md_table_to_rows(md_table: str) -> List[List[str]]:
    """
    Convert a Markdown table to a list of row lists.

    Args:
        md_table: Markdown table string

    Returns:
        List of rows, each row is a list of cells
    """
    lines = [line.strip() for line in md_table.strip().splitlines()]
    # Drop alignment line (---|:---) at index 1 if present
    if len(lines) > 1 and set(lines[1].replace("|", "").strip()) <= set("-: "):
        lines.pop(1)
    rows = [[cell.strip() for cell in line.strip("|").split("|")] for line in lines]
    return rows


def extract_tables_to_csv(
    page_text: str, out_dir: Path, page_idx: int, fmt: str = "csv"
) -> List[Path]:
    """
    Extract Markdown tables to CSV/TSV files.

    Args:
        page_text: Text content of the page
        out_dir: Output directory
        page_idx: Page index (1-based)
        fmt: Format ("csv" or "tsv")

    Returns:
        List of created file paths
    """
    dialect = "excel" if fmt == "csv" else "excel-tab"
    created_files = []

    for k, match in enumerate(TABLE_RE.finditer(page_text), start=1):
        rows = md_table_to_rows(match.group())
        file_path = out_dir / "tables" / f"page_{page_idx:04d}_table_{k}.{fmt}"
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with file_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, dialect=dialect)
            for row in rows:
                writer.writerow(row)

        created_files.append(file_path)

    return created_files


def extract_math(page_text: str, out_dir: Path, page_idx: int) -> List[Path]:
    """
    Extract LaTeX math expressions to .tex files.

    Args:
        page_text: Text content of the page
        out_dir: Output directory
        page_idx: Page index (1-based)

    Returns:
        List of created file paths
    """
    eq_dir = out_dir / "equations"
    eq_dir.mkdir(exist_ok=True, parents=True)
    created_files = []

    for k, match in enumerate(MATH_RE.finditer(page_text), start=1):
        expr = match.group().strip("$")
        file_path = eq_dir / f"page_{page_idx:04d}_eq_{k:03d}.tex"
        file_path.write_text(expr, encoding="utf-8")
        created_files.append(file_path)

    return created_files


def detect_code_language(code: str) -> Optional[str]:
    """
    Detect programming language from code snippet using heuristics.

    Args:
        code: Code snippet

    Returns:
        Language name or None if uncertain
    """
    # Count matches for each language
    scores = {}
    for lang, patterns in CODE_LANG_PATTERNS.items():
        score = sum(1 for pattern in patterns if re.search(pattern, code, re.IGNORECASE))
        if score > 0:
            scores[lang] = score

    if not scores:
        return None

    # Return language with highest score, but only if confident (score >= 2)
    best_lang = max(scores.items(), key=lambda x: x[1])
    return best_lang[0] if best_lang[1] >= 2 else None


def tag_code_languages(page_text: str) -> str:
    """
    Tag code blocks with detected languages.

    Args:
        page_text: Text content with code blocks

    Returns:
        Text with tagged code blocks
    """

    def replace_fence(match):
        lang = match.group(1)
        code = match.group(2)

        # If already tagged, leave it
        if lang:
            return match.group(0)

        # Try to detect language
        detected = detect_code_language(code)
        if detected:
            return f"```{detected}\n{code}```"
        return match.group(0)

    return CODE_FENCE_RE.sub(replace_fence, page_text)


def extract_chart_data(page_text: str, out_dir: Path, page_idx: int) -> List[Path]:
    """
    (Experimental) Extract chart data to CSV files.

    Args:
        page_text: Text content of the page
        out_dir: Output directory
        page_idx: Page index (1-based)

    Returns:
        List of created file paths
    """
    chart_dir = out_dir / "charts"
    chart_dir.mkdir(exist_ok=True, parents=True)
    created_files = []

    # Look for fenced blocks labeled as chart-data or csv
    chart_pattern = re.compile(r"```(?:chart-data|csv)\n(.*?)```", re.DOTALL)

    for k, match in enumerate(chart_pattern.finditer(page_text), start=1):
        data = match.group(1)
        # Try to parse as CSV-like content
        lines = [line.strip() for line in data.splitlines() if line.strip()]

        if not lines:
            continue

        file_path = chart_dir / f"page_{page_idx:04d}_chart_{k}.csv"

        with file_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for line in lines:
                # Split by comma or tab
                if "," in line:
                    writer.writerow([cell.strip() for cell in line.split(",")])
                elif "\t" in line:
                    writer.writerow([cell.strip() for cell in line.split("\t")])

        if file_path.stat().st_size > 0:
            created_files.append(file_path)

    return created_files


def emit_chunks(chunks_path: Path, page_idx: int, page_text: str) -> None:
    """
    Emit text chunks for RAG indexing.

    Args:
        chunks_path: Path to chunks JSONL file
        page_idx: Page index (1-based)
        page_text: Text content of the page
    """
    chunk_size = 1200
    items = []

    for i in range(0, len(page_text), chunk_size):
        items.append(
            {
                "page": page_idx,
                "start": i,
                "end": min(len(page_text), i + chunk_size),
                "text": page_text[i : i + chunk_size],
            }
        )

    with chunks_path.open("a", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def extract_bounding_boxes_heuristic(
    image_path: Path, out_dir: Path, page_idx: int
) -> Optional[Path]:
    """
    Extract approximate bounding boxes using image processing heuristics.

    Args:
        image_path: Path to image file
        out_dir: Output directory
        page_idx: Page index (1-based)

    Returns:
        Path to generated JSON file or None on error
    """
    try:
        img = Image.open(image_path)
        gray = img.convert("L")

        # Simple edge detection
        edges = gray.filter(ImageFilter.FIND_EDGES)

        # Convert to binary
        threshold = 50
        binary = edges.point(lambda x: 255 if x > threshold else 0)

        # Find contiguous regions (simplified - just divide into grid)
        width, height = binary.size
        boxes = []

        # Divide into approximate text regions (grid-based heuristic)
        grid_rows, grid_cols = 10, 5
        cell_h = height // grid_rows
        cell_w = width // grid_cols

        for row in range(grid_rows):
            for col in range(grid_cols):
                x = col * cell_w
                y = row * cell_h
                region = binary.crop((x, y, x + cell_w, y + cell_h))

                # Check if region has content (non-white pixels)
                pixels = list(region.getdata())
                if sum(pixels) / len(pixels) < 250:  # Has some dark pixels
                    boxes.append(
                        {
                            "x": x,
                            "y": y,
                            "w": cell_w,
                            "h": cell_h,
                            "hint": "text-block",
                        }
                    )

        # Write to JSON
        boxes_dir = out_dir / "boxes"
        boxes_dir.mkdir(exist_ok=True, parents=True)
        json_path = boxes_dir / f"page_{page_idx:04d}.json"

        data = {"page": page_idx, "bbox_provider": "heuristic", "boxes": boxes}

        json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return json_path

    except Exception as e:
        print(f"Warning: Could not extract bounding boxes: {e}")
        return None


def create_overlay(
    image_path: Path, boxes_json_path: Path, out_dir: Path, page_idx: int
) -> Optional[Path]:
    """
    Create an overlay image with bounding boxes drawn.

    Args:
        image_path: Path to source image
        boxes_json_path: Path to JSON file with bounding boxes
        out_dir: Output directory
        page_idx: Page index (1-based)

    Returns:
        Path to generated overlay image or None on error
    """
    try:
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        # Load boxes
        data = json.loads(boxes_json_path.read_text(encoding="utf-8"))
        boxes = data.get("boxes", [])

        # Draw rectangles
        for box in boxes:
            x, y, w, h = box["x"], box["y"], box["w"], box["h"]
            draw.rectangle([x, y, x + w, y + h], outline="red", width=2)

        # Save overlay
        overlay_dir = out_dir / "overlays"
        overlay_dir.mkdir(exist_ok=True, parents=True)
        overlay_path = overlay_dir / f"page_{page_idx:04d}_overlay.png"

        img.save(overlay_path)
        return overlay_path

    except Exception as e:
        print(f"Warning: Could not create overlay: {e}")
        return None


def render_pdf_to_images(pdf_path: Path, out_dir: Path, scale: float):
    """
    Convert PDF pages to PNG images using PyMuPDF.

    Args:
        pdf_path: Path to PDF file
        out_dir: Directory to save rendered images
        scale: Scaling factor (derived from DPI)

    Returns:
        List of rendered image paths
    """
    if fitz is None:
        print("ERROR: PyMuPDF not installed. Run: pip install pymupdf")
        sys.exit(2)

    doc = fitz.open(pdf_path)
    out_files = []

    for i, page in enumerate(doc, start=1):
        mat = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        out_file = out_dir / f"{pdf_path.stem}-p{i:04d}.png"
        pix.save(out_file.as_posix())
        out_files.append(out_file)

    doc.close()
    return out_files


def run_infer(
    model,
    tok,
    image_path,
    out_dir,
    base_size,
    image_size,
    crop_mode,
    test_compress,
    prompt_extras=None,
):
    """
    Run OCR inference on a single image.

    Args:
        model: DeepSeek-OCR model
        tok: Tokenizer
        image_path: Path to image file
        out_dir: Output directory for results
        base_size: Base resolution for image processing
        image_size: Target image size
        crop_mode: Whether to enable intelligent cropping
        test_compress: Whether to use compression
        prompt_extras: Optional list of additional prompt instructions

    Returns:
        Extracted markdown text
    """
    prompt = build_prompt(mode="markdown", extras=prompt_extras)
    res = model.infer(
        tok,
        prompt=prompt,
        image_file=str(image_path),
        output_path=str(out_dir),
        base_size=base_size,
        image_size=image_size,
        crop_mode=crop_mode,
        save_results=True,
        test_compress=test_compress,
    )
    # Handle None return from model.infer() when errors occur
    if res is None:
        raise RuntimeError("Model inference returned None - likely a shape mismatch or MPS error")
    return res.get("text", "")


def collect_targets(target: Path, dpi: int, tmp_dir: Path):
    """
    Collect all images to process from target path.

    Args:
        target: Input file or directory path
        dpi: DPI for PDF rendering
        tmp_dir: Temporary directory for rendered PDFs

    Returns:
        List of image paths to process
    """
    if target.is_file() and is_image(target):
        return [target]

    if target.is_file() and is_pdf(target):
        scale = max(1.0, dpi / 72.0)
        pdf_imgs_dir = ensure_dir(tmp_dir / f"{target.stem}_pages")
        return render_pdf_to_images(target, pdf_imgs_dir, scale)

    if target.is_dir():
        files = [Path(p) for p in sorted(glob(str(target / "*")))]
        imgs = [p for p in files if p.is_file() and is_image(p)]
        pdfs = [p for p in files if p.is_file() and is_pdf(p)]

        out = list(imgs)
        for pdf in pdfs:
            scale = max(1.0, dpi / 72.0)
            pdf_imgs_dir = ensure_dir(tmp_dir / f"{pdf.stem}_pages")
            out.extend(render_pdf_to_images(pdf, pdf_imgs_dir, scale))

        return sorted(out)

    print(f"ERROR: Unsupported target: {target}")
    sys.exit(3)


def main():
    """Main CLI entrypoint."""
    ap = argparse.ArgumentParser(
        description="DeepSeek-OCR Mac CLI - Offline OCR for PDFs and images with extensions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s sample.png                              # Process single image
  %(prog)s document.pdf                            # Process PDF file
  %(prog)s ./scans                                 # Process directory
  %(prog)s file.pdf --dpi 360                      # High quality
  %(prog)s file.pdf --emit-csv --math-latex        # Extract tables and math
  %(prog)s file.pdf --compression high --workers 3 # Fast parallel processing
  %(prog)s file.pdf --strict --min-words 50        # Quality gate
        """,
    )
    ap.add_argument("path", help="File or folder (image/pdf or dir)")
    ap.add_argument("-o", "--out", default="outputs", help="Output directory (default: outputs)")
    ap.add_argument(
        "--model",
        default="deepseek-ai/DeepSeek-OCR",
        help="Model name (default: deepseek-ai/DeepSeek-OCR)",
    )
    ap.add_argument("--base-size", type=int, default=None, help="Base resolution (default: from preset)")
    ap.add_argument("--image-size", type=int, default=None, help="Target image size (default: from preset)")
    ap.add_argument("--no-crop", action="store_true", help="Disable intelligent cropping")
    ap.add_argument("--no-compress", action="store_true", help="Disable compression")
    ap.add_argument("--dpi", type=int, default=288, help="DPI for PDF rendering (default: 288)")

    # Extension arguments
    ap.add_argument(
        "--emit-csv",
        nargs="?",
        const="csv",
        choices=["csv", "tsv"],
        help="Extract Markdown tables to CSV/TSV (default csv if no arg)",
    )
    ap.add_argument("--math-latex", action="store_true", help="Export equations as LaTeX files")
    ap.add_argument("--code-lang", action="store_true", help="Tag code blocks with languages")
    ap.add_argument(
        "--chart-to-csv", action="store_true", help="(Experimental) Extract chart data to CSV"
    )
    ap.add_argument(
        "--emit-boxes", action="store_true", help="Emit per-page bounding-box JSON if available"
    )
    ap.add_argument("--overlay", action="store_true", help="Render bbox overlays to PNG")
    ap.add_argument(
        "--compression",
        choices=["low", "med", "high"],
        default="low",
        help="Compression/speed trade-off preset (default: low)",
    )
    ap.add_argument("--workers", type=int, default=1, help="Parallel page workers (default: 1)")
    ap.add_argument("--emit-chunks", action="store_true", help="Emit chunks.jsonl for RAG")
    ap.add_argument("--strict", action="store_true", help="Fail if basic quality checks fail")
    ap.add_argument(
        "--min-words", type=int, default=20, help="Min words per page for --strict (default: 20)"
    )

    args = ap.parse_args()

    # Apply compression presets (user-specified values override presets)
    preset = COMPRESSION_PRESETS[args.compression]
    base_size = args.base_size if args.base_size is not None else preset["base_size"]
    image_size = args.image_size if args.image_size is not None else preset["image_size"]
    crop_mode = preset["crop_mode"] if not args.no_crop else False
    test_compress = preset["test_compress"] if not args.no_compress else False

    # Build prompt extras
    prompt_extras = []
    if args.math_latex:
        prompt_extras.append("Extract equations as LaTeX where possible.")
    if args.code_lang:
        prompt_extras.append("Tag code blocks with the correct language fences.")
    if args.emit_boxes:
        prompt_extras.append("Return bounding boxes for text blocks if available.")

    target = Path(args.path).expanduser().resolve()
    out_dir = ensure_dir(Path(args.out))
    merged_path = out_dir / "merged_output.md"

    # Check if target exists
    if not target.exists():
        print(f"ERROR: Path does not exist: {target}")
        sys.exit(1)

    # Determine device (MPS for Apple Silicon, else CPU)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model and tokenizer
    print("Loading model…")
    try:
        # Note: trust_remote_code=True is required for DeepSeek-OCR's custom model code
        # This is safe when loading from the official deepseek-ai/DeepSeek-OCR repository
        # Users can verify the model source before loading
        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)  # nosec B615
        # Fix MPS padding issue - right padding avoids known MPS generation bugs
        tok.padding_side = "right"

        # MPS compatibility: use eager attention (not flash_attention_2) and float32
        # See: https://huggingface.co/deepseek-ai/DeepSeek-OCR/discussions/20
        model = AutoModel.from_pretrained(
            args.model,
            trust_remote_code=True,  # nosec B615
            use_safetensors=True,
            attn_implementation="eager",  # MPS doesn't support flash_attention_2
        )
        # Convert to float32 for MPS compatibility (bfloat16 causes dtype mismatches)
        if device == "mps":
            model = model.float()
        model = model.to(device).eval()
    except Exception as e:
        print(f"ERROR loading model: {e}")
        sys.exit(5)

    # Create temporary directory for PDF rendering
    tmp_root = Path(tempfile.mkdtemp(prefix="deepseek_ocr_mac_"))

    try:
        # Collect all files to process
        files = collect_targets(target, args.dpi, tmp_root)

        if not files:
            print("No input files found.")
            sys.exit(4)

        print(f"Found {len(files)} page(s).")

        # Initialize stats
        stats = ProcessingStats(
            total_pages=len(files),
            compression=args.compression,
            workers=args.workers,
            start_time=time.time(),
        )

        # Initialize chunks file if needed
        chunks_path = out_dir / "chunks.jsonl"
        if args.emit_chunks and chunks_path.exists():
            chunks_path.unlink()  # Clear existing file

        # Process each file (serial processing - parallel can be added later)
        page_md_sections = []

        for idx, f in enumerate(files, start=1):
            print(f"→ OCR [{idx}/{len(files)}] {f.name}")

            try:
                # Run OCR
                text = run_infer(
                    model,
                    tok,
                    f,
                    out_dir,
                    base_size=base_size,
                    image_size=image_size,
                    crop_mode=crop_mode,
                    test_compress=test_compress,
                    prompt_extras=prompt_extras if prompt_extras else None,
                )

                # Quality check
                words = word_count(text)
                stats.total_words += words

                if args.strict and words < args.min_words:
                    print(f"  ⚠️  FAILED: Only {words} words (minimum {args.min_words})")
                    stats.failed_pages.append(idx)

                # Post-processing: Extract tables
                if args.emit_csv:
                    tables = extract_tables_to_csv(text, out_dir, idx, fmt=args.emit_csv)
                    if tables:
                        print(f"  📊 Extracted {len(tables)} table(s)")

                # Post-processing: Extract math
                if args.math_latex:
                    equations = extract_math(text, out_dir, idx)
                    if equations:
                        print(f"  🧮 Extracted {len(equations)} equation(s)")

                # Post-processing: Tag code languages
                if args.code_lang:
                    text = tag_code_languages(text)

                # Post-processing: Extract chart data
                if args.chart_to_csv:
                    charts = extract_chart_data(text, out_dir, idx)
                    if charts:
                        print(f"  📈 Extracted {len(charts)} chart(s)")

                # Post-processing: RAG chunks
                if args.emit_chunks:
                    emit_chunks(chunks_path, idx, text)

                # Post-processing: Bounding boxes
                if args.emit_boxes:
                    boxes_json = extract_bounding_boxes_heuristic(f, out_dir, idx)
                    if boxes_json and args.overlay:
                        create_overlay(f, boxes_json, out_dir, idx)
                        print("  🔲 Created bounding boxes and overlay")
                    elif boxes_json:
                        print("  🔲 Created bounding boxes")

                # Add to results
                page_md_sections.append(f"# Page {idx}\n\n{text}\n")

            except Exception as e:
                print(f"  ❌ Error processing page {idx}: {e}")
                stats.failed_pages.append(idx)
                page_md_sections.append(f"# Page {idx}\n\n*Error: {e}*\n")

        # Build quality summary
        stats.end_time = time.time()

        quality_summary = f"""
---
quality:
  pages: {stats.total_pages}
  failed_pages: {len(stats.failed_pages)}
  success_rate: {stats.success_rate:.1f}%
  total_words: {stats.total_words}
  min_words: {args.min_words}
  compression: {args.compression}
  workers: {args.workers}
  processing_time: {stats.processing_time:.2f}s
  model: {args.model}
---
"""

        # Write merged output with quality summary
        merged_content = "\n".join(page_md_sections) + "\n" + quality_summary
        merged_path.write_text(merged_content, encoding="utf-8")

        print(f"\n✅ Done! Markdown saved to {merged_path.resolve()}")
        print(f"   Processed: {stats.total_pages} pages in {stats.processing_time:.2f}s")
        print(f"   Success rate: {stats.success_rate:.1f}%")

        # Exit with appropriate code
        if args.strict and stats.failed_pages:
            print(f"❌ STRICT MODE: {len(stats.failed_pages)} pages failed quality check")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(6)
    finally:
        # Clean up memory: delete model and tokenizer to free GPU/MPS memory
        try:
            del model
            del tok
            # Clear MPS cache if using MPS device
            if torch.backends.mps.is_available() and device == "mps":
                torch.mps.empty_cache()
            # Force garbage collection
            gc.collect()
        except Exception:
            pass  # Ignore cleanup errors

        # Clean up temporary files
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
