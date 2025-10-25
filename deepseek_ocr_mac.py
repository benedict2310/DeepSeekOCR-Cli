#!/usr/bin/env python3
"""
DeepSeek-OCR Mac CLI
Offline OCR for PDFs and images using DeepSeek-OCR on macOS (Apple Silicon)
"""
import argparse
import shutil
import sys
import tempfile
from glob import glob
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


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


def run_infer(model, tok, image_path, out_dir, base_size, image_size, crop_mode, test_compress):
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

    Returns:
        Extracted markdown text
    """
    prompt = "<image>\n<|grounding|>Convert the document to markdown."
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
        description="DeepSeek-OCR Mac CLI - Offline OCR for PDFs and images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s sample.png                    # Process single image
  %(prog)s document.pdf                  # Process PDF file
  %(prog)s ./scans                       # Process directory
  %(prog)s file.pdf --dpi 360            # High quality
  %(prog)s file.pdf --no-crop            # Disable auto-crop
        """,
    )
    ap.add_argument("path", help="File or folder (image/pdf or dir)")
    ap.add_argument("-o", "--out", default="outputs", help="Output directory (default: outputs)")
    ap.add_argument(
        "--model",
        default="deepseek-ai/DeepSeek-OCR",
        help="Model name (default: deepseek-ai/DeepSeek-OCR)",
    )
    ap.add_argument("--base-size", type=int, default=1024, help="Base resolution (default: 1024)")
    ap.add_argument("--image-size", type=int, default=640, help="Target image size (default: 640)")
    ap.add_argument("--no-crop", action="store_true", help="Disable intelligent cropping")
    ap.add_argument("--no-compress", action="store_true", help="Disable compression")
    ap.add_argument("--dpi", type=int, default=288, help="DPI for PDF rendering (default: 288)")
    args = ap.parse_args()

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

        # Process each file
        results = []
        for idx, f in enumerate(files, start=1):
            print(f"→ OCR [{idx}/{len(files)}] {f.name}")
            text = run_infer(
                model,
                tok,
                f,
                out_dir,
                base_size=args.base_size,
                image_size=args.image_size,
                crop_mode=not args.no_crop,
                test_compress=not args.no_compress,
            )
            results.append(f"# {f.name}\n\n{text}\n")

        # Write merged output
        merged_path.write_text("\n".join(results), encoding="utf-8")
        print(f"\n✅ Done! Markdown saved to {merged_path.resolve()}")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(6)
    finally:
        # Clean up temporary files
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
