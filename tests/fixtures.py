"""
Test fixtures and utilities for DeepSeek-OCR Mac CLI tests.
"""

import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def create_test_image(path: Path, text: str = "Test Document", size=(800, 600)):
    """Create a simple test image with text."""
    img = Image.new("RGB", size, color="white")
    draw = ImageDraw.Draw(img)

    # Draw some text
    try:
        # Try to use a default font
        draw.text((50, 50), text, fill="black")
    except Exception:
        # Fallback if no font available
        draw.rectangle([50, 50, 750, 100], outline="black", width=2)

    img.save(path)
    return path


def create_test_pdf(path: Path, num_pages: int = 2):
    """Create a simple test PDF with multiple pages."""
    try:
        import fitz  # PyMuPDF

        doc = fitz.open()
        for i in range(num_pages):
            page = doc.new_page(width=595, height=842)  # A4 size
            text = f"Test Page {i + 1}"
            page.insert_text((50, 50), text, fontsize=20)

        doc.save(path)
        doc.close()
        return path
    except ImportError:
        # If PyMuPDF not available, create a dummy file
        path.write_bytes(b"%PDF-1.4\n%Dummy PDF for testing")
        return path


def create_fixture_files(tmp_path: Path):
    """Create a set of test fixture files."""
    # Create images
    img1 = tmp_path / "test_image_1.png"
    img2 = tmp_path / "test_image_2.jpg"

    create_test_image(img1, "Test Image 1")
    create_test_image(img2, "Test Image 2")

    # Create PDF
    pdf = tmp_path / "test_document.pdf"
    create_test_pdf(pdf, num_pages=3)

    # Create unsupported file
    unsupported = tmp_path / "test.txt"
    unsupported.write_text("This is not an image or PDF")

    return {"images": [img1, img2], "pdf": pdf, "unsupported": unsupported}
