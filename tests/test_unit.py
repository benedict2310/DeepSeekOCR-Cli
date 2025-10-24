"""
Unit tests for DeepSeek-OCR Mac CLI helper functions.
"""
import sys
import tempfile
from pathlib import Path
import pytest

# Add parent directory to path to import the module
sys.path.insert(0, str(Path(__file__).parent.parent))

from deepseek_ocr_mac import (
    is_image,
    is_pdf,
    ensure_dir,
    IMG_EXTS
)
from tests.fixtures import create_test_image, create_test_pdf


class TestFileTypeDetection:
    """Test file type detection functions."""

    def test_is_image_with_supported_formats(self):
        """Test that supported image formats are correctly identified."""
        for ext in IMG_EXTS:
            path = Path(f"test{ext}")
            assert is_image(path), f"Failed to identify {ext} as image"

    def test_is_image_case_insensitive(self):
        """Test that image detection is case-insensitive."""
        assert is_image(Path("test.PNG"))
        assert is_image(Path("test.JpG"))
        assert is_image(Path("test.JPEG"))

    def test_is_image_rejects_non_images(self):
        """Test that non-image files are rejected."""
        assert not is_image(Path("test.pdf"))
        assert not is_image(Path("test.txt"))
        assert not is_image(Path("test.doc"))

    def test_is_pdf_identifies_pdf(self):
        """Test that PDF files are correctly identified."""
        assert is_pdf(Path("document.pdf"))
        assert is_pdf(Path("document.PDF"))

    def test_is_pdf_rejects_non_pdf(self):
        """Test that non-PDF files are rejected."""
        assert not is_pdf(Path("image.png"))
        assert not is_pdf(Path("document.txt"))


class TestDirectoryOperations:
    """Test directory operation functions."""

    def test_ensure_dir_creates_directory(self, tmp_path):
        """Test that ensure_dir creates a new directory."""
        new_dir = tmp_path / "test_dir"
        assert not new_dir.exists()

        result = ensure_dir(new_dir)

        assert new_dir.exists()
        assert new_dir.is_dir()
        assert result == new_dir

    def test_ensure_dir_creates_nested_directories(self, tmp_path):
        """Test that ensure_dir creates nested directories."""
        nested_dir = tmp_path / "level1" / "level2" / "level3"
        assert not nested_dir.exists()

        result = ensure_dir(nested_dir)

        assert nested_dir.exists()
        assert nested_dir.is_dir()
        assert result == nested_dir

    def test_ensure_dir_idempotent(self, tmp_path):
        """Test that ensure_dir is idempotent (can be called multiple times)."""
        test_dir = tmp_path / "test_dir"

        result1 = ensure_dir(test_dir)
        result2 = ensure_dir(test_dir)

        assert test_dir.exists()
        assert result1 == result2 == test_dir


class TestPDFRendering:
    """Test PDF rendering functionality."""

    def test_render_pdf_to_images_basic(self, tmp_path):
        """Test basic PDF to image conversion."""
        pytest.importorskip("fitz")  # Skip if PyMuPDF not available

        from deepseek_ocr_mac import render_pdf_to_images

        # Create test PDF
        pdf_path = tmp_path / "test.pdf"
        create_test_pdf(pdf_path, num_pages=2)

        out_dir = tmp_path / "output"
        out_dir.mkdir()

        # Render PDF
        images = render_pdf_to_images(pdf_path, out_dir, scale=1.0)

        assert len(images) == 2
        assert all(img.exists() for img in images)
        assert all(img.suffix == '.png' for img in images)

    def test_render_pdf_to_images_naming(self, tmp_path):
        """Test that rendered images have correct naming."""
        pytest.importorskip("fitz")

        from deepseek_ocr_mac import render_pdf_to_images

        pdf_path = tmp_path / "mydoc.pdf"
        create_test_pdf(pdf_path, num_pages=3)

        out_dir = tmp_path / "output"
        out_dir.mkdir()

        images = render_pdf_to_images(pdf_path, out_dir, scale=1.0)

        assert images[0].name == "mydoc-p0001.png"
        assert images[1].name == "mydoc-p0002.png"
        assert images[2].name == "mydoc-p0003.png"

    def test_render_pdf_to_images_scaling(self, tmp_path):
        """Test that scaling parameter affects output."""
        pytest.importorskip("fitz")

        from deepseek_ocr_mac import render_pdf_to_images
        from PIL import Image

        pdf_path = tmp_path / "test.pdf"
        create_test_pdf(pdf_path, num_pages=1)

        out_dir1 = tmp_path / "output1"
        out_dir2 = tmp_path / "output2"
        out_dir1.mkdir()
        out_dir2.mkdir()

        # Render at different scales
        images1 = render_pdf_to_images(pdf_path, out_dir1, scale=1.0)
        images2 = render_pdf_to_images(pdf_path, out_dir2, scale=2.0)

        # Check that scaled image is larger
        img1 = Image.open(images1[0])
        img2 = Image.open(images2[0])

        assert img2.size[0] > img1.size[0]
        assert img2.size[1] > img1.size[1]


class TestCollectTargets:
    """Test target file collection functionality."""

    def test_collect_targets_single_image(self, tmp_path):
        """Test collecting a single image file."""
        from deepseek_ocr_mac import collect_targets

        img_path = tmp_path / "test.png"
        create_test_image(img_path)

        tmp_dir = tmp_path / "tmp"
        tmp_dir.mkdir()

        targets = collect_targets(img_path, dpi=288, tmp_dir=tmp_dir)

        assert len(targets) == 1
        assert targets[0] == img_path

    def test_collect_targets_directory_with_images(self, tmp_path):
        """Test collecting images from a directory."""
        from deepseek_ocr_mac import collect_targets

        # Create multiple images
        for i in range(3):
            create_test_image(tmp_path / f"test_{i}.png")

        tmp_dir = tmp_path / "tmp"
        tmp_dir.mkdir()

        targets = collect_targets(tmp_path, dpi=288, tmp_dir=tmp_dir)

        assert len(targets) >= 3
        assert all(is_image(t) for t in targets)

    def test_collect_targets_pdf_conversion(self, tmp_path):
        """Test that PDF files are converted to images."""
        pytest.importorskip("fitz")

        from deepseek_ocr_mac import collect_targets

        pdf_path = tmp_path / "test.pdf"
        create_test_pdf(pdf_path, num_pages=2)

        tmp_dir = tmp_path / "tmp"
        tmp_dir.mkdir()

        targets = collect_targets(pdf_path, dpi=288, tmp_dir=tmp_dir)

        assert len(targets) == 2
        assert all(t.suffix == '.png' for t in targets)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
