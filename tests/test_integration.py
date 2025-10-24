"""
Integration tests for DeepSeek-OCR Mac CLI with mocked model.
"""
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.fixtures import create_test_image, create_test_pdf, create_fixture_files


class MockModel:
    """Mock DeepSeek-OCR model for testing."""

    def __init__(self, device='cpu'):
        self.device = device

    def to(self, device):
        """Mock to() method."""
        self.device = device
        return self

    def eval(self):
        """Mock eval() method."""
        return self

    def infer(self, tokenizer, prompt, image_file, output_path, **kwargs):
        """Mock infer() method that returns fake OCR results."""
        img_name = Path(image_file).name
        fake_text = f"# OCR Result for {img_name}\n\nThis is a test OCR output.\n\n**Sample text** extracted from the document."
        return {"text": fake_text}


class MockTokenizer:
    """Mock tokenizer for testing."""

    @classmethod
    def from_pretrained(cls, model_name, **kwargs):
        return cls()


@pytest.fixture
def mock_transformers():
    """Fixture to mock transformers library."""
    with patch('deepseek_ocr_mac.AutoTokenizer') as mock_tok, \
         patch('deepseek_ocr_mac.AutoModel') as mock_model:

        mock_tok.from_pretrained.return_value = MockTokenizer()
        mock_model.from_pretrained.return_value = MockModel()

        yield {
            'tokenizer': mock_tok,
            'model': mock_model
        }


@pytest.fixture
def mock_torch():
    """Fixture to mock torch MPS availability."""
    with patch('deepseek_ocr_mac.torch') as mock_torch_module:
        mock_torch_module.backends.mps.is_available.return_value = True
        yield mock_torch_module


class TestCLIIntegration:
    """Integration tests for CLI functionality."""

    def test_run_infer_basic(self, tmp_path, mock_transformers):
        """Test basic inference run."""
        from deepseek_ocr_mac import run_infer

        # Create test image
        img_path = tmp_path / "test.png"
        create_test_image(img_path, "Test Document")

        out_dir = tmp_path / "output"
        out_dir.mkdir()

        # Create mock model and tokenizer
        model = MockModel()
        tokenizer = MockTokenizer()

        # Run inference
        result = run_infer(
            model, tokenizer, img_path, out_dir,
            base_size=1024, image_size=640,
            crop_mode=True, test_compress=True
        )

        assert isinstance(result, str)
        assert "OCR Result" in result
        assert "test.png" in result

    def test_main_with_single_image(self, tmp_path, mock_transformers, mock_torch, monkeypatch, capsys):
        """Test main CLI with a single image."""
        import deepseek_ocr_mac

        # Create test image
        img_path = tmp_path / "test.png"
        create_test_image(img_path)

        out_dir = tmp_path / "output"

        # Mock command-line arguments
        test_args = [
            'deepseek_ocr_mac.py',
            str(img_path),
            '-o', str(out_dir)
        ]

        monkeypatch.setattr(sys, 'argv', test_args)

        # Run main
        try:
            deepseek_ocr_mac.main()
        except SystemExit:
            pass  # Ignore successful exits

        # Check output
        captured = capsys.readouterr()
        assert "Using device:" in captured.out
        assert "Loading model" in captured.out

        # Check that output file was created
        merged_output = out_dir / "merged_output.md"
        assert merged_output.exists()
        content = merged_output.read_text()
        assert "# test.png" in content

    def test_main_with_pdf(self, tmp_path, mock_transformers, mock_torch, monkeypatch, capsys):
        """Test main CLI with a PDF file."""
        pytest.importorskip("fitz")

        import deepseek_ocr_mac

        # Create test PDF
        pdf_path = tmp_path / "test.pdf"
        create_test_pdf(pdf_path, num_pages=2)

        out_dir = tmp_path / "output"

        # Mock command-line arguments
        test_args = [
            'deepseek_ocr_mac.py',
            str(pdf_path),
            '-o', str(out_dir)
        ]

        monkeypatch.setattr(sys, 'argv', test_args)

        # Run main
        try:
            deepseek_ocr_mac.main()
        except SystemExit:
            pass

        # Check output
        merged_output = out_dir / "merged_output.md"
        assert merged_output.exists()
        content = merged_output.read_text()

        # Should have processed 2 pages
        assert "test-p0001.png" in content
        assert "test-p0002.png" in content

    def test_main_with_directory(self, tmp_path, mock_transformers, mock_torch, monkeypatch, capsys):
        """Test main CLI with a directory of files."""
        import deepseek_ocr_mac

        # Create test files
        fixtures = create_fixture_files(tmp_path)

        out_dir = tmp_path / "output"

        # Mock command-line arguments
        test_args = [
            'deepseek_ocr_mac.py',
            str(tmp_path),
            '-o', str(out_dir)
        ]

        monkeypatch.setattr(sys, 'argv', test_args)

        # Run main
        try:
            deepseek_ocr_mac.main()
        except SystemExit:
            pass

        # Check output
        captured = capsys.readouterr()
        assert "Found" in captured.out and "page(s)" in captured.out

        merged_output = out_dir / "merged_output.md"
        assert merged_output.exists()

    def test_main_with_custom_parameters(self, tmp_path, mock_transformers, mock_torch, monkeypatch):
        """Test main CLI with custom parameters."""
        import deepseek_ocr_mac

        img_path = tmp_path / "test.png"
        create_test_image(img_path)

        out_dir = tmp_path / "output"

        test_args = [
            'deepseek_ocr_mac.py',
            str(img_path),
            '-o', str(out_dir),
            '--base-size', '2048',
            '--image-size', '1024',
            '--dpi', '360',
            '--no-crop',
            '--no-compress'
        ]

        monkeypatch.setattr(sys, 'argv', test_args)

        try:
            deepseek_ocr_mac.main()
        except SystemExit:
            pass

        # Should complete without errors
        merged_output = out_dir / "merged_output.md"
        assert merged_output.exists()

    def test_main_nonexistent_file(self, tmp_path, monkeypatch, capsys):
        """Test main CLI with nonexistent file."""
        import deepseek_ocr_mac

        nonexistent = tmp_path / "does_not_exist.png"

        test_args = [
            'deepseek_ocr_mac.py',
            str(nonexistent)
        ]

        monkeypatch.setattr(sys, 'argv', test_args)

        with pytest.raises(SystemExit) as exc_info:
            deepseek_ocr_mac.main()

        assert exc_info.value.code == 1

        captured = capsys.readouterr()
        assert "ERROR" in captured.out
        assert "does not exist" in captured.out


class TestArgumentParsing:
    """Test CLI argument parsing."""

    def test_help_option(self, capsys):
        """Test --help option."""
        import deepseek_ocr_mac
        import sys

        with pytest.raises(SystemExit) as exc_info:
            sys.argv = ['deepseek_ocr_mac.py', '--help']
            deepseek_ocr_mac.main()

        assert exc_info.value.code == 0

        captured = capsys.readouterr()
        assert "usage:" in captured.out
        assert "DeepSeek-OCR Mac CLI" in captured.out

    def test_default_parameters(self, tmp_path, mock_transformers, mock_torch, monkeypatch):
        """Test that default parameters are applied correctly."""
        import deepseek_ocr_mac
        from unittest.mock import patch

        img_path = tmp_path / "test.png"
        create_test_image(img_path)

        test_args = [
            'deepseek_ocr_mac.py',
            str(img_path)
        ]

        monkeypatch.setattr(sys, 'argv', test_args)

        # Patch run_infer to check parameters
        original_run_infer = deepseek_ocr_mac.run_infer

        def check_run_infer(*args, **kwargs):
            assert kwargs['base_size'] == 1024  # Default
            assert kwargs['image_size'] == 640   # Default
            assert kwargs['crop_mode'] is True   # Default (no --no-crop)
            assert kwargs['test_compress'] is True  # Default (no --no-compress)
            return original_run_infer(*args, **kwargs)

        with patch.object(deepseek_ocr_mac, 'run_infer', side_effect=check_run_infer):
            try:
                deepseek_ocr_mac.main()
            except SystemExit:
                pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
