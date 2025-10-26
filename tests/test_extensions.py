"""
Unit tests for extension features added to DeepSeek-OCR Mac CLI.

Tests all post-processing functions including table extraction, math extraction,
code language detection, chart extraction, RAG chunks, and bounding boxes.
"""

import json
import tempfile
from pathlib import Path

import pytest

from deepseek_ocr_mac import (
    CODE_LANG_PATTERNS,
    COMPRESSION_PRESETS,
    ProcessingStats,
    build_prompt,
    create_overlay,
    detect_code_language,
    emit_chunks,
    extract_bounding_boxes_heuristic,
    extract_chart_data,
    extract_math,
    extract_tables_to_csv,
    md_table_to_rows,
    tag_code_languages,
    word_count,
)


class TestProcessingStats:
    """Tests for ProcessingStats dataclass."""

    def test_default_values(self):
        """Test default initialization."""
        stats = ProcessingStats()
        assert stats.total_pages == 0
        assert stats.failed_pages == []
        assert stats.total_words == 0
        assert stats.compression == "low"
        assert stats.workers == 1
        assert stats.start_time == 0.0
        assert stats.end_time == 0.0

    def test_success_rate_no_pages(self):
        """Test success rate with no pages."""
        stats = ProcessingStats(total_pages=0)
        assert stats.success_rate == 0.0

    def test_success_rate_all_success(self):
        """Test success rate with all pages successful."""
        stats = ProcessingStats(total_pages=10, failed_pages=[])
        assert stats.success_rate == 100.0

    def test_success_rate_partial_failure(self):
        """Test success rate with some failures."""
        stats = ProcessingStats(total_pages=10, failed_pages=[1, 3, 5])
        assert stats.success_rate == 70.0

    def test_processing_time(self):
        """Test processing time calculation."""
        stats = ProcessingStats(start_time=100.0, end_time=150.5)
        assert stats.processing_time == 50.5


class TestConstants:
    """Tests for module constants."""

    def test_compression_presets(self):
        """Test compression presets are properly configured."""
        assert "low" in COMPRESSION_PRESETS
        assert "med" in COMPRESSION_PRESETS
        assert "high" in COMPRESSION_PRESETS

        # Check low compression settings (baseline quality, safe base_size)
        assert COMPRESSION_PRESETS["low"]["base_size"] == 1024  # Safe maximum
        assert COMPRESSION_PRESETS["low"]["image_size"] == 640
        assert COMPRESSION_PRESETS["low"]["test_compress"] is False

        # Check high compression settings (reduced for speed)
        assert COMPRESSION_PRESETS["high"]["base_size"] == 896
        assert COMPRESSION_PRESETS["high"]["image_size"] == 512
        assert COMPRESSION_PRESETS["high"]["test_compress"] is True

        # Verify no preset exceeds safe base_size limit
        for preset_name, preset in COMPRESSION_PRESETS.items():
            assert preset["base_size"] <= 1024, f"{preset_name} preset exceeds safe base_size=1024"

    def test_code_lang_patterns(self):
        """Test code language patterns are defined."""
        assert "python" in CODE_LANG_PATTERNS
        assert "javascript" in CODE_LANG_PATTERNS
        assert "rust" in CODE_LANG_PATTERNS
        assert len(CODE_LANG_PATTERNS) == 10


class TestBuildPrompt:
    """Tests for build_prompt function."""

    def test_basic_markdown_prompt(self):
        """Test basic markdown prompt generation."""
        prompt = build_prompt()
        assert "<image>" in prompt
        assert "markdown" in prompt.lower()

    def test_prompt_with_extras(self):
        """Test prompt with additional instructions."""
        extras = ["Extract equations as LaTeX.", "Tag code blocks."]
        prompt = build_prompt(extras=extras)
        assert "LaTeX" in prompt
        assert "Tag code blocks" in prompt

    def test_prompt_with_empty_extras(self):
        """Test prompt with empty extras list."""
        prompt = build_prompt(extras=[])
        assert "<image>" in prompt
        assert "markdown" in prompt.lower()


class TestWordCount:
    """Tests for word_count function."""

    def test_simple_sentence(self):
        """Test word count on simple sentence."""
        assert word_count("Hello world") == 2
        assert word_count("The quick brown fox") == 4

    def test_with_punctuation(self):
        """Test word count with punctuation."""
        assert word_count("Hello, world! How are you?") == 5

    def test_with_numbers(self):
        """Test word count with numbers."""
        assert word_count("Test 123 test") == 3
        assert word_count("123 456") == 2

    def test_empty_string(self):
        """Test word count on empty string."""
        assert word_count("") == 0
        assert word_count("   ") == 0

    def test_special_characters(self):
        """Test word count with special characters."""
        assert word_count("@#$%") == 0
        assert word_count("word @#$% word") == 2


class TestTableExtraction:
    """Tests for table extraction functions."""

    def test_md_table_to_rows_simple(self):
        """Test simple Markdown table conversion."""
        table = "| A | B |\n|---|---|\n| 1 | 2 |"
        rows = md_table_to_rows(table)
        assert rows == [["A", "B"], ["1", "2"]]

    def test_md_table_to_rows_with_alignment(self):
        """Test table with alignment markers."""
        table = "| Left | Center | Right |\n|:-----|:------:|------:|\n| A | B | C |"
        rows = md_table_to_rows(table)
        assert rows == [["Left", "Center", "Right"], ["A", "B", "C"]]

    def test_md_table_to_rows_empty_cells(self):
        """Test table with empty cells."""
        table = "| A |  | C |\n|---|---|---|\n|   | B |   |"
        rows = md_table_to_rows(table)
        assert rows[0] == ["A", "", "C"]
        assert rows[1] == ["", "B", ""]

    def test_extract_tables_to_csv(self, tmp_path):
        """Test extracting tables to CSV files."""
        text = "| Col1 | Col2 |\n|------|------|\n| A    | B    |\n| C    | D    |"
        files = extract_tables_to_csv(text, tmp_path, 1, "csv")

        assert len(files) == 1
        assert files[0].exists()
        assert files[0].name == "page_0001_table_1.csv"

        content = files[0].read_text()
        assert "Col1,Col2" in content
        assert "A,B" in content

    def test_extract_tables_to_tsv(self, tmp_path):
        """Test extracting tables to TSV files."""
        text = "| X | Y |\n|---|---|\n| 1 | 2 |"
        files = extract_tables_to_csv(text, tmp_path, 5, "tsv")

        assert len(files) == 1
        assert files[0].suffix == ".tsv"
        assert "page_0005_table_1.tsv" in files[0].name

    def test_extract_multiple_tables(self, tmp_path):
        """Test extracting multiple tables from same text."""
        text = """
| Table1 |
|--------|
| A      |

Some text

| Table2 |
|--------|
| B      |
"""
        files = extract_tables_to_csv(text, tmp_path, 1, "csv")
        assert len(files) == 2


class TestMathExtraction:
    """Tests for math extraction function."""

    def test_extract_inline_math(self, tmp_path):
        """Test extracting inline LaTeX math."""
        text = "Inline math $x^2$ and more $y = mx + b$"
        files = extract_math(text, tmp_path, 1)

        assert len(files) == 2
        assert files[0].read_text() == "x^2"
        assert files[1].read_text() == "y = mx + b"

    def test_extract_display_math(self, tmp_path):
        """Test extracting display LaTeX math."""
        text = "Display math $$\\int_0^1 f(x) dx$$"
        files = extract_math(text, tmp_path, 1)

        assert len(files) == 1
        content = files[0].read_text()
        assert "int" in content
        assert "f(x)" in content

    def test_extract_mixed_math(self, tmp_path):
        """Test extracting mixed inline and display math."""
        text = "Inline $a+b$ and display $$c+d$$ and inline $e$"
        files = extract_math(text, tmp_path, 2)

        assert len(files) == 3
        assert files[0].name == "page_0002_eq_001.tex"
        assert files[1].name == "page_0002_eq_002.tex"
        assert files[2].name == "page_0002_eq_003.tex"

    def test_no_math(self, tmp_path):
        """Test text with no math expressions."""
        text = "Just regular text without any math"
        files = extract_math(text, tmp_path, 1)
        assert len(files) == 0


class TestCodeLanguageDetection:
    """Tests for code language detection functions."""

    def test_detect_python(self):
        """Test detecting Python code."""
        code = """
def hello():
    import os
    return 42
"""
        assert detect_code_language(code) == "python"

    def test_detect_javascript(self):
        """Test detecting JavaScript code."""
        code = "const x = () => { let y = 5; function foo() {} }"
        assert detect_code_language(code) == "javascript"

    def test_detect_rust(self):
        """Test detecting Rust code."""
        code = "fn main() { let mut x = 5; impl Trait {} }"
        assert detect_code_language(code) == "rust"

    def test_detect_sql(self):
        """Test detecting SQL code."""
        code = "SELECT * FROM users WHERE id = 1 JOIN orders"
        assert detect_code_language(code) == "sql"

    def test_uncertain_detection(self):
        """Test that uncertain code returns None."""
        code = "x = 1"  # Too simple to confidently detect
        result = detect_code_language(code)
        # Should return None because score < 2
        assert result is None

    def test_tag_code_languages_python(self):
        """Test tagging Python code blocks."""
        text = """
```
def hello():
    import os
    print("hi")
```
"""
        tagged = tag_code_languages(text)
        assert "```python" in tagged

    def test_tag_code_languages_already_tagged(self):
        """Test that already-tagged blocks are left alone."""
        text = """
```rust
fn main() {}
```
"""
        tagged = tag_code_languages(text)
        assert "```rust" in tagged
        # Should not add another language tag

    def test_tag_code_languages_uncertain(self):
        """Test that uncertain code blocks are not tagged."""
        text = """
```
x = 1
```
"""
        tagged = tag_code_languages(text)
        # Should remain untagged
        assert tagged.count("```") == 2


class TestChartExtraction:
    """Tests for chart data extraction."""

    def test_extract_chart_csv(self, tmp_path):
        """Test extracting chart data from CSV fence."""
        text = """
```csv
Date,Value
2024-01,100
2024-02,150
```
"""
        files = extract_chart_data(text, tmp_path, 1)
        assert len(files) == 1
        assert files[0].exists()

        content = files[0].read_text()
        assert "Date" in content
        assert "Value" in content

    def test_extract_chart_data_fence(self, tmp_path):
        """Test extracting from chart-data fence."""
        text = """
```chart-data
X,Y
1,10
2,20
```
"""
        files = extract_chart_data(text, tmp_path, 1)
        assert len(files) == 1

    def test_no_chart_data(self, tmp_path):
        """Test text with no chart data."""
        text = "Just regular text"
        files = extract_chart_data(text, tmp_path, 1)
        assert len(files) == 0


class TestRAGChunks:
    """Tests for RAG chunks emission."""

    def test_emit_chunks_short_text(self, tmp_path):
        """Test emitting chunks from short text."""
        chunks_path = tmp_path / "chunks.jsonl"
        text = "Short text" * 50  # About 550 characters

        emit_chunks(chunks_path, 1, text)

        assert chunks_path.exists()
        lines = chunks_path.read_text().splitlines()
        assert len(lines) == 1

        chunk = json.loads(lines[0])
        assert chunk["page"] == 1
        assert chunk["start"] == 0
        assert "text" in chunk

    def test_emit_chunks_long_text(self, tmp_path):
        """Test emitting chunks from long text."""
        chunks_path = tmp_path / "chunks.jsonl"
        text = "A" * 3000  # 3000 characters

        emit_chunks(chunks_path, 2, text)

        lines = chunks_path.read_text().splitlines()
        assert len(lines) == 3  # 3000 / 1200 = 2.5, rounds to 3

        # Verify all chunks are for same page
        for line in lines:
            chunk = json.loads(line)
            assert chunk["page"] == 2

    def test_emit_chunks_append_mode(self, tmp_path):
        """Test that chunks are appended, not overwritten."""
        chunks_path = tmp_path / "chunks.jsonl"
        text1 = "First"
        text2 = "Second"

        emit_chunks(chunks_path, 1, text1)
        emit_chunks(chunks_path, 2, text2)

        lines = chunks_path.read_text().splitlines()
        assert len(lines) == 2

        chunk1 = json.loads(lines[0])
        chunk2 = json.loads(lines[1])
        assert chunk1["page"] == 1
        assert chunk2["page"] == 2


class TestBoundingBoxes:
    """Tests for bounding box extraction and overlay creation."""

    def test_extract_bounding_boxes(self, tmp_path):
        """Test extracting bounding boxes from an image."""
        from tests.fixtures import create_test_image

        # Create a test image
        img_path = tmp_path / "test.png"
        create_test_image(img_path, "Test Text", size=(800, 600))

        # Extract bounding boxes
        boxes_json = extract_bounding_boxes_heuristic(img_path, tmp_path, 1)

        assert boxes_json is not None
        assert boxes_json.exists()
        assert boxes_json.name == "page_0001.json"

        # Verify JSON structure
        data = json.loads(boxes_json.read_text())
        assert data["page"] == 1
        assert data["bbox_provider"] == "heuristic"
        assert "boxes" in data
        assert isinstance(data["boxes"], list)

        # Should have some boxes (grid-based heuristic)
        assert len(data["boxes"]) > 0

        # Verify box structure
        if data["boxes"]:
            box = data["boxes"][0]
            assert "x" in box
            assert "y" in box
            assert "w" in box
            assert "h" in box
            assert "hint" in box

    def test_create_overlay(self, tmp_path):
        """Test creating overlay image with bounding boxes."""
        from tests.fixtures import create_test_image

        # Create test image
        img_path = tmp_path / "test.png"
        create_test_image(img_path, "Test", size=(400, 300))

        # Create boxes JSON
        boxes_data = {
            "page": 1,
            "bbox_provider": "test",
            "boxes": [
                {"x": 10, "y": 10, "w": 100, "h": 50, "hint": "text"},
                {"x": 150, "y": 150, "w": 80, "h": 40, "hint": "text"},
            ],
        }
        boxes_json = tmp_path / "boxes.json"
        boxes_json.write_text(json.dumps(boxes_data))

        # Create overlay
        overlay_path = create_overlay(img_path, boxes_json, tmp_path, 1)

        assert overlay_path is not None
        assert overlay_path.exists()
        assert overlay_path.name == "page_0001_overlay.png"

        # Verify it's a valid image
        from PIL import Image

        img = Image.open(overlay_path)
        assert img.size == (400, 300)

    def test_extract_bounding_boxes_error_handling(self, tmp_path):
        """Test error handling for invalid image path."""
        invalid_path = tmp_path / "nonexistent.png"
        result = extract_bounding_boxes_heuristic(invalid_path, tmp_path, 1)
        assert result is None

    def test_create_overlay_error_handling(self, tmp_path):
        """Test error handling for invalid inputs."""
        invalid_img = tmp_path / "nonexistent.png"
        invalid_json = tmp_path / "nonexistent.json"
        result = create_overlay(invalid_img, invalid_json, tmp_path, 1)
        assert result is None


@pytest.mark.integration
class TestIntegration:
    """Integration tests combining multiple features."""

    def test_full_pipeline_simulation(self, tmp_path):
        """Test simulating a full processing pipeline."""
        # Simulate OCR output with tables and math
        ocr_text = """
# Test Document

Here's a table:

| Name | Value |
|------|-------|
| X    | 10    |
| Y    | 20    |

And some math: $E = mc^2$

```
def example():
    import sys
    print("hello")
```
"""

        # Extract tables
        tables = extract_tables_to_csv(ocr_text, tmp_path, 1, "csv")
        assert len(tables) == 1

        # Extract math
        equations = extract_math(ocr_text, tmp_path, 1)
        assert len(equations) == 1
        assert "mc^2" in equations[0].read_text()

        # Tag code
        tagged_text = tag_code_languages(ocr_text)
        assert "```python" in tagged_text

        # Emit chunks
        chunks_path = tmp_path / "chunks.jsonl"
        emit_chunks(chunks_path, 1, ocr_text)
        assert chunks_path.exists()

        # Count words
        words = word_count(ocr_text)
        assert words > 10
