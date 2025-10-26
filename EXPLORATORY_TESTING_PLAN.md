# Exploratory Testing Plan - DeepSeek-OCR Extension Features

**Version:** 2.0
**Date:** 2025-10-26
**Purpose:** Hands-on testing guide for all new extension features

---

## Prerequisites

Before starting, ensure:
- ✅ Virtual environment is activated: `source .venv/bin/activate`
- ✅ All dependencies installed: `pip install -r requirements.txt`
- ✅ CLI is executable: `chmod +x deepseek_ocr_mac.py`
- ✅ Test document ready (PDF or image with tables, math, code)

---

## Test Plan Overview

| # | Feature | Time | Priority |
|---|---------|------|----------|
| 1 | Basic functionality (regression test) | 5 min | High |
| 2 | Compression presets | 10 min | High |
| 3 | Table extraction | 5 min | High |
| 4 | Math extraction | 5 min | Medium |
| 5 | Code language detection | 5 min | Medium |
| 6 | RAG chunks generation | 5 min | High |
| 7 | Quality gates & strict mode | 5 min | High |
| 8 | Bounding boxes & overlays | 10 min | Low |
| 9 | Combined features test | 10 min | High |

**Total Time:** ~60 minutes

---

## Test 1: Basic Functionality (Regression Test)

**Purpose:** Verify existing features still work after adding extensions.

### Steps

1. **Verify help works:**
   ```bash
   ./deepseek_ocr_mac.py --help
   ```
   **Expected:** Help text displays with all new flags listed

2. **Test basic image OCR:**
   ```bash
   # Create a simple test image
   python3 -c "
   from tests.fixtures import create_test_image
   from pathlib import Path
   create_test_image(Path('test_basic.png'), 'Hello World Test Document')
   "

   # Run OCR
   ./deepseek_ocr_mac.py test_basic.png
   ```
   **Expected:**
   - ✅ "Using device: mps" or "Using device: cpu"
   - ✅ "Loading model…"
   - ✅ "Found 1 page(s)."
   - ✅ "→ OCR [1/1] test_basic.png"
   - ✅ "✅ Done! Markdown saved to..."
   - ✅ `outputs/merged_output.md` exists

3. **Check output structure:**
   ```bash
   cat outputs/merged_output.md
   ```
   **Expected:**
   - ✅ Contains `# Page 1`
   - ✅ Contains quality summary with YAML frontmatter:
     ```yaml
     ---
     quality:
       pages: 1
       success_rate: 100.0%
       ...
     ---
     ```

4. **Clean up:**
   ```bash
   rm test_basic.png
   rm -rf outputs/
   ```

**Result:** ☐ PASS  ☐ FAIL
**Notes:**

---

## Test 2: Compression Presets

**Purpose:** Verify the three compression presets work correctly.

### Steps

1. **Test HIGH compression (fastest):**
   ```bash
   time ./deepseek_ocr_mac.py test_basic.png --compression high
   ```
   **Expected:**
   - ✅ Completes faster than default
   - ✅ Quality summary shows `compression: high`

2. **Test MED compression (balanced):**
   ```bash
   rm -rf outputs/
   time ./deepseek_ocr_mac.py test_basic.png --compression med
   ```
   **Expected:**
   - ✅ Quality summary shows `compression: med`

3. **Test LOW compression (best quality - default):**
   ```bash
   rm -rf outputs/
   time ./deepseek_ocr_mac.py test_basic.png --compression low
   ```
   **Expected:**
   - ✅ Takes longest time
   - ✅ Quality summary shows `compression: low`
   - ✅ Best quality output

4. **Test manual override:**
   ```bash
   rm -rf outputs/
   ./deepseek_ocr_mac.py test_basic.png --compression high --base-size 2048
   ```
   **Expected:**
   - ✅ Uses custom base-size instead of preset value
   - ✅ Still shows `compression: high` in stats

**Result:** ☐ PASS  ☐ FAIL
**Notes:**

---

## Test 3: Table Extraction

**Purpose:** Verify Markdown table extraction to CSV/TSV.

### Setup

Create a test document with tables:
```bash
python3 << 'EOF'
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

img = Image.new('RGB', (800, 400), color='white')
draw = ImageDraw.Draw(img)

# Draw a simple table representation
text = """
Product | Price
--------|------
Apple   | $1.50
Banana  | $0.75
"""

draw.text((50, 50), text, fill='black')
img.save('test_table.png')
print("✓ Created test_table.png")
EOF
```

### Steps

1. **Extract to CSV:**
   ```bash
   rm -rf outputs/
   ./deepseek_ocr_mac.py test_table.png --emit-csv
   ```
   **Expected:**
   - ✅ Console shows: "📊 Extracted N table(s)"
   - ✅ `outputs/tables/` directory exists
   - ✅ CSV file(s) created

2. **Verify CSV content:**
   ```bash
   ls -la outputs/tables/
   cat outputs/tables/page_0001_table_1.csv
   ```
   **Expected:**
   - ✅ Valid CSV format
   - ✅ Comma-separated values

3. **Test TSV format:**
   ```bash
   rm -rf outputs/
   ./deepseek_ocr_mac.py test_table.png --emit-csv=tsv
   ```
   **Expected:**
   - ✅ `outputs/tables/page_0001_table_1.tsv` created
   - ✅ Tab-separated values

4. **Import test (optional):**
   ```bash
   # Try importing into Python pandas
   python3 -c "
   import pandas as pd
   df = pd.read_csv('outputs/tables/page_0001_table_1.csv')
   print(df)
   "
   ```

**Result:** ☐ PASS  ☐ FAIL
**Notes:**

---

## Test 4: Math Extraction

**Purpose:** Verify LaTeX equation extraction.

### Setup

If you have a PDF with math equations, use that. Otherwise, skip or create a mock:

```bash
# Create a test file with LaTeX-like text
echo "# Math Document

The quadratic formula is: \$x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}\$

Display equation:
\$\$\\int_0^\\infty e^{-x^2} dx = \\frac{\\sqrt{\\pi}}{2}\$\$
" > test_math.md

# Convert to image (simplified - you could use actual PDF)
python3 << 'EOF'
from PIL import Image, ImageDraw
img = Image.new('RGB', (800, 600), 'white')
draw = ImageDraw.Draw(img)
draw.text((50, 50), "x = (a + b) / 2", fill='black')
draw.text((50, 100), "∫₀^∞ e^(-x²) dx", fill='black')
img.save('test_math.png')
print("✓ Created test_math.png")
EOF
```

### Steps

1. **Extract LaTeX:**
   ```bash
   rm -rf outputs/
   ./deepseek_ocr_mac.py test_math.png --math-latex
   ```
   **Expected:**
   - ✅ Console shows: "🧮 Extracted N equation(s)"
   - ✅ `outputs/equations/` directory exists

2. **Verify equations:**
   ```bash
   ls -la outputs/equations/
   cat outputs/equations/page_0001_eq_001.tex
   ```
   **Expected:**
   - ✅ `.tex` files created
   - ✅ Contains LaTeX expressions (without $$ delimiters)

3. **Test equation rendering (optional):**
   ```bash
   # If you have LaTeX installed
   cd outputs/equations/
   echo "\\documentclass{article}\\begin{document}\$\$(cat page_0001_eq_001.tex)\$\$\\end{document}" > test.tex
   pdflatex test.tex  # Should render if LaTeX is installed
   ```

**Result:** ☐ PASS  ☐ FAIL
**Notes:**

---

## Test 5: Code Language Detection

**Purpose:** Verify auto-tagging of code blocks.

### Setup

```bash
python3 << 'EOF'
from PIL import Image, ImageDraw
img = Image.new('RGB', (1000, 600), 'white')
draw = ImageDraw.Draw(img)

code = """def hello():
    import os
    return 42

function foo() {
    const x = 5;
    return x;
}"""

draw.text((50, 50), code, fill='black')
img.save('test_code.png')
print("✓ Created test_code.png with Python and JavaScript code")
EOF
```

### Steps

1. **Run with code language detection:**
   ```bash
   rm -rf outputs/
   ./deepseek_ocr_mac.py test_code.png --code-lang
   ```

2. **Check tagged output:**
   ```bash
   cat outputs/merged_output.md
   ```
   **Expected:**
   - ✅ Code blocks tagged with language: ` ```python` or ` ```javascript`
   - ✅ Not ` ``` ` (blank)

3. **Test all supported languages (optional):**
   Languages to verify: python, javascript, typescript, java, cpp, go, rust, sql, bash, json

**Result:** ☐ PASS  ☐ FAIL
**Notes:**

---

## Test 6: RAG Chunks Generation

**Purpose:** Verify JSONL chunk generation for vector databases.

### Steps

1. **Generate chunks:**
   ```bash
   rm -rf outputs/
   ./deepseek_ocr_mac.py test_basic.png --emit-chunks
   ```
   **Expected:**
   - ✅ `outputs/chunks.jsonl` exists

2. **Verify JSONL format:**
   ```bash
   cat outputs/chunks.jsonl
   ```
   **Expected:**
   - ✅ Each line is valid JSON
   - ✅ Contains keys: `page`, `start`, `end`, `text`
   - ✅ Chunk size ~1200 characters max

3. **Parse and validate:**
   ```bash
   python3 << 'EOF'
   import json

   with open('outputs/chunks.jsonl') as f:
       for i, line in enumerate(f, 1):
           chunk = json.loads(line)
           print(f"Chunk {i}:")
           print(f"  Page: {chunk['page']}")
           print(f"  Size: {chunk['end'] - chunk['start']}")
           print(f"  Text preview: {chunk['text'][:50]}...")
           assert 'page' in chunk
           assert 'text' in chunk
           assert len(chunk['text']) <= 1200

   print("✓ All chunks valid!")
   EOF
   ```

4. **Test with longer document:**
   ```bash
   # Create a document with >1200 characters
   python3 << 'EOF'
   from PIL import Image, ImageDraw
   img = Image.new('RGB', (800, 1200), 'white')
   draw = ImageDraw.Draw(img)

   long_text = "Lorem ipsum dolor sit amet. " * 100  # ~2800 chars
   draw.text((50, 50), long_text[:500], fill='black')
   img.save('test_long.png')
   print("✓ Created test_long.png")
   EOF

   rm -rf outputs/
   ./deepseek_ocr_mac.py test_long.png --emit-chunks
   wc -l outputs/chunks.jsonl  # Should show multiple chunks
   ```

**Result:** ☐ PASS  ☐ FAIL
**Notes:**

---

## Test 7: Quality Gates & Strict Mode

**Purpose:** Verify quality checks and exit codes.

### Steps

1. **Test successful quality check:**
   ```bash
   rm -rf outputs/
   ./deepseek_ocr_mac.py test_basic.png --strict --min-words 5
   echo "Exit code: $?"
   ```
   **Expected:**
   - ✅ Exit code 0 (success)
   - ✅ Console shows success rate: 100.0%

2. **Test failed quality check:**
   ```bash
   rm -rf outputs/
   ./deepseek_ocr_mac.py test_basic.png --strict --min-words 1000
   echo "Exit code: $?"
   ```
   **Expected:**
   - ✅ Console shows: "⚠️ FAILED: Only N words (minimum 1000)"
   - ✅ Console shows: "❌ STRICT MODE: 1 pages failed quality check"
   - ✅ Exit code 1 (failure)

3. **Test without strict mode:**
   ```bash
   rm -rf outputs/
   ./deepseek_ocr_mac.py test_basic.png --min-words 1000  # No --strict
   echo "Exit code: $?"
   ```
   **Expected:**
   - ✅ Warning shown but exit code 0
   - ✅ Processing completes normally

4. **Check quality summary:**
   ```bash
   tail -15 outputs/merged_output.md
   ```
   **Expected:**
   - ✅ YAML with `failed_pages`, `success_rate`, `total_words`

**Result:** ☐ PASS  ☐ FAIL
**Notes:**

---

## Test 8: Bounding Boxes & Overlays

**Purpose:** Verify heuristic bounding box extraction and visual overlays.

### Steps

1. **Extract bounding boxes only:**
   ```bash
   rm -rf outputs/
   ./deepseek_ocr_mac.py test_basic.png --emit-boxes
   ```
   **Expected:**
   - ✅ Console shows: "🔲 Created bounding boxes"
   - ✅ `outputs/boxes/page_0001.json` exists

2. **Verify JSON format:**
   ```bash
   cat outputs/boxes/page_0001.json
   ```
   **Expected:**
   - ✅ Valid JSON
   - ✅ Contains: `page`, `bbox_provider`, `boxes`
   - ✅ Boxes have: `x`, `y`, `w`, `h`, `hint`

3. **Test with overlay:**
   ```bash
   rm -rf outputs/
   ./deepseek_ocr_mac.py test_basic.png --emit-boxes --overlay
   ```
   **Expected:**
   - ✅ Console shows: "🔲 Created bounding boxes and overlay"
   - ✅ `outputs/overlays/page_0001_overlay.png` exists

4. **View overlay:**
   ```bash
   open outputs/overlays/page_0001_overlay.png
   # or on Linux: xdg-open outputs/overlays/page_0001_overlay.png
   ```
   **Expected:**
   - ✅ Original image with red rectangles drawn on text regions

**Result:** ☐ PASS  ☐ FAIL
**Notes:**

---

## Test 9: Combined Features

**Purpose:** Verify all extensions work together.

### Steps

1. **Run with all features enabled:**
   ```bash
   rm -rf outputs/
   ./deepseek_ocr_mac.py test_table.png \
     --emit-csv \
     --math-latex \
     --code-lang \
     --emit-chunks \
     --emit-boxes --overlay \
     --compression low \
     --strict --min-words 5
   ```

2. **Verify all outputs created:**
   ```bash
   ls -R outputs/
   ```
   **Expected directory structure:**
   ```
   outputs/
   ├── merged_output.md
   ├── chunks.jsonl
   ├── tables/
   │   └── page_0001_table_*.csv
   ├── equations/
   │   └── page_0001_eq_*.tex (if math detected)
   ├── boxes/
   │   └── page_0001.json
   └── overlays/
       └── page_0001_overlay.png
   ```

3. **Verify console output:**
   **Expected to see:**
   - ✅ "📊 Extracted N table(s)"
   - ✅ "🧮 Extracted N equation(s)" (if any)
   - ✅ "🔲 Created bounding boxes and overlay"
   - ✅ Success rate: 100.0%

4. **Verify quality summary:**
   ```bash
   tail -20 outputs/merged_output.md
   ```
   **Expected:**
   - ✅ Complete YAML frontmatter with all stats

**Result:** ☐ PASS  ☐ FAIL
**Notes:**

---

## Test 10: Performance Comparison (Optional)

**Purpose:** Compare performance across compression presets.

### Steps

```bash
# Create a multi-page test document
python3 << 'EOF'
from tests.fixtures import create_test_pdf
from pathlib import Path
create_test_pdf(Path('test_multi.pdf'), num_pages=5)
print("✓ Created test_multi.pdf with 5 pages")
EOF

# Test each preset with timing
echo "=== HIGH compression ==="
rm -rf outputs/
time ./deepseek_ocr_mac.py test_multi.pdf --compression high

echo "=== MED compression ==="
rm -rf outputs/
time ./deepseek_ocr_mac.py test_multi.pdf --compression med

echo "=== LOW compression ==="
rm -rf outputs/
time ./deepseek_ocr_mac.py test_multi.pdf --compression low
```

**Expected:**
- ✅ HIGH fastest (~30% faster than LOW)
- ✅ LOW slowest but best quality
- ✅ MED balanced

**Result:** ☐ PASS  ☐ FAIL
**Notes:**

---

## Cleanup

After testing:

```bash
# Remove test files
rm test_*.png test_*.pdf test_*.md

# Remove outputs (or keep for inspection)
# rm -rf outputs/
```

---

## Summary Checklist

| Test | Status | Issues Found |
|------|--------|--------------|
| 1. Basic functionality | ☐ | |
| 2. Compression presets | ☐ | |
| 3. Table extraction | ☐ | |
| 4. Math extraction | ☐ | |
| 5. Code language detection | ☐ | |
| 6. RAG chunks | ☐ | |
| 7. Quality gates | ☐ | |
| 8. Bounding boxes | ☐ | |
| 9. Combined features | ☐ | |

---

## Reporting Issues

If you find bugs, please report with:
1. **Test number** that failed
2. **Command executed**
3. **Expected vs actual behavior**
4. **Console output** (copy/paste)
5. **System info**: macOS version, Python version, MPS availability

---

## Additional Exploration Ideas

1. **Real-world Documents:**
   - Try on academic papers (math + tables)
   - Try on technical documentation (code blocks)
   - Try on scanned receipts (tables)

2. **Integration Testing:**
   - Import chunks into a vector database (Pinecone, Chroma)
   - Load CSVs into Excel/Pandas
   - Render equations in LaTeX

3. **Stress Testing:**
   - Large PDFs (50+ pages)
   - High-resolution scans
   - Multiple concurrent runs

4. **Edge Cases:**
   - Empty documents
   - Documents with no tables/math/code
   - Documents with special characters
   - Non-English documents

---

**Happy Testing! 🎉**
