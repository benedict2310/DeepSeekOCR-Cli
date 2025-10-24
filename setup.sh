#!/bin/bash
# DeepSeek-OCR Mac CLI Setup Script
# For macOS (Apple Silicon recommended)

set -e  # Exit on error

echo "🚀 DeepSeek-OCR Mac CLI Setup"
echo "================================"
echo ""

# Check if on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "⚠️  Warning: This tool is optimized for macOS."
    echo "   You may encounter issues on other platforms."
    echo ""
fi

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.9"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3,9) else 1)"; then
    echo "❌ Python 3.9+ required. Found: $python_version"
    echo "   Please install Python 3.9 or later."
    exit 1
fi

echo "✅ Python $python_version found"
echo ""

# Create virtual environment
echo "📦 Creating virtual environment..."
if [ -d ".venv" ]; then
    echo "   .venv already exists, skipping..."
else
    python3 -m venv .venv
    echo "✅ Virtual environment created"
fi
echo ""

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip -q

# Install dependencies
echo "📥 Installing dependencies..."
echo "   This may take several minutes..."
pip install -r requirements.txt -q

echo "✅ Dependencies installed"
echo ""

# Make CLI executable
echo "🔧 Making CLI executable..."
chmod +x deepseek_ocr_mac.py
echo "✅ CLI is now executable"
echo ""

# Check MPS availability
echo "🧪 Checking Metal Performance Shaders (MPS) support..."
mps_available=$(python3 -c "import torch; print('yes' if torch.backends.mps.is_available() else 'no')" 2>/dev/null || echo "no")

if [ "$mps_available" = "yes" ]; then
    echo "✅ MPS acceleration available (Apple Silicon detected)"
else
    echo "⚠️  MPS not available - will use CPU mode"
    echo "   (This is normal on Intel Macs)"
fi
echo ""

# Create outputs directory
mkdir -p outputs

echo "================================"
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Activate the virtual environment:"
echo "     source .venv/bin/activate"
echo ""
echo "  2. Run OCR on a file:"
echo "     ./deepseek_ocr_mac.py your_file.pdf"
echo ""
echo "  3. (Optional) Install globally:"
echo "     sudo ln -s \"$(pwd)/deepseek_ocr_mac.py\" /usr/local/bin/deepseek-ocr"
echo ""
echo "For help:"
echo "  ./deepseek_ocr_mac.py --help"
echo ""
