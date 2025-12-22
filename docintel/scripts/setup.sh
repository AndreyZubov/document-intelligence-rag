#!/bin/bash
# Quick setup script for DocIntel

set -e

echo "======================================"
echo "DocIntel Setup"
echo "======================================"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python 3.10 or higher required. Found: $python_version"
    exit 1
fi
echo "✓ Python $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip > /dev/null 2>&1
pip install -e . > /dev/null 2>&1
echo "✓ Dependencies installed"

# Create .env file
echo ""
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cp .env.example .env
    echo "✓ .env file created"
    echo ""
    echo "⚠️  IMPORTANT: Edit .env and add your API key:"
    echo "   - ANTHROPIC_API_KEY (for Claude)"
    echo "   - or OPENAI_API_KEY (for GPT)"
else
    echo "✓ .env file already exists"
fi

# Check Docker
echo ""
echo "Checking Docker..."
if command -v docker &> /dev/null; then
    echo "✓ Docker is installed"

    # Check if Qdrant is running
    if curl -s http://localhost:6333/health > /dev/null 2>&1; then
        echo "✓ Qdrant is already running"
    else
        echo ""
        echo "Starting Qdrant..."
        docker run -d -p 6333:6333 -p 6334:6334 --name qdrant qdrant/qdrant > /dev/null 2>&1
        echo "✓ Qdrant started"
    fi
else
    echo "⚠️  Docker not found. Install Docker to run Qdrant."
    echo "   Or use: docker-compose up -d"
fi

echo ""
echo "======================================"
echo "Setup Complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo "1. Edit .env and add your API key"
echo "2. Run: source venv/bin/activate"
echo "3. Test: docintel health"
echo "4. Upload: docintel upload document.pdf"
echo "5. Query: docintel query \"Your question?\""
echo ""
echo "Or start the API server:"
echo "  docintel serve"
echo ""
echo "Documentation: http://localhost:8000/docs"
echo ""
