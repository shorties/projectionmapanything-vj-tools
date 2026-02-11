#!/bin/bash
# Install ProjectionMapAnything VJ.Tools on RunPod

set -e

echo "=== ProjectionMapAnything VJ.Tools Installer ==="
echo

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo -e "${YELLOW}Note: Not running as root, may need sudo for some operations${NC}"
fi

echo "[1/6] Finding Python environment..."

# Find Python
PYTHON_CMD=""
if command -v python3.12 &> /dev/null; then
    PYTHON_CMD="python3.12"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo -e "${RED}Error: Python not found${NC}"
    exit 1
fi

echo "Found Python: $PYTHON_CMD ($($PYTHON_CMD --version))"

# Check for uv
if ! command -v uv &> /dev/null; then
    echo -e "${RED}Error: uv not found${NC}"
    exit 1
fi

echo "Found uv: $(uv --version)"

echo
echo "[2/6] Setting up plugin directory..."

PLUGIN_DIR="/root/projectionmapanything-vj-tools"
if [ ! -d "$PLUGIN_DIR" ]; then
    echo "Cloning from GitHub..."
    git clone https://github.com/shorties/projectionmapanything-vj-tools.git "$PLUGIN_DIR"
else
    echo "Plugin directory exists, pulling latest..."
    cd "$PLUGIN_DIR"
    git pull
fi

echo
echo "[3/6] Installing dependencies..."

# Try different installation methods
if uv pip install --system opencv-python transformers pillow 2>/dev/null; then
    echo -e "${GREEN}Dependencies installed with uv --system${NC}"
else
    echo -e "${YELLOW}Trying alternative method...${NC}"
    $PYTHON_CMD -m pip install --break-system-packages opencv-python transformers pillow 2>/dev/null || \
    $PYTHON_CMD -m pip install opencv-python transformers pillow
fi

echo
echo "[4/6] Installing plugin..."

# Clear broken cache
rm -f /root/.daydream-scope/plugins/resolved.txt

# Install plugin
if uv pip install --system -e "$PLUGIN_DIR" 2>/dev/null; then
    echo -e "${GREEN}Plugin installed with uv --system${NC}"
else
    echo -e "${YELLOW}Trying pip install...${NC}"
    $PYTHON_CMD -m pip install --break-system-packages -e "$PLUGIN_DIR" 2>/dev/null || \
    $PYTHON_CMD -m pip install -e "$PLUGIN_DIR"
fi

echo
echo "[5/6] Verifying installation..."

# Test import
if $PYTHON_CMD -c "from scope_depth.plugin import register_pipelines" 2>/dev/null; then
    echo -e "${GREEN}Plugin imports successfully${NC}"
else
    echo -e "${RED}Plugin import failed${NC}"
    exit 1
fi

# Check entry point
ENTRY_POINTS=$($PYTHON_CMD -c "from importlib.metadata import entry_points; eps = entry_points(group='scope'); print([e.name for e in eps])" 2>/dev/null || echo "[]")
echo "Scope entry points: $ENTRY_POINTS"

if echo "$ENTRY_POINTS" | grep -q "scope_depth"; then
    echo -e "${GREEN}Entry point registered: scope_depth${NC}"
else
    echo -e "${YELLOW}Warning: Entry point not found in metadata${NC}"
    echo "Trying alternative registration..."
    
    # Force reinstall
    uv pip install --system --force-reinstall -e "$PLUGIN_DIR" 2>/dev/null || \
    $PYTHON_CMD -m pip install --force-reinstall -e "$PLUGIN_DIR"
fi

echo
echo "[6/6] Testing pipeline..."

$PYTHON_CMD << 'PYEOF'
try:
    from scope_depth.pipelines import DepthWebcamPipeline
    p = DepthWebcamPipeline()
    config_class = p.get_config_class()
    config = config_class()
    print(f"Pipeline ID: {config.pipeline_id}")
    print(f"Pipeline Name: {config.pipeline_name}")
    print("✓ Pipeline configuration OK")
except Exception as e:
    print(f"✗ Pipeline test failed: {e}")
    import traceback
    traceback.print_exc()
PYEOF

echo
echo "=================================="
echo -e "${GREEN}Installation complete!${NC}"
echo
echo "Next steps:"
echo "1. Restart Daydream Scope completely"
echo "2. Check pipeline dropdown for 'ProjectionMapAnything VJ.Tools'"
echo
echo "If not appearing, check Scope logs:"
echo "  uv run daydream-scope 2>&1 | grep -i error"
echo "=================================="
