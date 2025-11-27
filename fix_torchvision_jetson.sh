#!/bin/bash
set -e

echo "======================================================================="
echo "🔧 Rebuilding torchvision with CUDA extensions for Jetson"
echo "======================================================================="

# Detect CUDA path
if [ -d "/usr/local/cuda-12.2" ]; then
    export CUDA_HOME=/usr/local/cuda-12.2
elif [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME=/usr/local/cuda
else
    echo "❌ CUDA not found"
    exit 1
fi

echo "✓ Using CUDA_HOME: $CUDA_HOME"

# Set LD_LIBRARY_PATH for CUDA libraries
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH

# Activate venv
VENV_DIR="venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "❌ Virtual environment not found"
    exit 1
fi

source "$VENV_DIR/bin/activate"

echo "✓ Virtual environment activated"

# Uninstall current torchvision
echo "🗑️  Uninstalling current torchvision..."
pip uninstall -y torchvision || true

# Install build dependencies
echo "📦 Installing build dependencies..."
pip install --upgrade pip setuptools wheel
pip install ninja

# Clone torchvision (use compatible version for PyTorch 2.4.0)
echo "📥 Cloning torchvision..."
WORK_DIR=$(mktemp -d)
cd "$WORK_DIR"

# Use v0.19.0 which is compatible with PyTorch 2.4.0
git clone --branch v0.19.0 --depth 1 https://github.com/pytorch/vision.git
cd vision

echo "🔨 Building torchvision with CUDA extensions..."
echo "   This will take 15-30 minutes on Jetson..."

# Set environment variables for compilation
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="8.7"  # Jetson AGX Orin compute capability
export MAX_JOBS=4  # Limit parallel jobs to avoid OOM

# Build and install
python setup.py clean
python setup.py bdist_wheel

# Install the wheel
echo "📦 Installing torchvision wheel..."
pip install dist/*.whl

# Cleanup
cd -
rm -rf "$WORK_DIR"

echo ""
echo "======================================================================="
echo "✅ Torchvision rebuild complete!"
echo "======================================================================="
echo ""
echo "🧪 Testing torchvision installation..."
python -c "
import torch
import torchvision
print(f'torch version: {torch.__version__}')
print(f'torchvision version: {torchvision.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

# Test nms operation
try:
    from torchvision.ops import nms
    import torch
    boxes = torch.tensor([[0, 0, 10, 10], [5, 5, 15, 15]], dtype=torch.float32)
    scores = torch.tensor([0.9, 0.8], dtype=torch.float32)
    keep = nms(boxes, scores, 0.5)
    print(f'✅ NMS operation works! Result: {keep}')
except Exception as e:
    print(f'❌ NMS operation failed: {e}')
"

echo ""
echo "🎉 All done! Try running 'python3 app.py' now."
