#!/bin/bash
echo "=== GPU Status ==="
nvidia-smi 2>&1 | head -25
echo ""
echo "=== Python ==="
python3 -c "import sys; print(sys.executable, sys.version)"
echo ""
echo "=== Installed packages ==="
pip3 list 2>&1 | head -50
echo ""
echo "=== Find numpy ==="
find /opt /usr /home -name "numpy" -type d 2>/dev/null | head -10
echo ""
echo "=== Check CUDA ==="
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count()); print('Device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')" 2>&1