# Colorization
## Setup
### Windows
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_conda.ps1

### Linux/macOS
bash ./scripts/bootstrap_conda.sh

## Train
conda activate colorization
python -m colorization.train --config configs/train.yaml
