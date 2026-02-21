Param(
  [string]$EnvName = "colorization",
  [string]$EnvFile = "environment.yml"
)

$ErrorActionPreference = "Stop"

# 1) check conda
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
  throw "conda not found. Install Miniconda/Anaconda first."
}

# 2) create env if not exists
$envList = conda env list
if ($envList -notmatch "^\s*$EnvName\s") {
  Write-Host "Creating conda env: $EnvName ..."
  conda env create -n $EnvName -f $EnvFile
} else {
  Write-Host "Env $EnvName already exists. Updating ..."
  conda env update -n $EnvName -f $EnvFile --prune
}

# 3) activate + editable install
Write-Host "Activating $EnvName and installing editable package..."
conda run -n $EnvName python -m pip install -U pip
conda run -n $EnvName python -m pip install -e .

Write-Host "Done."

Write-Host "To activate the environment, run: conda activate $EnvName"
