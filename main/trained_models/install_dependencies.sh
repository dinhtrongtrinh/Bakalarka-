#!/usr/bin/env bash
set -euo pipefail

# Spuštění: bash main/trained_models/install_dependencies.sh
# Použijeme stejné .venv jako ve zbytku bakalářky, nic neaktivujeme.
current_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$current_dir/../.." && pwd)"
python_path="$project_dir/.venv/bin/python"

if [[ ! -x "$python_path" ]]; then
    echo "Chybí společné .venv. Nejdřív spusť setup_venv.sh v kořeni projektu."
    exit 1
fi

# Unsloth přidá závislosti pro LoRA. Instalační nástroj ověří jejich kompatibilitu.
# Nepoužíváme --no-deps, aby nevznikla neúplná instalace.
uv pip install --python "$python_path" \
    unsloth \
    'sentence-transformers>=5.7,<6' \
    datasets \
    matplotlib \
    numpy \
    'scikit-learn>=1.5,<2' \
    joblib

# PyPI může nabídnout torchvision pro jinou CUDA než již nainstalovaný torch.
# Verzi torchvision ponecháme takovou, jakou vyřešil uv, ale jeho binární
# sestavení stáhneme ze stejné CUDA větve oficiálního serveru PyTorch.
torchvision_version="$("$python_path" -c 'from importlib.metadata import version; print(version("torchvision").split("+")[0])')"
pytorch_cuda_tag="$("$python_path" -c 'import torch; print("cu" + torch.version.cuda.replace(".", "") if torch.version.cuda else "cpu")')"
pytorch_index_url="https://download.pytorch.org/whl/$pytorch_cuda_tag"

uv pip install --python "$python_path" \
    --reinstall \
    --no-deps \
    --index-url "$pytorch_index_url" \
    "torchvision==$torchvision_version"

echo "Instalace dokončena. Prostředí se neaktivovalo a trénink se nespustil."
