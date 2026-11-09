#!/usr/bin/env bash
set -euo pipefail

# Spusť pomocí: bash setup_venv.sh
# Cestu odvodíme od tohoto skriptu, takže nezáleží na aktuální složce.
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
venv_dir="$project_dir/.venv"

if ! command -v uv >/dev/null 2>&1; then
    echo "Chybí uv. Nejprve ho nainstaluj a znovu spusť tento skript." >&2
    exit 1
fi

# Jedno prostředí pro celý projekt. Při opakování existující .venv zachováme.
# Pokud Python 3.11 není dostupný, uv ho může automaticky stáhnout.
if [[ ! -f "$venv_dir/pyvenv.cfg" ]]; then
    uv venv --python 3.11 "$venv_dir"
fi

# --python určuje přesný cíl instalace i bez aktivace prostředí.
# requests, beautifulsoup4: získávání dat z webu.
# openai: generování dat přes OpenAI a kompatibilní API.
# datasets: načítání veřejných datasetů z Hugging Face.
# numpy, scikit-learn: práce s vektory, dělení dat, k-NN a metriky.
# sentence-transformers, transformers, torch: načtení embedding modelu a inference.
# huggingface-hub: přístup k modelům a příkaz hf auth login.
uv pip install --python "$venv_dir/bin/python" \
    requests \
    beautifulsoup4 \
    'openai>=1.55,<3' \
    datasets \
    numpy \
    'scikit-learn>=1.5,<2' \
    'sentence-transformers>=5.1,<6' \
    'transformers>=4.56,<6' \
    torch \
    huggingface-hub

# Aktivaci necháváme na uživateli; tento skript ji neprovádí.
echo "Hotovo. Společné prostředí: $venv_dir"
printf 'Aktivace v tvém terminálu: source "%s/bin/activate"\n' "$venv_dir"
