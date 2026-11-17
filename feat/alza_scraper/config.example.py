# Zkopírujte jako config.py a doplňte vlastní nastavení.
"""Všechna nastavení generátoru; žádné přepínače příkazové řádky."""
ACTIVE_PROVIDER = "claude"  # gemini, groq, ollama, openai, claude
SENTENCES_PER_INTENT = 10   # Nejméně 10; rozdělení mezi pět stylů.
MIN_WORDS_PER_MESSAGE = 15  # Jedna položka je celá zákaznická zpráva.
MAX_WORDS_PER_MESSAGE = 60  # Přibližně 1–3 věty, od chatu po krátký e-mail.

import os
from pathlib import Path

# Přednost má neprázdná hodnota z prostředí; jinak vyplňte řetězec za „or“.

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or ""
GROQ_API_KEY = os.environ.get("GROQ_API_KEY") or ""
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or ""
CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY") or ""
CLAUDE_WORKSPACE_ID = os.environ.get("CLAUDE_WORKSPACE_ID") or ""  # Jen pokud váš typ klíče vyžaduje výběr workspace.

PROVIDERS = {
    "gemini": {
        "model": "gemini-3.8-flash",  # Gemini 2.5 nemusí být pro nové projekty dostupné.
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key": GEMINI_API_KEY,
        "requires_key": True,
        "json_mode": True,
    },
    "groq": {
        "model": "llama-3.3-70b-versatile",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key": GROQ_API_KEY,
        "requires_key": True,
        "json_mode": True,
    },
    "ollama": {
        "model": "qwen2.5:7b",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",  # SDK vyžaduje neprázdný text, lokální server klíč neověřuje.
        "requires_key": False,
        "json_mode": True,
    },
    "openai": {
        "model": "gpt-4o-mini",
        "base_url": "https://api.openai.com/v1",
        "api_key": OPENAI_API_KEY,
        "requires_key": True,
        "json_mode": True,
    },
    "claude": {
        "model": "claude-haiku-5-5",
        "base_url": "https://api.anthropic.com/v1/",
        "api_key": CLAUDE_API_KEY,
        "requires_key": True,
        "json_mode": False,  # Anthropic compatibility response_format ignoruje.
        "max_tokens": 8192,
        "default_headers": ({"anthropic-workspace-id": CLAUDE_WORKSPACE_ID}
                            if CLAUDE_WORKSPACE_ID.strip() else {}),
    },
}

ROOT = Path(__file__).resolve().parent
INTENTS_FILE = ROOT / "intents_tree.py"
OUTPUT_DIR = ROOT / "datasets"
OVERWRITE = True           # True dovolí nahradit výsledek stejného modelu.
MAX_GENERATION_ROUNDS = 99999999   # Doplňování neplatných nebo duplicitních vět.
REQUEST_TIMEOUT = 600.0     # V sekundách; vhodné i pro pomalejší lokální model.
API_RETRIES = 2             # Opakování dočasných síťových/API chyb uvnitř SDK.
REQUEST_PAUSE = 1.0
