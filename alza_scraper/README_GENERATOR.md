# Generátor datasetu z intentů

Ve stejné složce musí být `config.py`, `generate_dataset.py` a váš
`intents_tree.py` s doslovným slovníkem `ALZA_INTENTS`. Strom se nemění,
URL se nestahují. Není potřeba scraper ani jiná část předchozího projektu.

## Nastavení

Při novém klonování nejdřív vytvořte lokální konfiguraci:

```bash
cp config.example.py config.py
```

`config.py`, `.env`, prostředí `.venv`, cache a generované `datasets/`
jsou v `.gitignore`. Do Gitu patří `config.example.py` bez vyplněných klíčů.

V `config.py` upravte:

```python
ACTIVE_PROVIDER = "gemini"
GEMINI_API_KEY = "vas-klic"
SENTENCES_PER_INTENT = 10
MIN_WORDS_PER_MESSAGE = 15
MAX_WORDS_PER_MESSAGE = 60
```

Další možnosti jsou `groq`, `ollama`, `openai` a `claude`. Model a endpoint
změníte v příslušné položce `PROVIDERS`. Klíče nepřidávejte do veřejného repozitáře.

Výchozí model Gemini je `gemini-3.8-flash`. Podle [dokumentace Googlu](https://ai.google.dev/gemini-api/docs/deprecations)
je přístup k řadě 2.5 omezen na dosavadní uživatele. HTTP 404 u `gemini-2.5-flash`
proto nemusí znamenat chybnou URL. Kompatibilní endpoint zůstává
`https://generativelanguage.googleapis.com/v1beta/openai/`.
Před generováním skript ověří vybraný Gemini model přes `client.models.list()`.
Pokud chybí, vypíše dostupné textové modely pro daný klíč; model automaticky nemění.
Při chybě API vypíše také zprávu poskytovatele s odstraněním nakonfigurovaných klíčů.

- [Gemini klíč](https://aistudio.google.com/apikey)
- [Groq klíč](https://console.groq.com/keys)
- [OpenAI klíč](https://platform.openai.com/api-keys)

Pro lokální Ollamu stačí zvolit `ACTIVE_PROVIDER = "ollama"`, po instalaci
Ollamy stáhnout `ollama pull qwen2.5:7b` a zajistit běh `ollama serve`.
Po stažení modelu a instalaci závislostí může generování probíhat offline.

Pro Claude je místo původního `custom` připraven poskytovatel `claude`:

```python
ACTIVE_PROVIDER = "claude"
CLAUDE_API_KEY = "vas-anthropic-api-klic"
```

Klíč vytvořte v [Claude Console](https://platform.claude.com/).
Výchozí `claude-haiku-5-5` je rychlá úsporná varianta pro hromadnou tvorbu zpráv;
model můžete změnit v `PROVIDERS["claude"]["model"]`.
Používá se [oficiální OpenAI kompatibilita Anthropic](https://platform.claude.com/docs/en/cli-sdks-libraries/libraries/openai-sdk)
a endpoint `https://api.anthropic.com/v1/`, další knihovna není potřeba.
Toto rozhraní ignoruje `response_format`, proto je `json_mode = False`.
JSON se vyžaduje promptem a validuje dosavadním parserem. Limit odpovědi je
8192 tokenů; pro vyšší počty zpráv lze upravit `max_tokens`.
Pokud váš klíč vyžaduje výběr workspace, vyplňte také `CLAUDE_WORKSPACE_ID`.
Výsledky budou v `dataset_claude_claude-haiku-5-5.jsonl`.
Model a jeho zaměření jsou uvedeny v [přehledu modelů Anthropic](https://platform.claude.com/docs/en/models/overview).

## Spuštění

Python 3.10 nebo novější:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install 'openai>=1.55,<3'
python generate_dataset.py
```

Alternativně bez ruční instalace knihovny:

```bash
uv run generate_dataset.py
```

Není potřeba předávat žádné argumenty. Chybějící klíč způsobí krátkou hlášku
před síťovým požadavkem. Cloudové běhy se řídí cenami a limity vybrané služby.

## Výsledky

Výstup je ve složce `datasets` vedle skriptu, například:

- `dataset_gemini_gemini-3.8-flash.jsonl`
- `dataset_groq_llama-3.3-70b-versatile.jsonl`
- `dataset_ollama_qwen2.5%3A7b.jsonl`
- `dataset_openai_gpt-4o-mini.jsonl`

Zvláštní znaky se escapují, aby modely s různými názvy nekolidovaly.
Každý řádek má přesně pole `text`, `intent_id`, `main_intent`, `provider`, `model`.
`main_intent` je klíč hlavní kategorie, například `orders_and_payments`.

Každá položka je celá zákaznická zpráva, obvykle o 1–3 větách a 15–60 slovech.
Rozsah lze změnit pomocí `MIN_WORDS_PER_MESSAGE` a `MAX_WORDS_PER_MESSAGE`.
Původní název `SENTENCES_PER_INTENT` je ponechán pro nastavení počtu zpráv.
Pro 10 zpráv vzniknou dvě zprávy na každý z pěti stylů: formální, hovorový,
bez diakritiky, běžný neutrální chat (`conversational`) a urgentní.
Telegrafická kategorie `short` je nahrazena běžnou chatovou zprávou.
Vyšší počty se rozdělí
co nejrovnoměrněji. Skript prochází rekurzivně jen koncové podtřídy.
Podporuje `sub_intents` i starší `children`; duplicitní ID odmítne.
Názvy, popisy i cesta hlavní kategorií se posílají modelu jako kontext.

Normalizované duplicity uvnitř intentu se odmítají i při změně diakritiky,
interpunkce nebo čísel. Všechny styly musí splnit nastavený rozsah slov.
Delší zprávy mohou nabídnout více různých situací, ale nezaručují odstranění duplicit.
Pokud model vrátí text se špatnou délkou, ne-ASCII znaky v kategorii bez diakritiky
nebo duplicitu, skript vypíše důvod a počet odmítnutých zpráv pro daný styl a pokus.
Poškozený JSON, včetně neplatných struktur, vyvolá nový pokus. Markdownové
obaly ```json se odstraňují. Neúplné odpovědi kvůli limitu délky se odmítají.

Po vyčerpání pokusů se generátor zastaví místo exportu neúplného datasetu.
Dokončené intenty zůstanou v souboru `.partial.jsonl`; automatické pokračování
není implementováno. Finální název souboru vznikne až po kompletním běhu.
Existující výsledky se nepřepisují; pro opakování stejného modelu je přesuňte
nebo nastavte `OVERWRITE = True` (nahradí také starou průběžnou zálohu).

Kontrola stylu a sémantické jedinečnosti není úplná: výsledky vyžadují audit.
Některé podtřídy z vašeho stromu mají překrývající se význam (např. pronájem
nebo nákup na třetinku ve dvou kategoriích). Zachovávají se původní ID;
samotná věta nemusí vždy umožnit spolehlivě rozlišit takové štítky.

Implementace používá knihovnu `openai` a kompatibilitu popsanou v dokumentaci
[Gemini](https://ai.google.dev/gemini-api/docs/openai),
[Groq](https://console.groq.com/docs/openai),
[Ollamy](https://docs.ollama.com/api/openai-compatibility)
a [OpenAI](https://developers.openai.com/api/docs/guides/structured-outputs).
