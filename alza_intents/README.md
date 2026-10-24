# Generátor Alza intentů

Python 3.10+. Vše nastavujete v `config.py`; skript nemá přepínače příkazové řádky.

## Instalace a spuštění

```bash
cd /home/martin/Bakalarka-/alza_intents
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python alza_dataset.py
```

Před prvním spuštěním vložte klíč do `config.py`:

```python
ACTIVE_PROVIDER = "gemini"
GEMINI_API_KEY = "sem-vlozte-svuj-klic"
```

Při změně služby stačí přepsat `ACTIVE_PROVIDER` na `"groq"`, `"ollama"`
nebo `"openai"`. U cloudové služby vyplňte i její klíč. Pole ostatních služeb
mohou zůstat prázdná. Model měníte v `PROVIDERS["nazev_sluzby"]["model"]`.
Klíče neukládejte do veřejného repozitáře.

| Poskytovatel | Výchozí model | Kde získat klíč |
|---|---|---|
| Gemini | `gemini-2.5-flash` | [Google AI Studio](https://aistudio.google.com/apikey) |
| Groq | `openai/gpt-oss-20b` | [Groq Console](https://console.groq.com/keys) |
| Ollama | `qwen2.5:7b` | Lokálně není potřeba |
| OpenAI | `gpt-4o-mini` | [OpenAI Platform](https://platform.openai.com/api-keys) |

V AI Studiu a Groq Console se přihlaste a zvolte vytvoření API klíče.
Vytvoření klíče je zdarma; dostupnost bezplatného provozu a limity závisí
na modelu, účtu a regionu. Viz [Gemini ceník](https://ai.google.dev/gemini-api/docs/pricing)
a [Groq limity](https://console.groq.com/docs/rate-limits). Klíč OpenAI sám
nezaručuje kredit pro API; předplatné ChatGPT není API kredit.

## Ollama offline

Nainstalujte [Ollamu](https://ollama.com/download) a jednou s připojením stáhněte model:

```bash
ollama pull qwen2.5:7b
ollama serve
```

Pokud služba již běží, druhý příkaz není potřeba. V `config.py` nastavte:

```python
ACTIVE_PROVIDER = "ollama"
OFFLINE = True
```

Potom spusťte `python alza_dataset.py` v jiném terminálu. Po instalaci závislostí
a stažení modelu tento režim nepotřebuje internet: místo scrapingu používá
původní vestavěná témata Alzy. `OFFLINE = False` ponechá scraping i pro Ollamu.
Lokální model potřebuje dostatek paměti a může být výrazně pomalejší; jeho
časový limit lze upravit v `PROVIDERS["ollama"]["timeout"]`.

## Co zůstalo zachováno

Původní verze neměla pevný seznam pojmenovaných intentů s popisy. Funkce
`discover` je odvozovala z nalezených témat. Tato logika je zachována;
zdrojová témata `FALLBACK` a URL `SEEDS` jsou přesunuta doslova do `config.py`.
Názvy a popisy odvozené při jednotlivých bězích se proto mohou lišit i mezi
modely. Najdete je v `intents.json` příslušného běhu.

Generační prompt, 30 výpovědí na intent (6 formálních, 6 hovorových,
6 bez diakritiky, 6 krátkých, 6 urgentních), deduplikace a stratifikovaný
split zůstávají stejné. Deduplikace je lexikální; sémantiku a styly je třeba
auditem zkontrolovat. Výpovědi jsou syntetické, nikoli skutečné zákaznické chaty.

## Výstupy a další nastavení

Každý běh vytvoří novou podsložku `data/<cas_UTC>_<provider>/`. Obsahuje
`dataset.jsonl`, `train.jsonl`, `dev.jsonl`, `test.jsonl`, `intents.json`,
`sources.json`, `scrape_report.json`, průběžný `partial.jsonl` a po úspěšném
dokončení `manifest.json` s providerem, modelem a skutečnými počty.
Při 25 intentech je rozdělení 525/112/113, protože 15 % ze 750 není celé číslo.

V `config.py` lze upravit také `INTENT_COUNT`, `GENERATION_ROUNDS`,
`OUTPUT_DIR`, `SPLIT_SEED`, `SCRAPE_ONLY` či `REQUIRE_LIVE`. Poslední dvě
volby nelze kombinovat s `OFFLINE`. Seed reprodukuje pouze dělení stejné sady.

Všechny služby používají stejný REST endpoint `/chat/completions` a validaci
JSON schématu. Pokud jiný zvolený model nepodporuje strukturovaný výstup,
nastavte u něj `"response_format": "json_object"`; schéma se pak předá
v systémové instrukci a výsledek se nadále lokálně validuje. Neplatné odpovědi
se opakují nejvýše čtyřikrát. Chybějící klíč, odmítnutí přístupu, nenalezený
model a výpadky spojení končí srozumitelnou hláškou bez tracebacku.

## Testy a dokumentace API

```bash
python -m unittest -v test_alza_dataset.py
```

Testy simulují odpovědi služeb; nevyžadují klíče ani nevolají placená API.
Konfigurace vychází z dokumentace [Gemini](https://ai.google.dev/gemini-api/docs/openai),
[Groq](https://console.groq.com/docs/openai),
[Groq Structured Outputs](https://console.groq.com/docs/structured-outputs),
[Ollamy](https://docs.ollama.com/api/openai-compatibility)
a [OpenAI](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=chat).
