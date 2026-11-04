# Toy klasifikace intentů pomocí embeddingů a k-NN

Python 3.10+. Samostatný skript nepotřebuje `config.py` ani klíče LLM služeb.

```bash
cd /home/martin/Bakalarka-/alza_scraper
uv run toy_pipeline_knn.py
```

Výchozí vstup je `dataset_openai_gpt-4o-mini.jsonl` v aktuální složce, případně
`datasets/` vedle skriptu. Podporuje také název `dataset_openai.jsonl` v aktuální
složce. Libovolnou cestu můžete zadat:

```bash
uv run toy_pipeline_knn.py datasets/dataset_openai_gpt-4o-mini.jsonl --intents 5
```

Bez uv:

```bash
pip install 'sentence-transformers>=5.1,<6' 'transformers>=4.56,<6' 'scikit-learn>=1.5,<2'
python toy_pipeline_knn.py /cesta/dataset.jsonl
```

Skript vybere prvních pět unikátních intentů v pořadí JSONL a pro každý prvních
10 unikátních textů. Následuje náhodné stratifikované rozdělení 8 train / 2 val
na intent s pevně nastaveným seedem 42. To odpovídá 20 % pro validaci;
25 % by nebylo slučitelné s požadovaným počtem 8/2. Počty lze změnit přes
`--train-per-intent`, `--val-per-intent` a seed přes `--seed`.
Kopie stejného textu se odstraňují před dělením. Nedostatek zpráv, vadný JSON
nebo konfliktní štítky u stejného textu způsobí srozumitelnou chybu.

Nejdřív se zkusí `google/embeddinggemma-300m` z cache, pak stažení. Při chybě
načtení nebo výpočtu se stejným způsobem zkusí
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
EmbeddingGemma může vyžadovat [přijetí licence na Hugging Face](https://huggingface.co/google/embeddinggemma-300m)
a přihlášení pomocí `hf auth login` nebo nastavení `HF_TOKEN`. Záložní model
je veřejný. Pokud není dostupný ani jeden model, skript skončí s chybou.

```bash
uv run toy_pipeline_knn.py --offline
uv run toy_pipeline_knn.py --device cuda
```

`--offline` zakazuje síťové načítání modelů, ale závislosti musí být již
nainstalované; pro offline spouštění s uv použijte i jeho přepínač `uv run --offline`.
Výchozí CPU je předvídatelná volba pro malý vzorek; CUDA vyžaduje odpovídající
hardware a instalaci PyTorch.

Pro EmbeddingGemma používáme [doporučený klasifikační prompt](https://huggingface.co/google/embeddinggemma-300m#prompt-instructions)
`task: classification | query: ` stejně pro train i val. Model nedostává správné
štítky jako součást vstupu. `.encode(normalize_embeddings=True)` vytváří
jednotkové embeddingy. Klasifikátor používá cosine vzdálenost a hlasování
3 nejbližších sousedů bez učení embeddingového modelu.

Výstup zahrnuje skutečně použitý model, accuracy, classification report a tabulku
všech validačních textů s oběma štítky, confidence a stavem SPRAVNE/CHYBA.
Dlouhé texty se zalamují, nekrátí. Confidence je podíl hlasů sousedů, nikoli
kalibrovaná jistota: i nesprávná predikce může mít 100 %.
Toy přesnost na 10 syntetických validačních zprávách není reálný benchmark;
výběr prvních pěti intentů je záměrně malý zkušební vzorek.
