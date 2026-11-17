# Hierarchický strom podpory Alzy

Samostatný skript, Python 3.10+, bez LLM a API klíčů.

```bash
uv run scrape_intents.py
```

`uv` automaticky zajistí dvě závislosti uvedené přímo ve skriptu. Alternativně:

```bash
python3 -m pip install requests beautifulsoup4
python3 scrape_intents.py
```

Vedle skriptu vznikne `intents_tree.py`. Každý další běh jej nahradí.

```python
from intents_tree import ALZA_INTENTS, SCRAPE_METADATA

for category in ALZA_INTENTS.values():
    print(category["title"])
    for topic in category["children"]:
        print(topic["id"], topic["title"], topic["url"])
```

Kořenem je slovník s pěti klíči `orders`, `services`, `claims`, `account`,
`other`. Každá kategorie má `id`, `title`, `url`, `description` a seznam
`children`; každé podtéma má čtyři požadovaná pole bez dalších atributů.

Požadovaných pět kategorií je normalizované uspořádání výstupu, ne doslovná
kopie navigace webu. Veřejný formulář uvádí například „Služba“ a „Reklamace
a vrácení zboží“; témata účtu jsou také v sekci Ostatní. Tituly podtémat se
zachovávají z HTML, u fallbacku z uložených názvů veřejných formulářů.

Scraper navštíví mobilní rozcestník, případně desktopový kontakt, a následně
prochází podporované odkazy kategorií do hloubky 2 (nejvýše 8 stránek na
kategorii). Zachovává parametry `qid`/`qgid`, které rozlišují témata formulářů.
Nenavštěvuje účty za přihlášením ani produktové stránky. HTTP hlavičky
nezaručují odstranění blokace. Při 403, síťové chybě nebo HTML bez témat
použije statickou zálohu; chybějící známá témata doplňuje i při částečném načtení.

V `SCRAPE_METADATA["origins"]` je pro každý identifikátor uvedeno `scraped`
nebo `static_fallback`. U fallbacku bez ověřeného samostatného odkazu vede
`url` na skutečný nadřazený formulář, ne na vymyšlenou adresu.
Popisy stručně pojmenovávají situaci, neinterpretují obchodní podmínky.

Známá témata mají anglické významové identifikátory. Nové téma dostane stabilní
ID jako `services_topic_a1b2…` odvozené z názvu. To nevyžaduje překladač ani API;
nejde o automatický překlad neznámého českého názvu do angličtiny.

Záloha obsahuje 18 podtémat. Výpadek webu nezpůsobí prázdný výsledek;
chybějící knihovny či nemožnost zápisu na disk samozřejmě brání spuštění nebo
uložení. Aktuální změny HTML mohou vyžadovat úpravu selektorů `extract`.

Zdroje vestavěných témat (ověřeno 24. 10. 2026):

- [Objednávky](https://www.alza.cz/contact/orders.htm)
- [Služby](https://www.alza.cz/contact/services.htm)
- [Reklamace](https://www.alza.cz/kontakt/reklamace?lang=CZ)
- [Účet a ostatní](https://www.alza.cz/contact/others.htm)
