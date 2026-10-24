"""Nastavení generátoru. Pro změnu služby upravte jen ACTIVE_PROVIDER."""
ACTIVE_PROVIDER = "ollama"  # "gemini", "groq", "ollama", "openai"

from pathlib import Path

# Klíč vyplňte pouze pro služby, které chcete používat.
GEMINI_API_KEY = ""
GROQ_API_KEY = ""
OPENAI_API_KEY = ""

# base_url končí před /chat/completions.
# Pro jiný model lze zvolit response_format="json_object", pokud nepodporuje schema.
PROVIDERS = {
    "gemini": {
        "model": "gemini-2.5-flash",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "api_key": GEMINI_API_KEY,
        "requires_key": True,
        "response_format": "json_schema",
        "timeout": 180,
    },
    "groq": {
        "model": "openai/gpt-oss-20b",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key": GROQ_API_KEY,
        "requires_key": True,
        "response_format": "json_schema",
        "timeout": 180,
    },
    "ollama": {
        "model": "qwen2.5:7b",
        "base_url": "http://localhost:11434/v1",
        "api_key": "",  # Lokální Ollama klíč nevyžaduje.
        "requires_key": False,
        "response_format": "json_schema",
        "timeout": 600,
    },
    "openai": {
        "model": "gpt-4o-mini",
        "base_url": "https://api.openai.com/v1",
        "api_key": OPENAI_API_KEY,
        "requires_key": True,
        "response_format": "json_schema",
        "timeout": 180,
    },
}

# Každé spuštění vytvoří vlastní podsložku, starší výsledky nepřepíše.
OUTPUT_DIR = Path(__file__).resolve().parent / "data"
INTENT_COUNT = 25
MAX_PAGES = 12
SCRAPE_PAUSE = 1.0
GENERATION_ROUNDS = 8
SPLIT_SEED = 42
REQUIRE_LIVE = False
SCRAPE_ONLY = False
# True vynechá web Alzy. Pro plně offline běh zvolte také lokální Ollamu.
OFFLINE = False

# Původní kód neměl pevné názvy a popisy intentů: odvozuje je funkcí discover.
# Následují původní zdrojová témata doslova, bez změny jejich významu.
BASE = "https://www.alza.cz"
# Tyto URL byly při přípravě skriptu otevřeny a obsahovaly témata podpory.
FALLBACK = {
    BASE + "/kontakt/reklamace?lang=CZ": [
        "Založení reklamace vadného výrobku", "Odstoupení od nákupu a vrácení výrobku",
        "Poškození balíku během přepravy", "Chybějící položky v doručené objednávce",
        "Dodání jiného výrobku místo objednaného", "Odvoz vysloužilého spotřebiče",
        "Převzetí výrobku po dokončení reklamace", "Sledování průběhu reklamace",
        "Zmeškané vyzvednutí opraveného výrobku z boxu",
    ],
    BASE + "/contact/orders.htm": [
        "Vyžádání kopie daňového dokladu", "Úprava již vytvořené objednávky",
        "Pomoc s uhrazením dokladu", "Dotaz na refundaci peněz",
        "Výběr způsobu doručení", "Rozložení ceny do tří plateb",
        "Potíže se zakoupeným digitálním obsahem", "Pronajímání elektroniky přes Alza NEO",
        "Financování nákupu pomocí splátek",
    ],
    BASE + "/contact/services.htm": [
        "Členství v programu AlzaPlus", "Prodloužení záručního krytí",
        "Pojistné krytí při odcizení či rozbití", "Dodatečná služba delší lhůty pro vrácení",
        "Pomoc s úvodním nastavením zařízení", "Služba rychlé výměny vadného výrobku",
        "Výkup původního zařízení při nákupu nového", "Zabalení nákupu jako dárku",
        "Pojistná ochrana doručovaného balíku", "Zabudování domácího spotřebiče",
        "Upevnění televizoru na stěnu", "Odeslání balíku prostřednictvím AlzaBoxu",
    ],
    BASE + "/contact/others.htm": [
        "Hlášení technické chyby e-shopu", "Podání stížnosti na zákaznický servis",
        "Žádost týkající se osobních dat", "Odhlášení reklamních sdělení",
        "Obnovení přístupu po ztrátě hesla", "Zapojení firmy do B2B programu",
    ],
}
SEEDS = [BASE + "/kontakt", BASE + "/reklamace-a-vraceni", *FALLBACK]
