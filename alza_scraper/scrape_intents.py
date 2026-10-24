# /// script
# requires-python = ">=3.10"
# dependencies = ["requests>=2.32,<3", "beautifulsoup4>=4.12,<5"]
# ///
"""Veřejná nápověda Alzy -> intents_tree.py, bez LLM a bez API klíče."""
from collections import deque
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from pprint import pformat
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit
import logging
import re
import time
import unicodedata

import requests
from bs4 import BeautifulSoup

START_URL = "https://m.alza.cz/contact-and-support"
BASE = "https://www.alza.cz"
OUTPUT = Path(__file__).resolve().with_name("intents_tree.py")
MAX_PAGES_PER_CATEGORY = 8
MAX_DEPTH = 2
REQUEST_DELAY = 0.7
LOG = logging.getLogger("alza")

# Pevné uspořádání požadované uživatelem. Není tvrzením, že web má
# přesně tuto navigaci. Účet sdílí veřejný formulář se sekcí Ostatní.
CATEGORIES = {
    "orders": ("Objednávky a platby", "/contact/orders.htm", "Podpora objednávek, úhrad a doručení."),
    "services": ("Služby", "/contact/services.htm", "Podpora doplňkových služeb Alzy."),
    "claims": ("Reklamace a vrácené zboží", "/kontakt/reklamace?lang=CZ", "Řešení vad výrobků a vrácení nákupu."),
    "account": ("Můj účet", "/contact/others.htm", "Pomoc s přístupem k účtu a osobními údaji."),
    "other": ("Ostatní", "/contact/others.htm", "Další požadavky zákazníků."),
}

# Tituly z veřejných kontaktních formulářů, ověřené 24. 10. 2026.
# Prázdná cesta znamená skutečnou URL nadřazeného formuláře:
# samostatný odkaz podtématu nebyl ověřen, proto jej nevymýšlíme.
# id, title, path, description
FALLBACK = {
    "orders": [
        ("order_invoice", "Zaslání faktury/dobropisu", "", "Zákazník potřebuje kopii dokladu k nákupu."),
        ("order_change", "Změna objednávky", "", "Zákazník chce upravit již vytvořenou objednávku."),
        ("order_refund", "Vrácení platby", "", "Zákazník řeší vrácení zaplacené částky."),
        ("order_shipping", "Možnosti dopravy", "", "Zákazník vybírá způsob doručení nákupu."),
        ("order_installments", "Nákup na splátky", "", "Zákazník se zajímá o financování nákupu."),
    ],
    "services": [
        ("services_alzaplus", "AlzaPlus+", "/contact/services.htm?qgid=69&qid=691", "Zákazník řeší členství v programu AlzaPlus+."),
        ("services_warranty", "Prodloužená záruka", "/contact/services.htm?qgid=69&qid=338", "Zákazník se ptá na doplňkové záruční krytí."),
        ("services_alzabox_shipping", "Pošli to Alzaboxem", "/contact/services.htm?qgid=69&qid=723", "Zákazník chce odeslat balík prostřednictvím AlzaBoxu."),
        ("services_gift_wrap", "Dárkové balení", "", "Zákazník chce nechat nákup zabalit jako dárek."),
    ],
    "claims": [
        ("claim_create", "Chci reklamovat zboží", "", "Zákazník potřebuje uplatnit vadu zakoupeného výrobku."),
        ("claim_return", "Chci vrátit zboží", "", "Zákazník chce vrátit zakoupený výrobek."),
        ("claim_status", "Stav reklamace", "", "Zákazník potřebuje zjistit průběh vyřizování reklamace."),
        ("claim_damaged_delivery", "Nahlásit poškozenou zásilku", "", "Zákazník obdržel balík poškozený při přepravě."),
    ],
    "account": [
        ("account_password", "Zapomenuté heslo", "", "Zákazník potřebuje obnovit přístup ke svému účtu."),
        ("account_personal_data", "Osobní údaje", "", "Zákazník řeší své osobní údaje u obchodníka."),
        ("account_unsubscribe", "Zrušení zasílání akčních nabídek", "/contact/others.htm?qgid=68&qid=377", "Zákazník nechce dostávat reklamní nabídky."),
    ],
    "other": [
        ("other_feedback", "Pochvala / stížnost", "", "Zákazník předává zpětnou vazbu ke své zkušenosti."),
        ("other_cooperation", "Spolupráce", "", "Zákazník nebo firma nabízí obchodní spolupráci."),
    ],
}


def normalize(text):
    text = unicodedata.normalize("NFKD", text.casefold())
    return " ".join(re.findall(r"[a-z0-9]+", "".join(
        c for c in text if not unicodedata.combining(c))))


def absolute_url(href, base):
    """Zachová identifikátory témat qid/qgid i fragment; odstraní tracking."""
    try:
        parts = urlsplit(urljoin(base, href))
        valid = (parts.scheme == "https" and parts.hostname in {"m.alza.cz", "www.alza.cz"}
                 and not parts.username and not parts.password and parts.port in (None, 443))
    except ValueError:
        return None
    if not valid:
        return None
    query = urlencode(sorted((k, v) for k, v in parse_qsl(parts.query)
                             if k.lower() in {"qid", "qgid", "lang"}))
    return urlunsplit((parts.scheme, parts.netloc, parts.path, query, parts.fragment))


def support_url(url):
    path = urlsplit(url).path
    return path.startswith(("/contact", "/kontakt", "/reklamace-a-vraceni"))


def category_for(title, url, default=None):
    text = normalize(title)
    aliases = {"objednavky a platby": "orders", "sluzba": "services",
               "sluzby": "services", "reklamace a vraceni zbozi": "claims",
               "reklamace a vracene zbozi": "claims", "muj ucet": "account",
               "ostatni": "other"}
    if text in aliases:
        return aliases[text]
    if any(word in text for word in ("heslo", "osobni udaj", "muj ucet", "prihlaseni", "akcnich nabidek")):
        return "account"
    path = urlsplit(url).path.lower()
    for words, category in ((('orders', 'objednavky-a-platby'), 'orders'),
                            (('services', 'sluzby'), 'services'),
                            (('reklamace', 'claims'), 'claims')):
        if any(word in path for word in words):
            return category
    if "others" in path or "ostatni" in path:
        return "other"
    return default


def make_node(category, title, url):
    # Známá témata mají čitelné anglické ID; neznámá stabilní technické ID.
    # Pouhá transliterace češtiny by nebyla anglický překlad.
    for ident, known_title, _, description in FALLBACK[category]:
        if normalize(title) == normalize(known_title):
            return {"id": ident, "title": title, "url": url, "description": description}
    digest = sha256(normalize(title).encode("utf-8")).hexdigest()[:16]
    return {"id": f"{category}_topic_{digest}", "title": title, "url": url,
            "description": f'Zákazník potřebuje pomoc s tématem „{title}“.'}


class Scraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "cs-CZ,cs;q=0.9,en;q=0.5",
        })
        self.cache = {}
        self.report = []

    def fetch(self, url):
        url = url.split("#", 1)[0]
        if url in self.cache:
            return self.cache[url]
        self.cache[url] = None
        time.sleep(REQUEST_DELAY)
        try:
            # Kontrola každého redirectu; žádné přihlašování ani obcházení CAPTCHA.
            current = url
            for _ in range(6):
                response = self.session.get(current, timeout=(8, 20), allow_redirects=False)
                if response.status_code not in {301, 302, 303, 307, 308}:
                    break
                target = absolute_url(response.headers.get("Location", ""), current)
                if not target or not support_url(target):
                    raise ValueError("Přesměrování mimo veřejnou nápovědu")
                current = target
            else:
                raise ValueError("Příliš mnoho přesměrování")
            response.raise_for_status()
            if "html" not in response.headers.get("Content-Type", "").lower():
                raise ValueError("Odpověď není HTML")
            soup = BeautifulSoup(response.content, "html.parser")
            title = soup.title.get_text(" ", strip=True) if soup.title else ""
            if re.search(r"access denied|just a moment|captcha", title, re.I):
                raise ValueError("Ochranná stránka místo nápovědy")
            for node in soup.select("script, style, header, footer, nav, [id*='cookie'], [class*='cookie']"):
                node.decompose()
            root = soup.select_one("main, [role=main]") or soup
            self.cache[url] = (root, current)
            self.report.append({"url": url, "status": "downloaded", "final_url": current})
        except (requests.RequestException, ValueError) as error:
            LOG.warning("%s: %s", url, error)
            self.report.append({"url": url, "status": "failed", "error": str(error)})
        return self.cache[url]

    def extract(self, root, url, category):
        known = {normalize(t) for _, t, _, _ in FALLBACK[category]}
        generic = {normalize(v[0]) for v in CATEGORIES.values()} | {
            "sluzba", "reklamace a vraceni zbozi", "zbozi", "nejcastejsi dotazy",
            "s cim potrebujete poradit", "s cim vam muzeme pomoci"}
        candidates = []
        for element in root.select("a[href], [data-href], summary, h2, h3, h4, label, option, span"):
            title = " ".join(element.get_text(" ", strip=True).split())
            key = normalize(title)
            if not 4 <= len(title) <= 180 or key in generic:
                continue
            href = element.get("href") or element.get("data-href")
            target = absolute_url(href, url) if href else url
            if not target or (href and not support_url(target)):
                continue
            # Bez odkazu přijmeme známé téma nebo nadpis skutečné otázky.
            if not href and key not in known and not (
                    element.name in {"summary", "h2", "h3", "h4"} and title.endswith("?")):
                continue
            if any(word in key for word in ("cookie", "personaliz", "prihlasit se", "preskocit")):
                continue
            owner = category_for(title, target, category)
            # Účet a Ostatní sdílejí formulář, ale ne stejná podtémata.
            if owner != category:
                continue
            candidates.append((title, target))
        return candidates

    def build(self):
        tree, provenance = {}, {}
        landing = self.fetch(START_URL) or self.fetch(BASE + "/kontakt")
        for category, (title, path, description) in CATEGORIES.items():
            root_url = BASE + path
            # Základní URL doplní nově nalezené odkazy na příslušnou kategorii.
            seeds = [root_url]
            if landing:
                for anchor in landing[0].select("a[href]"):
                    target = absolute_url(anchor["href"], landing[1])
                    if target and support_url(target) and category_for(anchor.get_text(), target) == category:
                        seeds.append(target)
            queue, visited, nodes = deque((u, 0) for u in dict.fromkeys(seeds)), set(), {}
            while queue and len(visited) < MAX_PAGES_PER_CATEGORY:
                url, depth = queue.popleft()
                fetch_url = url.split("#", 1)[0]
                if fetch_url in visited:
                    continue
                visited.add(fetch_url)
                page = self.fetch(fetch_url)
                if not page:
                    continue
                candidates = self.extract(page[0], page[1], category)
                if not candidates:
                    self.report.append({"url": fetch_url, "status": "no_topics", "category": category})
                for topic_title, topic_url in candidates:
                    node = make_node(category, topic_title, topic_url)
                    nodes.setdefault(node["id"], node)
                    provenance[node["id"]] = "scraped"
                    if depth < MAX_DEPTH and topic_url.split("#", 1)[0] != fetch_url:
                        queue.append((topic_url, depth + 1))
            # Doplňujeme i částečně načtené kategorie, nejen zcela prázdné.
            for ident, topic_title, topic_path, desc in FALLBACK[category]:
                if ident not in nodes:
                    nodes[ident] = {"id": ident, "title": topic_title,
                                    "url": BASE + topic_path if topic_path else root_url,
                                    "description": desc}
                    provenance[ident] = "static_fallback"
            tree[category] = {"id": category, "title": title, "url": root_url,
                              "description": description,
                              "children": sorted(nodes.values(), key=lambda n: n["id"])}
        return tree, {"created_at": datetime.now(timezone.utc).isoformat(),
                      "category_layout": "user_defined", "fallback_verified_on": "2026-10-24",
                      "origins": provenance, "requests": self.report}


def write_tree(tree, metadata):
    content = ("# Automaticky vytvořeno scrape_intents.py; UTF-8.\n"
               "# Popisy jsou generované shrnutí tématu, nikoli obchodní podmínky.\n\n"
               "ALZA_INTENTS = " + pformat(tree, width=100, sort_dicts=False) + "\n\n"
               "SCRAPE_METADATA = " + pformat(metadata, width=100, sort_dicts=False) + "\n")
    # repr/pformat bezpečně escapuje názvy získané z nedůvěryhodného webu.
    compile(content, str(OUTPUT), "exec")
    temporary = OUTPUT.with_suffix(".py.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(OUTPUT)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    scraper = Scraper()
    try:
        tree, metadata = scraper.build()
        write_tree(tree, metadata)
        fallback_count = sum(v == "static_fallback" for v in metadata["origins"].values())
        LOG.info("Uloženo %s: %d kategorií, %d podtémat (%d ze zálohy).",
                 OUTPUT, len(tree), sum(len(n["children"]) for n in tree.values()), fallback_count)
    except OSError as error:
        LOG.error("Soubor nelze uložit: %s", error)
        return 1
    finally:
        scraper.session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
