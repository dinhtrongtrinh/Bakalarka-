# Automaticky vytvořeno scrape_intents.py; UTF-8.
# Popisy jsou generované shrnutí tématu, nikoli obchodní podmínky.

ALZA_INTENTS = {
    "orders_and_payments": {
        "title": "Objednávky a platby",
        "url": "https://www.alza.cz/contact-and-support/objednavky-a-platby/4",
        "description": "Zákazník řeší objednávku, její úhradu, doručení nebo související doklady.",
        "sub_intents": [
            {
                "id": "order_invoice_request",
                "title": "Zaslání faktury nebo dobropisu",
                "url": "https://www.alza.cz/contact-and-support/zaslani-faktury-nebo-dobropisu/34",
                "description": "Zákazník potřebuje zaslat fakturu nebo dobropis ke své objednávce."
            },
            {
                "id": "order_change_or_extension",
                "title": "Změna nebo prodloužení objednávky",
                "url": "https://www.alza.cz/contact-and-support/zmena-nebo-prodlouzeni-objednavky/52",
                "description": "Zákazník chce upravit objednávku nebo prodloužit lhůtu pro její vyzvednutí."
            },
            {
                "id": "order_delay",
                "title": "Zpožděná objednávka",
                "url": "https://www.alza.cz/contact-and-support/zpozdena-objednavka/457",
                "description": "Zákazník řeší zpoždění objednávky a potřebuje zjistit nový termín doručení."
            },
            {
                "id": "order_invoice_or_credit_note_payment",
                "title": "Úhrada faktury nebo dobropisu",
                "url": "https://www.alza.cz/contact-and-support/uhrada-faktury-nebo-dobropisu/53",
                "description": "Zákazník řeší úhradu faktury nebo vyplacení částky uvedené na dobropisu."
            },
            {
                "id": "order_payment_refund",
                "title": "Vrácení platby",
                "url": "https://www.alza.cz/contact-and-support/vraceni-platby/54",
                "description": "Zákazník potřebuje zjistit stav nebo způsob vrácení zaplacené částky."
            },
            {
                "id": "order_delivery_options",
                "title": "Možnosti dopravy",
                "url": "https://www.alza.cz/contact-and-support/moznosti-dopravy/55",
                "description": "Zákazník se informuje o dostupných způsobech doručení nebo vyzvednutí objednávky."
            },
            {
                "id": "order_pay_in_three",
                "title": "Nákup na třetinku",
                "url": "https://www.alza.cz/contact-and-support/nakup-na-tretinku/56",
                "description": "Zákazník chce při objednávání rozdělit úhradu nákupu do tří plateb."
            },
            {
                "id": "order_digital_content",
                "title": "Elektronický obsah",
                "url": "https://www.alza.cz/contact-and-support/elektronicky-obsah/57",
                "description": "Zákazník řeší doručení, zpřístupnění nebo aktivaci zakoupeného elektronického obsahu."
            },
            {
                "id": "order_business_purchase",
                "title": "Chci nakoupit na firmu",
                "url": "https://www.alza.cz/contact-and-support/chci-nakoupit-na-firmu/459",
                "description": "Zákazník potřebuje vytvořit objednávku na firmu a uvést firemní fakturační údaje."
            },
            {
                "id": "order_device_rental",
                "title": "Pronájem mobilů a počítačů (Alza NEO)",
                "url": "https://www.alza.cz/contact-and-support/pronajem-mobilu-a-pocitacu-alza-neo/58",
                "description": "Zákazník řeší objednání pronájmu mobilu nebo počítače prostřednictvím Alza NEO."
            },
            {
                "id": "order_installment_purchase",
                "title": "Nákup na splátky",
                "url": "https://www.alza.cz/contact-and-support/nakup-na-splatky/59",
                "description": "Zákazník se informuje o financování objednávky na splátky nebo řeší jeho sjednání."
            }
        ]
    },
    "services": {
        "title": "Služby",
        "url": "https://www.alza.cz/contact-and-support/sluzby/15",
        "description": "Zákazník řeší sjednání, využívání nebo ukončení doplňkových služeb Alzy.",
        "sub_intents": [
            {
                "id": "services_alzaplus",
                "title": "AlzaPlus+",
                "url": "https://www.alza.cz/contact-and-support/alzaplus/272",
                "description": "Zákazník řeší aktivaci, výhody, obnovování nebo ukončení členství AlzaPlus+."
            },
            {
                "id": "services_alza_benefit",
                "title": "Alza Benefit",
                "url": "https://www.alza.cz/contact-and-support/alza-benefit/43",
                "description": "Zákazník potřebuje informace o programu Alza Benefit nebo pomoc s využitím jeho výhod."
            },
            {
                "id": "services_device_rental",
                "title": "Pronájem mobilů a počítačů (Alza Neo)",
                "url": "https://www.alza.cz/contact-and-support/pronajem-mobilu-a-pocitacu-alza-neo/64",
                "description": "Zákazník se informuje o službě pronájmu zařízení nebo řeší průběh svého pronájmu."
            },
            {
                "id": "services_extended_warranty",
                "title": "Prodloužená záruka",
                "url": "https://www.alza.cz/contact-and-support/prodlouzena-zaruka/65",
                "description": "Zákazník řeší sjednání, rozsah nebo uplatnění prodloužené záruky."
            },
            {
                "id": "services_insurance_withdrawal",
                "title": "Žádost o odstoupení od smlouvy - služba pojištění",
                "url": "https://www.alza.cz/contact-and-support/zadost-o-odstoupeni-od-smlouvy---sluzba-pojisteni/487",
                "description": "Zákazník chce odstoupit od smlouvy o sjednané službě pojištění."
            },
            {
                "id": "services_damage_and_theft_insurance",
                "title": "Pojištění proti rozbití a krádeži",
                "url": "https://www.alza.cz/contact-and-support/pojisteni-proti-rozbiti-a-kradezi/66",
                "description": "Zákazník řeší sjednání pojištění zařízení proti rozbití a krádeži nebo pojistnou událost."
            },
            {
                "id": "services_return_within_sixty_days",
                "title": "Vrácení zboží do 60 dnů",
                "url": "https://www.alza.cz/contact-and-support/vraceni-zbozi-do-60-dnu/67",
                "description": "Zákazník se informuje o službě vrácení zboží do 60 dnů nebo ji chce využít."
            },
            {
                "id": "services_initial_setup",
                "title": "První spuštění",
                "url": "https://www.alza.cz/contact-and-support/prvni-spusteni/68",
                "description": "Zákazník potřebuje pomoc se službou prvního spuštění a základního nastavení zařízení."
            },
            {
                "id": "services_immediate_replacement",
                "title": "Okamžitá výměna",
                "url": "https://www.alza.cz/contact-and-support/okamzita-vymena/69",
                "description": "Zákazník řeší sjednání nebo využití služby okamžité výměny vadného výrobku."
            },
            {
                "id": "services_trade_in",
                "title": "Staré za nové",
                "url": "https://www.alza.cz/contact-and-support/stare-za-nove/70",
                "description": "Zákazník chce využít své staré zařízení při pořízení nového prostřednictvím služby Staré za nové."
            },
            {
                "id": "services_pay_in_three",
                "title": "Nákup na třetinku",
                "url": "https://www.alza.cz/contact-and-support/nakup-na-tretinku/137",
                "description": "Zákazník potřebuje informace nebo podporu ke službě rozdělení nákupu do tří plateb."
            },
            {
                "id": "services_gift_wrapping",
                "title": "Dárkové balení",
                "url": "https://www.alza.cz/contact-and-support/darkove-baleni/72",
                "description": "Zákazník chce objednat dárkové zabalení zboží nebo řeší provedení této služby."
            },
            {
                "id": "services_shipment_insurance",
                "title": "Pojištění zásilky",
                "url": "https://www.alza.cz/contact-and-support/pojisteni-zasilky/73",
                "description": "Zákazník se informuje o pojištění zásilky nebo řeší jeho uplatnění."
            },
            {
                "id": "services_built_in_appliance_installation",
                "title": "Montáž vestavného spotřebiče",
                "url": "https://www.alza.cz/contact-and-support/montaz-vestaveneho-spotrebice/74",
                "description": "Zákazník potřebuje sjednat nebo vyřešit montáž vestavného spotřebiče."
            },
            {
                "id": "services_tv_wall_mounting",
                "title": "Instalace TV na zeď",
                "url": "https://www.alza.cz/contact-and-support/instalace-tv-na-zed/75",
                "description": "Zákazník chce objednat instalaci televizoru na zeď nebo řeší její průběh."
            },
            {
                "id": "services_solar_power_systems",
                "title": "Solární elektrárny",
                "url": "https://www.alza.cz/contact-and-support/solarni-elektrarny/76",
                "description": "Zákazník potřebuje informace nebo podporu k nabídce solárních elektráren."
            },
            {
                "id": "services_warranty_or_replacement_callback",
                "title": "Nabídka prodloužené záruky nebo okamžité výměny – kontaktujte mě telefonicky.",
                "url": "https://www.alza.cz/contact-and-support/nabidka-prodlouzene-zaruky-nebo-okamzite-vymeny-kontaktujte-me-telefonicky/77",
                "description": "Zákazník žádá telefonický kontakt s nabídkou prodloužené záruky nebo okamžité výměny."
            },
            {
                "id": "services_send_via_alzabox",
                "title": "Pošli to Alzaboxem",
                "url": "https://www.alza.cz/contact-and-support/posli-to-alzaboxem/78",
                "description": "Zákazník potřebuje pomoc s odesláním zásilky prostřednictvím služby Pošli to Alzaboxem."
            }
        ]
    },
    "claims_and_returns": {
        "title": "Reklamace a vrácení zboží",
        "url": "https://www.alza.cz/contact-and-support/reklamace-a-vraceni-zbozi/3",
        "description": "Zákazník řeší reklamaci, vrácení zboží, problém s doručenou zásilkou nebo svoz starého spotřebiče.",
        "sub_intents": [
            {
                "id": "claim_status",
                "title": "Stav reklamace",
                "url": "https://www.alza.cz/contact-and-support/stav-reklamace/19",
                "description": "Zákazník chce zjistit aktuální stav a průběh vyřizování své reklamace."
            },
            {
                "id": "return_goods",
                "title": "Chci vrátit zboží",
                "url": "https://www.alza.cz/contact-and-support/chci-vratit-zbozi/21",
                "description": "Zákazník chce vrátit zakoupené zboží a potřebuje znát postup."
            },
            {
                "id": "claim_defective_goods",
                "title": "Chci reklamovat zboží",
                "url": "https://www.alza.cz/contact-and-support/chci-reklamovat-zbozi/20",
                "description": "Zákazník chce uplatnit reklamaci kvůli závadě zakoupeného výrobku."
            },
            {
                "id": "claim_damaged_shipment",
                "title": "Nahlásit poškozenou zásilku",
                "url": "https://www.alza.cz/contact-and-support/nahlasit-poskozenou-zasilku/60",
                "description": "Zákazník potřebuje nahlásit poškození doručené zásilky."
            },
            {
                "id": "claim_incomplete_shipment",
                "title": "Nahlásit nekompletní zásilku",
                "url": "https://www.alza.cz/contact-and-support/nahlasit-nekompletni-zasilku/61",
                "description": "Zákazník oznamuje, že v doručené zásilce chybí objednané zboží nebo jeho součást."
            },
            {
                "id": "claim_wrong_goods",
                "title": "Dorazilo mi jiné zboží",
                "url": "https://www.alza.cz/contact-and-support/dorazilo-mi-jine-zbozi/62",
                "description": "Zákazník obdržel jiné zboží, než si objednal, a chce vyřešit záměnu."
            },
            {
                "id": "old_appliance_collection",
                "title": "Svoz starého elektrospotřebiče",
                "url": "https://www.alza.cz/contact-and-support/svoz-stareho-elektrospotrebice/40",
                "description": "Zákazník chce zajistit odvoz starého elektrospotřebiče nebo řeší jeho neuskutečněný svoz."
            },
            {
                "id": "claim_completed_follow_up",
                "title": "Mám již vyřízenou reklamaci",
                "url": "https://www.alza.cz/contact-and-support/mam-jiz-vyrizenou-reklamaci/41",
                "description": "Zákazník potřebuje vyřešit navazující požadavek týkající se již dokončené reklamace."
            },
            {
                "id": "claim_alzabox_pickup_expired",
                "title": "Nestihl/a jsem vyzvednout vyřízenou reklamaci z AlzaBoxu",
                "url": "https://www.alza.cz/contact-and-support/nestihla-jsem-vyzvednout-vyrizenou-reklamaci-z-alzaboxu/42",
                "description": "Zákazník zmeškal lhůtu pro vyzvednutí vyřízené reklamace z AlzaBoxu a potřebuje další postup."
            }
        ]
    },
    "my_account": {
        "title": "Můj účet",
        "url": "https://www.alza.cz/contact-and-support/muj-ucet/387",
        "description": "Zákazník řeší přístup k účtu, osobní údaje, zasílání nabídek nebo další požadavky dostupné v této sekci.",
        "sub_intents": [
            {
                "id": "account_forgotten_password",
                "title": "Zapomenuté heslo",
                "url": "https://www.alza.cz/contact-and-support/zapomenute-heslo/388",
                "description": "Zákazník zapomněl heslo a potřebuje obnovit přístup ke svému účtu."
            },
            {
                "id": "account_personal_data",
                "title": "Osobní údaje",
                "url": "https://www.alza.cz/contact-and-support/osobni-udaje/391",
                "description": "Zákazník řeší správu, změnu nebo zpracování svých osobních údajů."
            },
            {
                "id": "account_marketing_unsubscribe",
                "title": "Zrušení zasílání akčních nabídek",
                "url": "https://www.alza.cz/contact-and-support/zruseni-zasilani-akcnich-nabidek/390",
                "description": "Zákazník chce přestat dostávat reklamní sdělení a akční nabídky."
            },
            {
                "id": "account_business_purchase",
                "title": "Chci nakoupit na firmu",
                "url": "https://www.alza.cz/contact-and-support/chci-nakoupit-na-firmu/461",
                "description": "Zákazník potřebuje pomoc s nákupem na firmu a používáním firemních údajů v účtu."
            },
            {
                "id": "account_careers",
                "title": "Kariéra v Alza.cz",
                "url": "https://www.alza.cz/contact-and-support/kariera-v-alzacz/454",
                "description": "Zájemce hledá pracovní příležitosti v Alza.cz nebo informace k výběrovému řízení."
            },
            {
                "id": "account_report_illegal_content",
                "title": "Nahlášení nezákonného obsahu",
                "url": "https://www.alza.cz/contact-and-support/nahlaseni-nezakonneho-obsahu/455",
                "description": "Uživatel chce prostřednictvím sekce účtu upozornit na obsah, který považuje za nezákonný."
            }
        ]
    },
    "other": {
        "title": "Ostatní",
        "url": "https://www.alza.cz/contact-and-support/ostatni/17",
        "description": "Zákazník řeší zpětnou vazbu, spolupráci, produktové dotazy nebo jiné požadavky zařazené do sekce Ostatní.",
        "sub_intents": [
            {
                "id": "other_bug_report_or_suggestion",
                "title": "Nahlášení chyby / návrh na zlepšení",
                "url": "https://www.alza.cz/contact-and-support/nahlaseni-chyby-navrh-na-zlepseni/79",
                "description": "Zákazník hlásí chybu webu či služby nebo navrhuje jejich zlepšení."
            },
            {
                "id": "other_cooperation",
                "title": "Spolupráce",
                "url": "https://www.alza.cz/contact-and-support/spoluprace/82",
                "description": "Zájemce chce navázat spolupráci s Alzou nebo projednat její možnosti."
            },
            {
                "id": "other_alza_neo_payment_due_dates",
                "title": "Alza NEO splatnosti",
                "url": "https://www.alza.cz/contact-and-support/alza-neo-splatnosti/92",
                "description": "Zákazník řeší termíny splatnosti nebo úhrady plateb za pronájem Alza NEO."
            },
            {
                "id": "other_compliment_or_complaint",
                "title": "Pochvala nebo stížnost",
                "url": "https://www.alza.cz/contact-and-support/pochvala-nebo-stiznost/96",
                "description": "Zákazník chce předat pochvalu nebo podat stížnost na svou zkušenost s Alzou."
            },
            {
                "id": "other_product_price_verification",
                "title": "Ověření správného nastavení ceny u produktu",
                "url": "https://www.alza.cz/contact-and-support/overeni-spravneho-nastaveni-ceny-u-produktu/108",
                "description": "Zákazník žádá o prověření, zda je u konkrétního produktu uvedena správná cena."
            },
            {
                "id": "other_ring_size_selection",
                "title": "Výběr velikosti prstenu",
                "url": "https://www.alza.cz/contact-and-support/vyber-velikosti-prstenu/111",
                "description": "Zákazník potřebuje poradit s výběrem správné velikosti prstenu."
            },
            {
                "id": "other_diplomatic_sales",
                "title": "Diplomatický prodej",
                "url": "https://www.alza.cz/contact-and-support/diplomaticky-prodej/112",
                "description": "Zákazník potřebuje informace nebo pomoc s nákupem v režimu diplomatického prodeje."
            },
            {
                "id": "other_branch_and_alzabox_reviews",
                "title": "Recenze poboček a Alzaboxů",
                "url": "https://www.alza.cz/contact-and-support/recenze-pobocek-a-alzaboxu/114",
                "description": "Zákazník chce sdílet nebo řešit hodnocení konkrétní pobočky či AlzaBoxu."
            },
            {
                "id": "other_alzee_product_advice",
                "title": "Alzee - dotazy a rady k produktům",
                "url": "https://www.alza.cz/contact-and-support/alzee---dotazy-a-rady-k-produktum/115",
                "description": "Zákazník hledá odpovědi na produktové dotazy nebo radu s výběrem a používáním výrobku."
            },
            {
                "id": "other_trade_in_bank_account",
                "title": "Výkup - číslo účtu",
                "url": "https://www.alza.cz/contact-and-support/vykup---cislo-uctu/456",
                "description": "Zákazník potřebuje sdělit nebo opravit číslo bankovního účtu pro vyplacení částky za výkup."
            },
            {
                "id": "other_eudr_product_information",
                "title": "Žádost o EUDR údaje k produktu",
                "url": "https://www.alza.cz/contact-and-support/zadost-o-eudr-udaje-k-produktu/480",
                "description": "Zákazník žádá o údaje ke konkrétnímu produktu související s požadavky EUDR."
            },
            {
                "id": "other_report_illegal_content",
                "title": "Nahlášení nezákonného obsahu",
                "url": "https://www.alza.cz/contact-and-support/nahlaseni-nezakonneho-obsahu/482",
                "description": "Uživatel chce prostřednictvím sekce Ostatní nahlásit obsah, který považuje za nezákonný."
            }
        ]
    }
}

SCRAPE_METADATA = {'created_at': '2026-10-24T17:10:10.961291+00:00',
 'category_layout': 'user_defined',
 'fallback_verified_on': '2026-10-24',
 'origins': {'order_invoice': 'static_fallback',
             'order_change': 'static_fallback',
             'order_refund': 'static_fallback',
             'order_shipping': 'static_fallback',
             'order_installments': 'static_fallback',
             'services_alzaplus': 'static_fallback',
             'services_warranty': 'static_fallback',
             'services_alzabox_shipping': 'static_fallback',
             'services_gift_wrap': 'static_fallback',
             'claim_create': 'static_fallback',
             'claim_return': 'static_fallback',
             'claim_status': 'static_fallback',
             'claim_damaged_delivery': 'static_fallback',
             'account_password': 'static_fallback',
             'account_personal_data': 'static_fallback',
             'account_unsubscribe': 'static_fallback',
             'other_feedback': 'static_fallback',
             'other_cooperation': 'static_fallback'},
 'requests': [{'url': 'https://m.alza.cz/contact-and-support',
               'status': 'failed',
               'error': '403 Client Error: Forbidden for url: '
                        'https://m.alza.cz/contact-and-support'},
              {'url': 'https://www.alza.cz/kontakt',
               'status': 'failed',
               'error': '403 Client Error: Forbidden for url: https://www.alza.cz/kontakt'},
              {'url': 'https://www.alza.cz/contact/orders.htm',
               'status': 'failed',
               'error': '403 Client Error: Forbidden for url: '
                        'https://www.alza.cz/contact/orders.htm'},
              {'url': 'https://www.alza.cz/contact/services.htm',
               'status': 'failed',
               'error': '403 Client Error: Forbidden for url: '
                        'https://www.alza.cz/contact/services.htm'},
              {'url': 'https://www.alza.cz/kontakt/reklamace?lang=CZ',
               'status': 'failed',
               'error': '403 Client Error: Forbidden for url: '
                        'https://www.alza.cz/kontakt/reklamace?lang=CZ'},
              {'url': 'https://www.alza.cz/contact/others.htm',
               'status': 'failed',
               'error': '403 Client Error: Forbidden for url: '
                        'https://www.alza.cz/contact/others.htm'}]}
