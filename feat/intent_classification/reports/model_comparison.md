# Srovnání klasifikátorů zákaznických intentů

Datum: 09. 11. 2026.

**Nejlepší z vyzkoušených konfigurací podle validačního macro F1: Logistická regrese, C=233.572147.** Dosáhla accuracy **82.68 %** a macro F1 **0.8203**.

Jde o validační srovnání po výběru hyperparametrů. Testovací sada nebyla použita; výsledky nejsou závěrečným testovacím hodnocením ani důkazem převahy v reálném provozu.

## Data a postup

- Trénovací sada: **1211 vět**; validační sada: **231 vět**.
- **57 intentů včetně OOS**, embeddingy mají **768 rozměrů**.
- Embedding model: `google/embeddinggemma-300m`, instrukce `task: classification | query: `, L2 normalizace.
- Zdroj vstupů: `embedding_models/outputs/train_embeddinggemma.npz` a `validation_embeddinggemma.npz`.
- Každá konfigurace se učila pouze na train. Výběr: nejvyšší validační macro F1; při shodě první konfigurace v pořadí smyčky.
- Většina intentů má 21 trénovacích a 4 validační věty; OOS má 35 a 7 vět.

## Celkové výsledky

| Model | Nejlepší nastavení | Správně / celkem | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Logistická regrese | C=233.572147 | 191 / 231 | 82.68 % | 0.8453 | 0.8246 | 0.8203 | 0.8226 |
| MLP | vrstvy (1024, 128) | 187 / 231 | 80.95 % | 0.8194 | 0.8070 | 0.7948 | 0.7975 |
| k-NN | k=10 | 163 / 231 | 70.56 % | 0.7282 | 0.7018 | 0.6955 | 0.6995 |

Macro F1 dává všem intentům stejnou váhu. Weighted F1 váží jejich F1 počtem validačních příkladů. Podíl správně rozpoznaných vět skutečné třídy (požadavek „accuracy per class“) je zde uváděn jako recall; nejde o one-vs-rest accuracy zahrnující i true negatives.

## Prohledaná nastavení

| Model | Prostor hledání | Počet konfigurací | Další nastavení | Čas celého hledání |
|---|---|---:|---|---:|
| Logistická regrese | 20 geometricky rozmístěných C od 0,001 do 1000 | 20 | `solver=lbfgs`, L2, `max_iter=3000` | 101.8 s |
| k-NN | k od 1 do 20 | 20 | stejné váhy sousedů, eukleidovská vzdálenost | 0.3 s |
| MLP | každá vrstva: 2, 4, 8, …, 1024 | 100 | `activation=relu`, `solver=adam`, `alpha=0.0001`, `max_iter=1000`, `random_state=42` | 899.5 s |

Časy zahrnují celé hledání a vyhodnocení, nikoli pouze trénování nejlepšího modelu. Výpočet běžel na CPU. MLP a k-NN měly jedno vlákno BLAS, logistická regrese výchozí počet vláken jako při původním spuštění. Časy proto neslouží k přímému srovnání rychlosti modelů ani jako měření latence nasazení. U normalizovaných vektorů dávají eukleidovská a kosinová vzdálenost stejné pořadí sousedů až na numerické shody; to neznamená stejnou vzdálenost ani stejné vážení podle vzdálenosti.

## OOS a nulové F1

| Model | OOS precision | OOS recall | OOS F1 | Intenty s F1 = 0 |
|---|---:|---:|---:|---|
| Logistická regrese | 1.0000 | 1.0000 | 1.0000 | žádný |
| k-NN | 1.0000 | 1.0000 | 1.0000 | order_change_or_extension, order_pay_in_three, other_compliment_or_complaint, services_pay_in_three |
| MLP | 1.0000 | 1.0000 | 1.0000 | services_device_rental |

OOS obsahuje pouze 7 validačních příkladů z omezeného zdroje obecných otázek. Ani perfektní skóre zde nezaručuje odmítání všech budoucích dotazů mimo doménu.

## Porovnání F1 všech intentů

| Intent | Train | Validation | Logistická regrese F1 | k-NN F1 | MLP F1 |
|---|---:|---:|---:|---:|---:|
| `account_business_purchase` | 21 | 4 | 0.5000 | 0.5455 | 0.6667 |
| `account_careers` | 21 | 4 | 1.0000 | 0.8889 | 0.8889 |
| `account_forgotten_password` | 21 | 4 | 1.0000 | 0.8000 | 1.0000 |
| `account_marketing_unsubscribe` | 21 | 4 | 1.0000 | 1.0000 | 1.0000 |
| `account_personal_data` | 21 | 4 | 1.0000 | 0.8571 | 1.0000 |
| `account_report_illegal_content` | 21 | 4 | 0.8571 | 0.7500 | 0.8571 |
| `claim_alzabox_pickup_expired` | 21 | 4 | 1.0000 | 0.8889 | 1.0000 |
| `claim_completed_follow_up` | 21 | 4 | 0.8000 | 0.8571 | 0.8000 |
| `claim_damaged_shipment` | 21 | 4 | 0.6667 | 0.3636 | 0.7500 |
| `claim_defective_goods` | 21 | 4 | 0.6667 | 0.2857 | 0.6000 |
| `claim_incomplete_shipment` | 21 | 4 | 0.6667 | 0.4000 | 0.7273 |
| `claim_status` | 21 | 4 | 0.5714 | 0.8889 | 0.5714 |
| `claim_wrong_goods` | 21 | 4 | 0.7500 | 0.6667 | 0.8889 |
| `old_appliance_collection` | 21 | 4 | 1.0000 | 0.5000 | 0.8889 |
| `oos` | 35 | 7 | 1.0000 | 1.0000 | 1.0000 |
| `order_business_purchase` | 21 | 4 | 0.5714 | 0.8571 | 0.6667 |
| `order_change_or_extension` | 21 | 4 | 0.5000 | 0.0000 | 0.3333 |
| `order_delay` | 21 | 4 | 0.8571 | 0.6667 | 1.0000 |
| `order_delivery_options` | 21 | 4 | 1.0000 | 1.0000 | 0.8889 |
| `order_device_rental` | 21 | 4 | 0.8571 | 0.5000 | 0.5714 |
| `order_digital_content` | 21 | 4 | 1.0000 | 0.8571 | 1.0000 |
| `order_installment_purchase` | 21 | 4 | 0.8571 | 0.6000 | 0.8571 |
| `order_invoice_or_credit_note_payment` | 21 | 4 | 0.8000 | 1.0000 | 0.8000 |
| `order_invoice_request` | 21 | 4 | 1.0000 | 0.5714 | 0.8571 |
| `order_pay_in_three` | 21 | 4 | 0.5000 | 0.0000 | 0.4444 |
| `order_payment_refund` | 21 | 4 | 0.4000 | 0.5714 | 0.4000 |
| `other_alza_neo_payment_due_dates` | 21 | 4 | 1.0000 | 0.8889 | 0.8889 |
| `other_alzee_product_advice` | 21 | 4 | 0.8571 | 0.6667 | 1.0000 |
| `other_branch_and_alzabox_reviews` | 21 | 4 | 0.8571 | 0.7500 | 0.8571 |
| `other_bug_report_or_suggestion` | 21 | 4 | 0.5714 | 0.5714 | 0.4444 |
| `other_compliment_or_complaint` | 21 | 4 | 0.3333 | 0.0000 | 0.5000 |
| `other_cooperation` | 21 | 4 | 1.0000 | 0.8571 | 0.7500 |
| `other_diplomatic_sales` | 21 | 4 | 1.0000 | 1.0000 | 1.0000 |
| `other_eudr_product_information` | 21 | 4 | 1.0000 | 1.0000 | 1.0000 |
| `other_product_price_verification` | 21 | 4 | 1.0000 | 0.8571 | 1.0000 |
| `other_report_illegal_content` | 21 | 4 | 0.7500 | 0.5714 | 0.7500 |
| `other_ring_size_selection` | 21 | 4 | 0.8889 | 1.0000 | 1.0000 |
| `other_trade_in_bank_account` | 21 | 4 | 1.0000 | 1.0000 | 1.0000 |
| `return_goods` | 21 | 4 | 0.8889 | 0.4444 | 0.6667 |
| `services_alza_benefit` | 21 | 4 | 1.0000 | 0.8571 | 1.0000 |
| `services_alzaplus` | 21 | 4 | 0.8571 | 0.8571 | 0.8571 |
| `services_built_in_appliance_installation` | 21 | 4 | 1.0000 | 0.8571 | 0.8571 |
| `services_damage_and_theft_insurance` | 21 | 4 | 0.7500 | 0.8571 | 0.8571 |
| `services_device_rental` | 21 | 4 | 0.3333 | 0.4000 | 0.0000 |
| `services_extended_warranty` | 21 | 4 | 1.0000 | 1.0000 | 1.0000 |
| `services_gift_wrapping` | 21 | 4 | 1.0000 | 0.6667 | 1.0000 |
| `services_immediate_replacement` | 21 | 4 | 0.6667 | 0.5000 | 0.8000 |
| `services_initial_setup` | 21 | 4 | 0.4444 | 0.3636 | 0.5000 |
| `services_insurance_withdrawal` | 21 | 4 | 1.0000 | 1.0000 | 1.0000 |
| `services_pay_in_three` | 21 | 4 | 0.5000 | 0.0000 | 0.6667 |
| `services_return_within_sixty_days` | 21 | 4 | 0.8571 | 0.8571 | 0.4000 |
| `services_send_via_alzabox` | 21 | 4 | 0.8889 | 0.7500 | 0.8889 |
| `services_shipment_insurance` | 21 | 4 | 0.8571 | 0.7500 | 0.8571 |
| `services_solar_power_systems` | 21 | 4 | 1.0000 | 1.0000 | 1.0000 |
| `services_trade_in` | 21 | 4 | 0.8889 | 0.5714 | 0.6667 |
| `services_tv_wall_mounting` | 21 | 4 | 0.8889 | 0.8889 | 0.8889 |
| `services_warranty_or_replacement_callback` | 21 | 4 | 0.8571 | 0.5455 | 0.7500 |

## Omezení a další krok

- Výběr mnoha konfigurací na stejné validaci může vést k optimistickému skóre; MLP navíc dostalo 100 pokusů, ostatní modely po 20.
- Čtyři validační věty na většinu tříd znamenají, že jediná chyba změní recall o 25 procentních bodů. Malé rozdíly mezi modely nelze považovat za prokázanou převahu.
- MLP bylo hodnoceno s jedním seedem. Stabilita napříč inicializacemi ani cross-validation zde nebyly měřeny.
- Některé kategorie se významově překrývají, například pronájem zařízení nebo stejné téma v sekci objednávek a služeb.
- Po uzavření konfigurací provést závěrečné testovací vyhodnocení. Test nepoužívat k dalšímu ladění.

## Podrobné reporty a nejčastější záměny

<details><summary>Logistická regrese: precision, recall, F1 a záměny</summary>

| Intent | Precision | Recall / správnost ve třídě | F1 | Support |
|---|---:|---:|---:|---:|
| `account_business_purchase` | 0.5000 | 0.5000 | 0.5000 | 4 |
| `account_careers` | 1.0000 | 1.0000 | 1.0000 | 4 |
| `account_forgotten_password` | 1.0000 | 1.0000 | 1.0000 | 4 |
| `account_marketing_unsubscribe` | 1.0000 | 1.0000 | 1.0000 | 4 |
| `account_personal_data` | 1.0000 | 1.0000 | 1.0000 | 4 |
| `account_report_illegal_content` | 1.0000 | 0.7500 | 0.8571 | 4 |
| `claim_alzabox_pickup_expired` | 1.0000 | 1.0000 | 1.0000 | 4 |
| `claim_completed_follow_up` | 0.6667 | 1.0000 | 0.8000 | 4 |
| `claim_damaged_shipment` | 0.6000 | 0.7500 | 0.6667 | 4 |
| `claim_defective_goods` | 0.6000 | 0.7500 | 0.6667 | 4 |
| `claim_incomplete_shipment` | 0.5000 | 1.0000 | 0.6667 | 4 |
| `claim_status` | 0.6667 | 0.5000 | 0.5714 | 4 |
| `claim_wrong_goods` | 0.7500 | 0.7500 | 0.7500 | 4 |
| `old_appliance_collection` | 1.0000 | 1.0000 | 1.0000 | 4 |
| `oos` | 1.0000 | 1.0000 | 1.0000 | 7 |
| `order_business_purchase` | 0.6667 | 0.5000 | 0.5714 | 4 |
| `order_change_or_extension` | 0.5000 | 0.5000 | 0.5000 | 4 |
| `order_delay` | 1.0000 | 0.7500 | 0.8571 | 4 |
| `order_delivery_options` | 1.0000 | 1.0000 | 1.0000 | 4 |
| `order_device_rental` | 1.0000 | 0.7500 | 0.8571 | 4 |
| `order_digital_content` | 1.0000 | 1.0000 | 1.0000 | 4 |
| `order_installment_purchase` | 1.0000 | 0.7500 | 0.8571 | 4 |
| `order_invoice_or_credit_note_payment` | 0.6667 | 1.0000 | 0.8000 | 4 |
| `order_invoice_request` | 1.0000 | 1.0000 | 1.0000 | 4 |
| `order_pay_in_three` | 0.5000 | 0.5000 | 0.5000 | 4 |
| `order_payment_refund` | 1.0000 | 0.2500 | 0.4000 | 4 |
| `other_alza_neo_payment_due_dates` | 1.0000 | 1.0000 | 1.0000 | 4 |
| `other_alzee_product_advice` | 1.0000 | 0.7500 | 0.8571 | 4 |
| `other_branch_and_alzabox_reviews` | 1.0000 | 0.7500 | 0.8571 | 4 |
| `other_bug_report_or_suggestion` | 0.6667 | 0.5000 | 0.5714 | 4 |
| `other_compliment_or_complaint` | 0.5000 | 0.2500 | 0.3333 | 4 |
| `other_cooperation` | 1.0000 | 1.0000 | 1.0000 | 4 |
| `other_diplomatic_sales` | 1.0000 | 1.0000 | 1.0000 | 4 |
| `other_eudr_product_information` | 1.0000 | 1.0000 | 1.0000 | 4 |
| `other_product_price_verification` | 1.0000 | 1.0000 | 1.0000 | 4 |
| `other_report_illegal_content` | 0.7500 | 0.7500 | 0.7500 | 4 |
| `other_ring_size_selection` | 0.8000 | 1.0000 | 0.8889 | 4 |
| `other_trade_in_bank_account` | 1.0000 | 1.0000 | 1.0000 | 4 |
| `return_goods` | 0.8000 | 1.0000 | 0.8889 | 4 |
| `services_alza_benefit` | 1.0000 | 1.0000 | 1.0000 | 4 |
| `services_alzaplus` | 1.0000 | 0.7500 | 0.8571 | 4 |
| `services_built_in_appliance_installation` | 1.0000 | 1.0000 | 1.0000 | 4 |
| `services_damage_and_theft_insurance` | 0.7500 | 0.7500 | 0.7500 | 4 |
| `services_device_rental` | 0.5000 | 0.2500 | 0.3333 | 4 |
| `services_extended_warranty` | 1.0000 | 1.0000 | 1.0000 | 4 |
| `services_gift_wrapping` | 1.0000 | 1.0000 | 1.0000 | 4 |
| `services_immediate_replacement` | 0.5000 | 1.0000 | 0.6667 | 4 |
| `services_initial_setup` | 0.4000 | 0.5000 | 0.4444 | 4 |
| `services_insurance_withdrawal` | 1.0000 | 1.0000 | 1.0000 | 4 |
| `services_pay_in_three` | 0.5000 | 0.5000 | 0.5000 | 4 |
| `services_return_within_sixty_days` | 1.0000 | 0.7500 | 0.8571 | 4 |
| `services_send_via_alzabox` | 0.8000 | 1.0000 | 0.8889 | 4 |
| `services_shipment_insurance` | 1.0000 | 0.7500 | 0.8571 | 4 |
| `services_solar_power_systems` | 1.0000 | 1.0000 | 1.0000 | 4 |
| `services_trade_in` | 0.8000 | 1.0000 | 0.8889 | 4 |
| `services_tv_wall_mounting` | 0.8000 | 1.0000 | 0.8889 | 4 |
| `services_warranty_or_replacement_callback` | 1.0000 | 0.7500 | 0.8571 | 4 |

Nejčastější směrované záměny:

| Skutečný intent | Předpovězený intent | Počet |
|---|---|---:|
| `services_device_rental` | `services_immediate_replacement` | 2 |
| `services_initial_setup` | `claim_incomplete_shipment` | 2 |
| `services_pay_in_three` | `order_pay_in_three` | 2 |
| `order_business_purchase` | `account_business_purchase` | 2 |
| `services_return_within_sixty_days` | `services_initial_setup` | 1 |
| `other_report_illegal_content` | `claim_defective_goods` | 1 |
| `order_delay` | `claim_incomplete_shipment` | 1 |
| `order_pay_in_three` | `other_bug_report_or_suggestion` | 1 |
| `account_report_illegal_content` | `other_report_illegal_content` | 1 |
| `account_business_purchase` | `order_business_purchase` | 1 |

</details>

<details><summary>k-NN: precision, recall, F1 a záměny</summary>

| Intent | Precision | Recall / správnost ve třídě | F1 | Support |
|---|---:|---:|---:|---:|
| `account_business_purchase` | 0.4286 | 0.7500 | 0.5455 | 4 |
| `account_careers` | 0.8000 | 1.0000 | 0.8889 | 4 |
| `account_forgotten_password` | 0.6667 | 1.0000 | 0.8000 | 4 |
| `account_marketing_unsubscribe` | 1.0000 | 1.0000 | 1.0000 | 4 |
| `account_personal_data` | 1.0000 | 0.7500 | 0.8571 | 4 |
| `account_report_illegal_content` | 0.7500 | 0.7500 | 0.7500 | 4 |
| `claim_alzabox_pickup_expired` | 0.8000 | 1.0000 | 0.8889 | 4 |
| `claim_completed_follow_up` | 1.0000 | 0.7500 | 0.8571 | 4 |
| `claim_damaged_shipment` | 0.2857 | 0.5000 | 0.3636 | 4 |
| `claim_defective_goods` | 0.3333 | 0.2500 | 0.2857 | 4 |
| `claim_incomplete_shipment` | 0.2727 | 0.7500 | 0.4000 | 4 |
| `claim_status` | 0.8000 | 1.0000 | 0.8889 | 4 |
| `claim_wrong_goods` | 0.6000 | 0.7500 | 0.6667 | 4 |
| `old_appliance_collection` | 0.3750 | 0.7500 | 0.5000 | 4 |
| `oos` | 1.0000 | 1.0000 | 1.0000 | 7 |
| `order_business_purchase` | 1.0000 | 0.7500 | 0.8571 | 4 |
| `order_change_or_extension` | 0.0000 | 0.0000 | 0.0000 | 4 |
| `order_delay` | 0.6000 | 0.7500 | 0.6667 | 4 |
| `order_delivery_options` | 1.0000 | 1.0000 | 1.0000 | 4 |
| `order_device_rental` | 0.5000 | 0.5000 | 0.5000 | 4 |
| `order_digital_content` | 1.0000 | 0.7500 | 0.8571 | 4 |
| `order_installment_purchase` | 0.5000 | 0.7500 | 0.6000 | 4 |
| `order_invoice_or_credit_note_payment` | 1.0000 | 1.0000 | 1.0000 | 4 |
| `order_invoice_request` | 0.6667 | 0.5000 | 0.5714 | 4 |
| `order_pay_in_three` | 0.0000 | 0.0000 | 0.0000 | 4 |
| `order_payment_refund` | 0.6667 | 0.5000 | 0.5714 | 4 |
| `other_alza_neo_payment_due_dates` | 0.8000 | 1.0000 | 0.8889 | 4 |
| `other_alzee_product_advice` | 1.0000 | 0.5000 | 0.6667 | 4 |
| `other_branch_and_alzabox_reviews` | 0.7500 | 0.7500 | 0.7500 | 4 |
| `other_bug_report_or_suggestion` | 0.6667 | 0.5000 | 0.5714 | 4 |
| `other_compliment_or_complaint` | 0.0000 | 0.0000 | 0.0000 | 4 |
| `other_cooperation` | 1.0000 | 0.7500 | 0.8571 | 4 |
| `other_diplomatic_sales` | 1.0000 | 1.0000 | 1.0000 | 4 |
| `other_eudr_product_information` | 1.0000 | 1.0000 | 1.0000 | 4 |
| `other_product_price_verification` | 1.0000 | 0.7500 | 0.8571 | 4 |
| `other_report_illegal_content` | 0.6667 | 0.5000 | 0.5714 | 4 |
| `other_ring_size_selection` | 1.0000 | 1.0000 | 1.0000 | 4 |
| `other_trade_in_bank_account` | 1.0000 | 1.0000 | 1.0000 | 4 |
| `return_goods` | 0.4000 | 0.5000 | 0.4444 | 4 |
| `services_alza_benefit` | 1.0000 | 0.7500 | 0.8571 | 4 |
| `services_alzaplus` | 1.0000 | 0.7500 | 0.8571 | 4 |
| `services_built_in_appliance_installation` | 1.0000 | 0.7500 | 0.8571 | 4 |
| `services_damage_and_theft_insurance` | 1.0000 | 0.7500 | 0.8571 | 4 |
| `services_device_rental` | 1.0000 | 0.2500 | 0.4000 | 4 |
| `services_extended_warranty` | 1.0000 | 1.0000 | 1.0000 | 4 |
| `services_gift_wrapping` | 1.0000 | 0.5000 | 0.6667 | 4 |
| `services_immediate_replacement` | 0.5000 | 0.5000 | 0.5000 | 4 |
| `services_initial_setup` | 0.2857 | 0.5000 | 0.3636 | 4 |
| `services_insurance_withdrawal` | 1.0000 | 1.0000 | 1.0000 | 4 |
| `services_pay_in_three` | 0.0000 | 0.0000 | 0.0000 | 4 |
| `services_return_within_sixty_days` | 1.0000 | 0.7500 | 0.8571 | 4 |
| `services_send_via_alzabox` | 0.7500 | 0.7500 | 0.7500 | 4 |
| `services_shipment_insurance` | 0.7500 | 0.7500 | 0.7500 | 4 |
| `services_solar_power_systems` | 1.0000 | 1.0000 | 1.0000 | 4 |
| `services_trade_in` | 0.6667 | 0.5000 | 0.5714 | 4 |
| `services_tv_wall_mounting` | 0.8000 | 1.0000 | 0.8889 | 4 |
| `services_warranty_or_replacement_callback` | 0.4286 | 0.7500 | 0.5455 | 4 |

Nejčastější směrované záměny:

| Skutečný intent | Předpovězený intent | Počet |
|---|---|---:|
| `order_pay_in_three` | `services_pay_in_three` | 2 |
| `services_device_rental` | `services_warranty_or_replacement_callback` | 2 |
| `other_alzee_product_advice` | `services_initial_setup` | 2 |
| `services_pay_in_three` | `order_pay_in_three` | 2 |
| `services_pay_in_three` | `order_invoice_request` | 1 |
| `services_trade_in` | `claim_incomplete_shipment` | 1 |
| `other_compliment_or_complaint` | `claim_incomplete_shipment` | 1 |
| `claim_defective_goods` | `services_immediate_replacement` | 1 |
| `services_return_within_sixty_days` | `order_device_rental` | 1 |
| `claim_incomplete_shipment` | `claim_damaged_shipment` | 1 |

</details>

<details><summary>MLP: precision, recall, F1 a záměny</summary>

| Intent | Precision | Recall / správnost ve třídě | F1 | Support |
|---|---:|---:|---:|---:|
| `account_business_purchase` | 0.6000 | 0.7500 | 0.6667 | 4 |
| `account_careers` | 0.8000 | 1.0000 | 0.8889 | 4 |
| `account_forgotten_password` | 1.0000 | 1.0000 | 1.0000 | 4 |
| `account_marketing_unsubscribe` | 1.0000 | 1.0000 | 1.0000 | 4 |
| `account_personal_data` | 1.0000 | 1.0000 | 1.0000 | 4 |
| `account_report_illegal_content` | 1.0000 | 0.7500 | 0.8571 | 4 |
| `claim_alzabox_pickup_expired` | 1.0000 | 1.0000 | 1.0000 | 4 |
| `claim_completed_follow_up` | 0.6667 | 1.0000 | 0.8000 | 4 |
| `claim_damaged_shipment` | 0.7500 | 0.7500 | 0.7500 | 4 |
| `claim_defective_goods` | 0.5000 | 0.7500 | 0.6000 | 4 |
| `claim_incomplete_shipment` | 0.5714 | 1.0000 | 0.7273 | 4 |
| `claim_status` | 0.6667 | 0.5000 | 0.5714 | 4 |
| `claim_wrong_goods` | 0.8000 | 1.0000 | 0.8889 | 4 |
| `old_appliance_collection` | 0.8000 | 1.0000 | 0.8889 | 4 |
| `oos` | 1.0000 | 1.0000 | 1.0000 | 7 |
| `order_business_purchase` | 1.0000 | 0.5000 | 0.6667 | 4 |
| `order_change_or_extension` | 0.5000 | 0.2500 | 0.3333 | 4 |
| `order_delay` | 1.0000 | 1.0000 | 1.0000 | 4 |
| `order_delivery_options` | 0.8000 | 1.0000 | 0.8889 | 4 |
| `order_device_rental` | 0.6667 | 0.5000 | 0.5714 | 4 |
| `order_digital_content` | 1.0000 | 1.0000 | 1.0000 | 4 |
| `order_installment_purchase` | 1.0000 | 0.7500 | 0.8571 | 4 |
| `order_invoice_or_credit_note_payment` | 0.6667 | 1.0000 | 0.8000 | 4 |
| `order_invoice_request` | 1.0000 | 0.7500 | 0.8571 | 4 |
| `order_pay_in_three` | 0.4000 | 0.5000 | 0.4444 | 4 |
| `order_payment_refund` | 1.0000 | 0.2500 | 0.4000 | 4 |
| `other_alza_neo_payment_due_dates` | 0.8000 | 1.0000 | 0.8889 | 4 |
| `other_alzee_product_advice` | 1.0000 | 1.0000 | 1.0000 | 4 |
| `other_branch_and_alzabox_reviews` | 1.0000 | 0.7500 | 0.8571 | 4 |
| `other_bug_report_or_suggestion` | 0.4000 | 0.5000 | 0.4444 | 4 |
| `other_compliment_or_complaint` | 0.5000 | 0.5000 | 0.5000 | 4 |
| `other_cooperation` | 0.7500 | 0.7500 | 0.7500 | 4 |
| `other_diplomatic_sales` | 1.0000 | 1.0000 | 1.0000 | 4 |
| `other_eudr_product_information` | 1.0000 | 1.0000 | 1.0000 | 4 |
| `other_product_price_verification` | 1.0000 | 1.0000 | 1.0000 | 4 |
| `other_report_illegal_content` | 0.7500 | 0.7500 | 0.7500 | 4 |
| `other_ring_size_selection` | 1.0000 | 1.0000 | 1.0000 | 4 |
| `other_trade_in_bank_account` | 1.0000 | 1.0000 | 1.0000 | 4 |
| `return_goods` | 0.6000 | 0.7500 | 0.6667 | 4 |
| `services_alza_benefit` | 1.0000 | 1.0000 | 1.0000 | 4 |
| `services_alzaplus` | 1.0000 | 0.7500 | 0.8571 | 4 |
| `services_built_in_appliance_installation` | 1.0000 | 0.7500 | 0.8571 | 4 |
| `services_damage_and_theft_insurance` | 1.0000 | 0.7500 | 0.8571 | 4 |
| `services_device_rental` | 0.0000 | 0.0000 | 0.0000 | 4 |
| `services_extended_warranty` | 1.0000 | 1.0000 | 1.0000 | 4 |
| `services_gift_wrapping` | 1.0000 | 1.0000 | 1.0000 | 4 |
| `services_immediate_replacement` | 0.6667 | 1.0000 | 0.8000 | 4 |
| `services_initial_setup` | 0.5000 | 0.5000 | 0.5000 | 4 |
| `services_insurance_withdrawal` | 1.0000 | 1.0000 | 1.0000 | 4 |
| `services_pay_in_three` | 0.6000 | 0.7500 | 0.6667 | 4 |
| `services_return_within_sixty_days` | 1.0000 | 0.2500 | 0.4000 | 4 |
| `services_send_via_alzabox` | 0.8000 | 1.0000 | 0.8889 | 4 |
| `services_shipment_insurance` | 1.0000 | 0.7500 | 0.8571 | 4 |
| `services_solar_power_systems` | 1.0000 | 1.0000 | 1.0000 | 4 |
| `services_trade_in` | 0.6000 | 0.7500 | 0.6667 | 4 |
| `services_tv_wall_mounting` | 0.8000 | 1.0000 | 0.8889 | 4 |
| `services_warranty_or_replacement_callback` | 0.7500 | 0.7500 | 0.7500 | 4 |

Nejčastější směrované záměny:

| Skutečný intent | Předpovězený intent | Počet |
|---|---|---:|
| `services_device_rental` | `services_immediate_replacement` | 2 |
| `services_initial_setup` | `claim_incomplete_shipment` | 2 |
| `other_bug_report_or_suggestion` | `other_compliment_or_complaint` | 2 |
| `order_business_purchase` | `account_business_purchase` | 2 |
| `services_device_rental` | `order_device_rental` | 1 |
| `services_return_within_sixty_days` | `services_initial_setup` | 1 |
| `other_report_illegal_content` | `claim_defective_goods` | 1 |
| `services_return_within_sixty_days` | `services_trade_in` | 1 |
| `order_pay_in_three` | `other_bug_report_or_suggestion` | 1 |
| `account_report_illegal_content` | `other_report_illegal_content` | 1 |

</details>

## Všechny vyzkoušené konfigurace

<details><summary>Logistická regrese: 20 konfigurací</summary>

| Parametry | Accuracy | Macro F1 | Epochy MLP | Varování při trénování |
|---|---:|---:|---:|---|
| C=0.001 | 3.03 % | 0.0010 | — | — |
| C=0.00206913808 | 3.03 % | 0.0010 | — | — |
| C=0.0042813324 | 3.03 % | 0.0010 | — | — |
| C=0.0088586679 | 3.03 % | 0.0010 | — | — |
| C=0.0183298071 | 3.03 % | 0.0010 | — | — |
| C=0.0379269019 | 3.03 % | 0.0010 | — | — |
| C=0.078475997 | 3.03 % | 0.0010 | — | — |
| C=0.162377674 | 48.92 % | 0.5158 | — | — |
| C=0.335981829 | 71.86 % | 0.6864 | — | — |
| C=0.695192796 | 75.76 % | 0.7301 | — | — |
| C=1.43844989 | 78.79 % | 0.7683 | — | — |
| C=2.97635144 | 80.09 % | 0.7836 | — | — |
| C=6.15848211 | 82.25 % | 0.8098 | — | — |
| C=12.7427499 | 82.25 % | 0.8107 | — | — |
| C=26.366509 | 81.82 % | 0.8082 | — | — |
| C=54.5559478 | 81.82 % | 0.8095 | — | — |
| C=112.883789 | 82.25 % | 0.8154 | — | — |
| C=233.572147 | 82.68 % | 0.8203 | — | — |
| C=483.293024 | 82.68 % | 0.8171 | — | — |
| C=1000 | 82.25 % | 0.8136 | — | — |

</details>

<details><summary>k-NN: 20 konfigurací</summary>

| Parametry | Accuracy | Macro F1 | Epochy MLP | Varování při trénování |
|---|---:|---:|---:|---|
| k=1 | 63.64 % | 0.6338 | — | — |
| k=2 | 58.87 % | 0.5751 | — | — |
| k=3 | 66.67 % | 0.6687 | — | — |
| k=4 | 65.80 % | 0.6446 | — | — |
| k=5 | 67.10 % | 0.6576 | — | — |
| k=6 | 67.10 % | 0.6567 | — | — |
| k=7 | 66.67 % | 0.6500 | — | — |
| k=8 | 67.10 % | 0.6586 | — | — |
| k=9 | 70.13 % | 0.6870 | — | — |
| k=10 | 70.56 % | 0.6955 | — | — |
| k=11 | 69.70 % | 0.6893 | — | — |
| k=12 | 70.13 % | 0.6908 | — | — |
| k=13 | 70.13 % | 0.6945 | — | — |
| k=14 | 68.83 % | 0.6718 | — | — |
| k=15 | 68.40 % | 0.6736 | — | — |
| k=16 | 69.70 % | 0.6844 | — | — |
| k=17 | 68.40 % | 0.6743 | — | — |
| k=18 | 68.83 % | 0.6797 | — | — |
| k=19 | 67.97 % | 0.6703 | — | — |
| k=20 | 67.97 % | 0.6721 | — | — |

</details>

<details><summary>MLP: 100 konfigurací</summary>

| Parametry | Accuracy | Macro F1 | Epochy MLP | Varování při trénování |
|---|---:|---:|---:|---|
| (2, 2) | 6.06 % | 0.0145 | 1000 | ano |
| (2, 4) | 22.08 % | 0.1875 | 1000 | ano |
| (2, 8) | 23.81 % | 0.2038 | 1000 | ano |
| (2, 16) | 7.79 % | 0.0270 | 1000 | ano |
| (2, 32) | 20.35 % | 0.1681 | 1000 | ano |
| (2, 64) | 22.51 % | 0.1806 | 777 | — |
| (2, 128) | 19.48 % | 0.1573 | 1000 | ano |
| (2, 256) | 21.65 % | 0.1934 | 906 | — |
| (2, 512) | 22.51 % | 0.1808 | 713 | — |
| (2, 1024) | 24.24 % | 0.2109 | 495 | — |
| (4, 2) | 18.18 % | 0.1272 | 1000 | ano |
| (4, 4) | 32.47 % | 0.3057 | 1000 | ano |
| (4, 8) | 35.50 % | 0.3447 | 1000 | ano |
| (4, 16) | 47.19 % | 0.4613 | 1000 | ano |
| (4, 32) | 46.32 % | 0.4446 | 1000 | ano |
| (4, 64) | 51.52 % | 0.4930 | 782 | — |
| (4, 128) | 38.10 % | 0.3630 | 992 | — |
| (4, 256) | 48.48 % | 0.4765 | 499 | — |
| (4, 512) | 52.38 % | 0.5174 | 503 | — |
| (4, 1024) | 49.35 % | 0.4776 | 474 | — |
| (8, 2) | 3.03 % | 0.0010 | 63 | — |
| (8, 4) | 31.60 % | 0.2763 | 990 | — |
| (8, 8) | 49.35 % | 0.4905 | 1000 | ano |
| (8, 16) | 57.14 % | 0.5588 | 1000 | ano |
| (8, 32) | 56.28 % | 0.5575 | 1000 | ano |
| (8, 64) | 64.50 % | 0.6341 | 1000 | ano |
| (8, 128) | 54.98 % | 0.5404 | 1000 | ano |
| (8, 256) | 61.47 % | 0.5996 | 1000 | ano |
| (8, 512) | 60.61 % | 0.5958 | 877 | — |
| (8, 1024) | 64.50 % | 0.6309 | 576 | — |
| (16, 2) | 15.58 % | 0.1028 | 1000 | ano |
| (16, 4) | 6.93 % | 0.0249 | 1000 | ano |
| (16, 8) | 60.61 % | 0.5937 | 1000 | ano |
| (16, 16) | 68.83 % | 0.6732 | 1000 | ano |
| (16, 32) | 71.00 % | 0.6976 | 1000 | ano |
| (16, 64) | 71.86 % | 0.7110 | 742 | — |
| (16, 128) | 73.16 % | 0.7241 | 780 | — |
| (16, 256) | 73.16 % | 0.7193 | 664 | — |
| (16, 512) | 71.43 % | 0.7001 | 596 | — |
| (16, 1024) | 69.26 % | 0.6839 | 494 | — |
| (32, 2) | 14.72 % | 0.1070 | 1000 | ano |
| (32, 4) | 47.19 % | 0.4506 | 1000 | ano |
| (32, 8) | 57.58 % | 0.5694 | 1000 | ano |
| (32, 16) | 68.40 % | 0.6693 | 929 | — |
| (32, 32) | 71.43 % | 0.7062 | 806 | — |
| (32, 64) | 74.46 % | 0.7363 | 662 | — |
| (32, 128) | 71.00 % | 0.6978 | 581 | — |
| (32, 256) | 74.03 % | 0.7276 | 528 | — |
| (32, 512) | 71.00 % | 0.6981 | 445 | — |
| (32, 1024) | 75.32 % | 0.7382 | 374 | — |
| (64, 2) | 18.18 % | 0.1386 | 806 | — |
| (64, 4) | 11.69 % | 0.0662 | 554 | — |
| (64, 8) | 56.28 % | 0.5504 | 753 | — |
| (64, 16) | 66.23 % | 0.6521 | 767 | — |
| (64, 32) | 75.32 % | 0.7489 | 701 | — |
| (64, 64) | 74.89 % | 0.7411 | 585 | — |
| (64, 128) | 73.16 % | 0.7245 | 471 | — |
| (64, 256) | 73.16 % | 0.7220 | 396 | — |
| (64, 512) | 70.13 % | 0.6870 | 388 | — |
| (64, 1024) | 70.56 % | 0.6954 | 322 | — |
| (128, 2) | 8.66 % | 0.0264 | 699 | — |
| (128, 4) | 43.29 % | 0.4005 | 616 | — |
| (128, 8) | 65.80 % | 0.6445 | 923 | — |
| (128, 16) | 70.56 % | 0.6975 | 597 | — |
| (128, 32) | 76.62 % | 0.7565 | 550 | — |
| (128, 64) | 78.35 % | 0.7722 | 495 | — |
| (128, 128) | 77.49 % | 0.7669 | 417 | — |
| (128, 256) | 77.06 % | 0.7633 | 327 | — |
| (128, 512) | 77.06 % | 0.7624 | 317 | — |
| (128, 1024) | 74.89 % | 0.7396 | 258 | — |
| (256, 2) | 17.75 % | 0.1154 | 798 | — |
| (256, 4) | 44.16 % | 0.4426 | 709 | — |
| (256, 8) | 55.41 % | 0.5524 | 914 | — |
| (256, 16) | 71.43 % | 0.7043 | 521 | — |
| (256, 32) | 74.89 % | 0.7468 | 449 | — |
| (256, 64) | 74.03 % | 0.7251 | 376 | — |
| (256, 128) | 80.09 % | 0.7925 | 328 | — |
| (256, 256) | 79.22 % | 0.7781 | 309 | — |
| (256, 512) | 73.59 % | 0.7248 | 237 | — |
| (256, 1024) | 75.76 % | 0.7504 | 205 | — |
| (512, 2) | 3.03 % | 0.0010 | 115 | — |
| (512, 4) | 30.30 % | 0.3014 | 842 | — |
| (512, 8) | 40.26 % | 0.3881 | 1000 | ano |
| (512, 16) | 64.94 % | 0.6287 | 461 | — |
| (512, 32) | 74.46 % | 0.7375 | 393 | — |
| (512, 64) | 75.76 % | 0.7479 | 329 | — |
| (512, 128) | 79.22 % | 0.7864 | 290 | — |
| (512, 256) | 77.06 % | 0.7580 | 249 | — |
| (512, 512) | 74.89 % | 0.7377 | 222 | — |
| (512, 1024) | 77.06 % | 0.7607 | 189 | — |
| (1024, 2) | 20.78 % | 0.1455 | 442 | — |
| (1024, 4) | 3.03 % | 0.0010 | 115 | — |
| (1024, 8) | 55.41 % | 0.5335 | 550 | — |
| (1024, 16) | 70.13 % | 0.6889 | 431 | — |
| (1024, 32) | 74.46 % | 0.7368 | 333 | — |
| (1024, 64) | 77.49 % | 0.7617 | 290 | — |
| (1024, 128) | 80.95 % | 0.7948 | 244 | — |
| (1024, 256) | 78.35 % | 0.7737 | 206 | — |
| (1024, 512) | 78.35 % | 0.7720 | 193 | — |
| (1024, 1024) | 77.92 % | 0.7682 | 164 | — |

</details>

MLP: varování během trénování u 26 ze 100 konfigurací. Nejlepší konfigurace skončila po 244 epochách; varování: ne. Varování o dosažení limitu znamená, že optimalizátor před limitem nesplnil kritérium konvergence.

## Reprodukovatelnost

- Verze: Python 3.12.7, scikit-learn 1.9.1, NumPy 2.5.4, SciPy 1.18.1.
- Původní trénovací skripty nebyly tímto měřením změněny. Jejich SHA-256 otisky jsou v JSON výsledcích.
- [Úplné výsledky JSON](model_results.json): všechna nastavení, metriky, matice záměn, předpovědi a chybně klasifikované věty.
- Textové protokoly: [logistická regrese](train_logistic_regression.log), [k-NN](train_KNN.log), [MLP](train_mlp.log).

Kontrola numerické stability: při doplňkovém opakování logistické regrese s jedním vláknem BLAS vyšlo nejlepší C=112.883789, accuracy 82.25 % a macro F1 0.8154. Souhrnná tabulka používá běh s výchozím počtem vláken, který zopakoval původní výsledek. Rozdíl upozorňuje na citlivost vybraného C na numerický průběh optimalizace a malé validační rozdíly. [Protokol kontrolního běhu](train_logistic_regression_single_thread.log).
