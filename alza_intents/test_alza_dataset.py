"""Lokální testy bez placeného API a bez požadavků na web Alzy."""
from collections import Counter
import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

import config

from alza_dataset import (
    STYLES, add_fallback, discover, duplicate, extract, fingerprint,
    generate, split_rows, valid_style, LLM, object_schema,
)


class ProviderTests(unittest.TestCase):
    def test_all_provider_endpoints_and_models(self):
        schema = object_schema({"answer": {"type": "string"}})
        for provider in config.PROVIDERS:
            settings = copy.deepcopy(config.PROVIDERS)
            settings[provider]["api_key"] = "test-key" if provider != "ollama" else ""
            with self.subTest(provider=provider), patch.object(config, "ACTIVE_PROVIDER", provider), \
                    patch.object(config, "PROVIDERS", settings), patch("alza_dataset.requests.Session") as session:
                response = session.return_value.post.return_value
                response.status_code = 200
                response.ok = True
                response.json.return_value = {"choices": [{"finish_reason": "stop", "message": {
                    "content": json.dumps({"answer": "ano"})}}]}
                llm = LLM()
                self.assertEqual(llm.ask("Dotaz", schema), {"answer": "ano"})
                call = session.return_value.post.call_args
                self.assertEqual(call.args[0], settings[provider]["base_url"] + "/chat/completions")
                self.assertEqual(call.kwargs["json"]["model"], settings[provider]["model"])
                self.assertEqual(call.kwargs["json"]["response_format"]["json_schema"]["schema"], schema)
                if provider == "ollama":
                    session.return_value.headers.update.assert_not_called()

    def test_missing_keys_and_unknown_provider_fail_before_connection(self):
        for provider in ("gemini", "groq", "openai", "unknown"):
            settings = copy.deepcopy(config.PROVIDERS)
            if provider in settings:
                settings[provider]["api_key"] = "   "
            with self.subTest(provider=provider), patch.object(config, "ACTIVE_PROVIDER", provider), \
                    patch.object(config, "PROVIDERS", settings), patch("alza_dataset.requests.Session") as session:
                with self.assertRaisesRegex(ValueError, "config.py"):
                    LLM()
                session.assert_not_called()

    def test_json_mode_keeps_schema_validation_and_retries(self):
        settings = copy.deepcopy(config.PROVIDERS)
        settings["ollama"]["response_format"] = "json_object"
        with patch.object(config, "ACTIVE_PROVIDER", "ollama"), \
                patch.object(config, "PROVIDERS", settings), patch("alza_dataset.time.sleep"):
            llm = LLM()
            llm.session = Mock()
            response = llm.session.post.return_value
            response.status_code = 200
            response.ok = True
            response.json.side_effect = [
                {"choices": [{"finish_reason": "stop", "message": {"content": '{"wrong": 1}'}}]},
                {"choices": [{"finish_reason": "stop", "message": {"content": '{"answer": "ano"}'}}]},
            ]
            schema = object_schema({"answer": {"type": "string"}})
            self.assertEqual(llm.ask("Dotaz", schema), {"answer": "ano"})
            self.assertEqual(llm.session.post.call_count, 2)
            body = llm.session.post.call_args.kwargs["json"]
            self.assertEqual(body["response_format"], {"type": "json_object"})
            self.assertIn('"answer"', body["messages"][0]["content"])

    def test_http_auth_error_is_friendly_and_does_not_echo_body(self):
        with patch.object(config, "ACTIVE_PROVIDER", "ollama"):
            llm = LLM()
        llm.session = Mock()
        response = llm.session.post.return_value
        response.status_code, response.ok = 401, False
        response.text = "DO-NOT-LOG-SECRET"
        with self.assertRaisesRegex(RuntimeError, "API klíč není platný") as error:
            llm.ask("Dotaz", object_schema({}))
        self.assertNotIn("DO-NOT-LOG-SECRET", str(error.exception))
        self.assertEqual(llm.session.post.call_count, 1)

    def test_entrypoint_missing_key_has_no_traceback(self):
        # Vlastní proces, aby se ověřil i __main__ a jeho návratový kód.
        code = "import config, runpy; config.ACTIVE_PROVIDER='gemini'; config.PROVIDERS['gemini']['api_key']=''; runpy.run_module('alza_dataset', run_name='__main__')"
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                                cwd=Path(__file__).resolve().parent)
        self.assertEqual(result.returncode, 1)
        self.assertIn("GEMINI_API_KEY", result.stderr)
        self.assertNotIn("Traceback", result.stderr)


class PipelineTests(unittest.TestCase):
    def test_extract_dynamic_form_and_json_ld(self):
        html = '''<header><a>Nezajímavá navigace</a></header><main>
        <h2>Pomoc s reklamací</h2><span>Schránka nejde otevřít</span>
        <a href="/contact/orders.htm#help">Úprava objednávky</a>
        <a href="https://example.org/contact">Externí formulář</a>
        <div class="cookie-banner">Pomoc s cookies</div></main>
        <script type="application/ld+json">{"@type":"FAQPage","mainEntity":[
        {"@type":"Question","name":"Kde najdu svoji fakturu?"}]}</script>'''
        topics, links = extract(html, "https://www.alza.cz/kontakt")
        self.assertIn("Schránka nejde otevřít", topics)
        self.assertIn("Kde najdu svoji fakturu?", topics)
        self.assertNotIn("Nezajímavá navigace", topics)
        self.assertEqual(links, ["https://www.alza.cz/contact/orders.htm"])

    def test_deduplicate_numbers_punctuation_and_accents(self):
        self.assertEqual(fingerprint("Kde je zásilka 42891902?"),
                         fingerprint("kde je zasilka 51928304!"))
        self.assertTrue(duplicate("Kde je zasilka 12345678?", ["Kde je zásilka 87654321!"]))
        self.assertFalse(duplicate("Vrácení peněz", ["Změna adresy"]))

    def test_style_boundaries(self):
        self.assertTrue(valid_style("Kde je můj balík", "short"))
        self.assertFalse(valid_style("Kde je právě můj balík", "short"))
        self.assertTrue(valid_style("kde najdu svuj doklad k objednavce", "no_diacritics"))
        self.assertFalse(valid_style("kde najdu svůj doklad k objednavce", "no_diacritics"))
        self.assertFalse(valid_style("Prosím fakturu", "formal"))

    def test_fallback_is_explicit_and_idempotent(self):
        sources = add_fallback([])
        self.assertGreaterEqual(len(sources), 25)
        self.assertEqual(sources, add_fallback(sources))
        self.assertTrue(all(s["origin"] == "curated_fallback" for s in sources))
        self.assertEqual(len(sources), len({s["id"] for s in sources}))

    def test_discovery_rejects_invented_source(self):
        llm = Mock()
        llm.ask.return_value = {"intents": [
            {"name": f"intent_{i}", "description": f"Popis {i}", "source_ids": ["nonexistent"]}
            for i in range(25)]}
        self.assertIsNone(discover(llm, [{"id": "s0000"}], 25))
        self.assertEqual(llm.ask.call_count, 3)

    def test_generation_never_exports_underfilled_intent(self):
        llm = Mock()
        llm.ask.return_value = {s: ["vrácení peněz"] * 6 for s in STYLES}
        intent = {"name": "refund", "description": "Refundace", "source_ids": ["s0000"]}
        with self.assertRaises(RuntimeError):
            generate(llm, intent, [], [], rounds=2)
        self.assertEqual(llm.ask.call_count, 2)

    def test_split_has_all_classes_and_no_overlap(self):
        for classes in (25, 26):
            rows = [{"text": f"{i}-{j}", "intent": f"intent_{i}"}
                    for i in range(classes) for j in range(30)]
            parts = split_rows(rows, 42)
            self.assertEqual(parts, split_rows(rows, 42))
            self.assertEqual(len(parts["train"]), classes * 21)
            self.assertEqual(sum(map(len, parts.values())), len(rows))
            ids = [{r["text"] for r in part} for part in parts.values()]
            self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
            for part in parts.values():
                self.assertEqual(len(Counter(r["intent"] for r in part)), classes)
            self.assertLessEqual(abs(len(parts["dev"]) - len(rows) * .15), .5)
            self.assertLessEqual(abs(len(parts["test"]) - len(rows) * .15), .5)


if __name__ == "__main__":
    unittest.main()
