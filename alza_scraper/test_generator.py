import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import config
import generate_dataset as g


class GeneratorTests(unittest.TestCase):
    def test_claude_configuration_client_and_output(self):
        settings = copy.deepcopy(config.PROVIDERS)
        self.assertNotIn('custom', settings)
        self.assertFalse(settings['claude']['json_mode'])
        settings['claude']['api_key'] = 'test-anthropic-key'
        settings['claude']['default_headers'] = {'anthropic-workspace-id': 'test-workspace'}
        leaf = {'id': 'invoice', 'main_id': 'orders'}
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(config, 'ACTIVE_PROVIDER', 'claude'), \
                patch.object(config, 'PROVIDERS', settings), \
                patch.object(config, 'OUTPUT_DIR', Path(directory)), \
                patch.object(g, 'OpenAI') as factory, \
                patch.object(g, 'load_intents', return_value=[leaf]), \
                patch.object(g, 'generate_intent', return_value=['Testovací zpráva']):
            g.main()
            options = factory.call_args.kwargs
            self.assertEqual(options['base_url'], 'https://api.anthropic.com/v1/')
            self.assertEqual(options['api_key'], 'test-anthropic-key')
            self.assertEqual(options['default_headers']['anthropic-workspace-id'], 'test-workspace')
            path = g.output_path('claude', settings['claude']['model'])
            row = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(row['provider'], 'claude')
            self.assertEqual(row['model'], settings['claude']['model'])

    def test_gemini_model_catalog_and_missing_model(self):
        client = Mock()
        client.models.list.return_value = [SimpleNamespace(id='models/gemini-3.8-flash'),
                                           SimpleNamespace(id='gemini-embedding-001')]
        g.check_gemini_model(client, 'gemini-3.8-flash')
        with self.assertRaisesRegex(ValueError, 'Dostupné modely: gemini-3.8-flash'):
            g.check_gemini_model(client, 'gemini-2.5-flash')
        client.chat.completions.create.assert_not_called()

    def test_gemini_preflight_failure_preserves_existing_files(self):
        settings = copy.deepcopy(config.PROVIDERS)
        settings['gemini']['api_key'] = 'fake-key'
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(config, 'ACTIVE_PROVIDER', 'gemini'), \
                patch.object(config, 'PROVIDERS', settings), \
                patch.object(config, 'OUTPUT_DIR', Path(directory)), \
                patch.object(config, 'OVERWRITE', True), patch.object(g, 'OpenAI') as factory:
            partial = g.output_path('gemini', settings['gemini']['model']).with_suffix('.partial.jsonl')
            partial.write_text('previous-results', encoding='utf-8')
            client = factory.return_value.__enter__.return_value
            client.models.list.return_value = [SimpleNamespace(id='another-model')]
            with self.assertRaisesRegex(ValueError, 'není v katalogu'):
                g.main()
            self.assertEqual(partial.read_text(encoding='utf-8'), 'previous-results')
            client.chat.completions.create.assert_not_called()

    def test_api_error_message_redacts_key(self):
        settings = copy.deepcopy(config.PROVIDERS)
        settings['gemini']['api_key'] = 'fake-secret-key'
        error = SimpleNamespace(body={'error': {'message': 'No model for fake-secret-key or AIzaABC-xyz'}})
        with patch.object(config, 'PROVIDERS', settings):
            message = g.api_error_detail(error)
        self.assertNotIn('fake-secret-key', message)
        self.assertNotIn('AIzaABC-xyz', message)
        self.assertIn('No model', message)

    def test_real_tree_and_nested_leaves(self):
        leaves = g.load_intents(config.INTENTS_FILE)
        self.assertTrue(leaves)
        self.assertEqual(len(leaves), len({x['id'] for x in leaves}))
        tree = {'main': {'title': 'Hlavní', 'sub_intents': [
            {'title': 'Vnořená', 'sub_intents': [
                {'id': 'leaf', 'title': 'Podtéma', 'description': 'Popis'}]}]}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'tree.py'
            path.write_text('ALZA_INTENTS = ' + repr(tree), encoding='utf-8')
            result = g.load_intents(path)
        self.assertEqual([x['id'] for x in result], ['leaf'])
        self.assertEqual(result[0]['path'], ['Hlavní', 'Vnořená', 'Podtéma'])

    def test_json_fences_and_bad_shapes(self):
        data = {s: [] for s in g.STYLES}
        self.assertEqual(g.parse_response('```json\n' + json.dumps(data) + '\n```'), data)
        for invalid in ('[]', '{}', 'not JSON', '{"formal": [1]}'):
            with self.assertRaises(ValueError):
                g.parse_response(invalid)

    def test_counts_filenames_and_normalization(self):
        for count in (10, 11, 19, 30):
            quotas = g.style_targets(count)
            self.assertEqual(sum(quotas.values()), count)
            self.assertLessEqual(max(quotas.values()) - min(quotas.values()), 1)
        self.assertNotEqual(g.output_path('custom', 'a/b'), g.output_path('custom', 'a_b'))
        self.assertEqual(g.output_path('gemini', 'gemini-2.5-flash').name,
                         'dataset_gemini_gemini-2.5-flash.jsonl')
        self.assertEqual(g.fingerprint('Zásilka 123!'), g.fingerprint('zasilka 456?'))

    def test_provider_validation(self):
        for provider in ('gemini', 'groq', 'openai', 'ollama', 'claude'):
            settings = copy.deepcopy(config.PROVIDERS)
            settings[provider]['api_key'] = ''
            with patch.object(config, 'ACTIVE_PROVIDER', provider), patch.object(config, 'PROVIDERS', settings):
                if provider == 'ollama':
                    self.assertEqual(g.provider_settings()[0], provider)
                else:
                    with self.assertRaisesRegex(ValueError, 'API klíč'):
                        g.provider_settings()

    def test_malformed_response_retries_then_generates_ten(self):
        batch = {
            'formal': ['Dobrý den, potřebuji kopii své faktury.', 'Prosím o zaslání dokladu k objednávce.'],
            'colloquial': ['Hele, někam se mi ztratil účet.', 'Čau, hoďte mi ten doklad znova.'],
            'no_diacritics': ['nemuzu nikde najit fakturu za notebook', 'poslete mi doklad zase na mail'],
            'conversational': ['Prosím fakturu', 'Kopie daňového dokladu'],
            'urgent': ['Okamžitě mi pošlete tu chybějící fakturu!', 'Účetní na doklad čeká už týden!'],
        }
        batch = {style: [text + " Potrebuji tento doklad predat ucetni jeste dnes, protoze uzavira firemni naklady za minuly mesic."
                         for text in texts] for style, texts in batch.items()}
        def response(content):
            return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop', message=SimpleNamespace(content=content, refusal=None))])
        client = Mock()
        client.chat.completions.create.side_effect = [response('broken'), response(json.dumps(batch))]
        intent = {'id': 'invoice', 'title': 'Faktura', 'description': 'Zaslání dokladu.',
                  'main_id': 'orders', 'main_title': 'Objednávky', 'main_description': 'Nákupy', 'path': ['Objednávky', 'Faktura']}
        with patch.object(config, 'REQUEST_PAUSE', 0):
            sentences = g.generate_intent(client, {'model': 'test', 'json_mode': False, 'max_tokens': 8192}, intent, g.style_targets(10))
        self.assertEqual(len(sentences), 10)
        self.assertEqual(len(set(sentences)), 10)
        self.assertEqual(client.chat.completions.create.call_count, 2)
        self.assertEqual(client.chat.completions.create.call_args.kwargs['max_tokens'], 8192)
        self.assertNotIn('response_format', client.chat.completions.create.call_args.kwargs)
        self.assertIn('Objednávky', client.chat.completions.create.call_args.kwargs['messages'][1]['content'])

    def test_long_message_limits_and_ascii(self):
        with patch.object(config, 'MIN_WORDS_PER_MESSAGE', 15), patch.object(config, 'MAX_WORDS_PER_MESSAGE', 60):
            for style in g.STYLES:
                self.assertFalse(g.valid_sentence('slovo ' * 14, style))
                self.assertTrue(g.valid_sentence('slovo ' * 15, style))
                self.assertTrue(g.valid_sentence('slovo ' * 60, style))
                self.assertFalse(g.valid_sentence('slovo ' * 61, style))
            self.assertFalse(g.valid_sentence('zpráva ' * 20, 'no_diacritics'))
            self.assertIn('ASCII', g.rejection_reason('zpráva ' * 20, 'no_diacritics'))
        self.assertNotIn('short', g.STYLES)

    def test_underfilled_response_cannot_be_exported(self):
        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(
            finish_reason='stop', message=SimpleNamespace(content=json.dumps({s: [] for s in g.STYLES}), refusal=None))])
        intent = {k: 'test' for k in ('id', 'title', 'description', 'main_id', 'main_title', 'main_description', 'path')}
        with patch.object(config, 'REQUEST_PAUSE', 0), patch.object(config, 'MAX_GENERATION_ROUNDS', 2):
            with self.assertRaisesRegex(ValueError, 'chybí zprávy'):
                g.generate_intent(client, {'model': 'test', 'json_mode': False}, intent, g.style_targets(10))


if __name__ == '__main__':
    unittest.main()
