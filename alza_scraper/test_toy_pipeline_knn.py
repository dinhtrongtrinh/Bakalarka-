from collections import Counter
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np
import toy_pipeline_knn as toy


class ToyPipelineTests(unittest.TestCase):
    def test_existing_dataset_stratified_split(self):
        path = toy.resolve_dataset()
        texts, labels = toy.load_sample(path)
        train, val, train_labels, val_labels = toy.split_sample(texts, labels)
        self.assertEqual(len(train), 40)
        self.assertEqual(len(val), 10)
        self.assertEqual(set(Counter(train_labels).values()), {8})
        self.assertEqual(set(Counter(val_labels).values()), {2})
        self.assertFalse(set(train) & set(val))
        self.assertEqual(toy.split_sample(texts, labels), [train, val, train_labels, val_labels])

    def test_friendly_missing_file_and_invalid_json(self):
        with self.assertRaisesRegex(ValueError, 'nebyl nalezen'):
            toy.resolve_dataset('/definitely/not/a/dataset.jsonl')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.jsonl'
            path.write_text('{broken}\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'řádku 1'):
                toy.load_sample(path)
            path.write_text(json.dumps({'text': 'A', 'intent_id': 'first'}) + '\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'jen 1 intentů'):
                toy.load_sample(path)

    def test_embedding_fallback_and_normalization_options(self):
        encoder = Mock()
        encoder.encode.side_effect = [np.array([[1., 0.], [0., 1.]]), np.array([[1., 0.]])]
        def factory(name, **options):
            self.assertTrue(options['local_files_only'])
            if name == toy.PRIMARY_MODEL:
                raise OSError('unavailable')
            return encoder
        train, val, used = toy.embed_texts(['a', 'b'], ['c'], offline=True, model_factory=factory)
        self.assertEqual(used, toy.FALLBACK_MODEL)
        self.assertEqual(train.shape, (2, 2))
        for call in encoder.encode.call_args_list:
            self.assertTrue(call.kwargs['normalize_embeddings'])
            self.assertEqual(call.kwargs['prompt'], '')

    def test_cosine_knn_probabilities_and_full_text_table(self):
        train = np.array([[1., 0.]] * 8 + [[0., 1.]] * 8)
        predicted, confidence = toy.classify(train, ['invoice'] * 8 + ['delay'] * 8,
                                             np.array([[1., 0.], [0., 1.]]))
        self.assertEqual(list(predicted), ['invoice', 'delay'])
        self.assertEqual(confidence, [1., 1.])
        text = 'Dlouhá zákaznická zpráva ' * 10 + 'NEZKRACENY_KONEC'
        output = io.StringIO()
        with redirect_stdout(output):
            toy.print_results([text, 'Čekám na balík'], ['invoice', 'delay'], predicted, confidence)
        self.assertIn('100.00%', output.getvalue())
        self.assertIn('SPRAVNE', output.getvalue())
        self.assertIn('NEZKRACENY_KONEC', output.getvalue())


if __name__ == '__main__':
    unittest.main()
