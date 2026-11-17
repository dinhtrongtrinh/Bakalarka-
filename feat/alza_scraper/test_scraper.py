import ast
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from bs4 import BeautifulSoup
import scrape_intents as s


class ScraperTests(unittest.TestCase):
    def test_urls_keep_topic_parameters(self):
        self.assertEqual(s.absolute_url('/contact/services.htm?qid=691&qgid=69&utm_source=x#faq', s.BASE),
                         s.BASE + '/contact/services.htm?qgid=69&qid=691#faq')
        for bad in ('https://evil.example/contact', 'javascript:alert(1)', 'https://www.alza.cz:bad/contact'):
            self.assertIsNone(s.absolute_url(bad, s.BASE))

    def test_parser_preserves_title_and_separates_account(self):
        html = BeautifulSoup('''<main><span>Osobní údaje</span>
        <a href="/contact/others.htm?qid=377&qgid=68">Zrušení zasílání akčních nabídek</a>
        <span>Pochvala / stížnost</span><a href="https://evil.example">Nesouvisející obsah</a>
        <h2>Nejčastější dotazy</h2></main>''', 'html.parser')
        scraper = s.Scraper()
        account = scraper.extract(html, s.BASE + '/contact/others.htm', 'account')
        other = scraper.extract(html, s.BASE + '/contact/others.htm', 'other')
        self.assertEqual([x[0] for x in account], ['Osobní údaje', 'Zrušení zasílání akčních nabídek'])
        self.assertEqual([x[0] for x in other], ['Pochvala / stížnost'])
        scraper.session.close()

    def test_empty_site_produces_valid_nonempty_tree(self):
        scraper = s.Scraper()
        with patch.object(scraper, 'fetch', return_value=None):
            tree, metadata = scraper.build()
        self.assertEqual(set(tree), set(s.CATEGORIES))
        nodes = [n for category in tree.values() for n in category['children']]
        self.assertEqual(len(nodes), 18)
        self.assertEqual(len({n['id'] for n in nodes}), 18)
        for node in nodes:
            self.assertEqual(set(node), {'id', 'title', 'url', 'description'})
            self.assertRegex(node['id'], r'^[a-z][a-z0-9_]*$')
            self.assertTrue(node['url'].startswith('https://'))
            self.assertEqual(metadata['origins'][node['id']], 'static_fallback')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'intents_tree.py'
            with patch.object(s, 'OUTPUT', path):
                s.write_tree(tree, metadata)
            parsed = ast.parse(path.read_text(encoding='utf-8'))
            self.assertEqual(ast.literal_eval(parsed.body[0].value), tree)
        scraper.session.close()

    def test_live_topic_survives_partial_fallback(self):
        page = BeautifulSoup('<a href="/contact/services.htm?qgid=69&qid=691">AlzaPlus+</a>', 'html.parser')
        scraper = s.Scraper()
        with patch.object(scraper, 'fetch', return_value=(page, s.BASE + '/contact/services.htm')):
            tree, metadata = scraper.build()
        self.assertEqual(metadata['origins']['services_alzaplus'], 'scraped')
        self.assertEqual(metadata['origins']['services_warranty'], 'static_fallback')
        self.assertEqual(len(tree['services']['children']), 4)
        scraper.session.close()


if __name__ == '__main__':
    unittest.main()
