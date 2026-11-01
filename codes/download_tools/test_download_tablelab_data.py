"""下载器离线测试。全部数据是极小合成夹具，不代表真实数据已下载。"""
import contextlib
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError
import zipfile

import download_tablelab_data as d


ADULT_ROW = '39, State-gov, 77516, Bachelors, 13, Never-married, Adm-clerical, Not-in-family, White, Male, 2174, 0, 40, United-States, <=50K\n'


def payloads(key):
    if key == 'adult':
        return {
            'adult.data': ADULT_ROW.encode(),
            'adult.test': ('|1x3 Cross validator\n' + ADULT_ROW.replace('<=50K', '>50K.')).encode(),
            'adult.names': b'SYNTHETIC TEST FIXTURE; NOT AN ORIGINAL DATASET.\n',
        }
    result = {}
    for name in d.DATASETS[key].files:
        if name.endswith('.csv'):
            header = sorted(d.REQUIRED_COLUMNS[name])
            stream = io.StringIO(newline='')
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerow(['1'] * len(header))
            result[name] = stream.getvalue().encode()
        else:
            result[name] = b'SYNTHETIC TEST FIXTURE; NOT AN ORIGINAL DATASET.\n'
    return result


def make_zip(path, key, changes=None):
    data = payloads(key)
    if changes:
        data.update(changes)
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, body in data.items():
            archive.writestr(name, body)


def fake_adult(path):
    make_zip(path, 'adult')


def fake_kaggle(dataset, output):
    output.mkdir(parents=True)
    for name, body in payloads(dataset.key).items():
        (output / name).write_bytes(body)
    return output


class DownloaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = {'format_version': 1, 'script_version': d.VERSION, 'datasets': {}}

    def prepare_fixture(self, key='adult'):
        with patch.object(d, 'download_adult', side_effect=fake_adult), \
             patch.object(d, 'download_kaggle', side_effect=fake_kaggle), \
             patch.object(d.importlib.metadata, 'version', return_value='1.0.2'), \
             contextlib.redirect_stdout(io.StringIO()):
            d.prepare_dataset(self.root, d.DATASETS[key], self.manifest)

    def test_all_four_archive_contracts(self):
        for key, spec in d.DATASETS.items():
            with self.subTest(dataset=key):
                path = self.root / spec.archive_name
                make_zip(path, key)
                info = d.inspect_archive(path, spec)
                self.assertEqual(set(info['files']), set(spec.files))
                self.assertEqual(info['archive_sha256'], hashlib.sha256(path.read_bytes()).hexdigest())

    def test_adult_test_comment_and_label_period(self):
        meta = d.inspect_payload('adult.test', payloads('adult')['adult.test'])
        self.assertEqual(meta['rows'], 1)
        self.assertEqual(meta['column_count'], 15)

    def test_adult_bad_label(self):
        with self.assertRaisesRegex(d.DownloadError, '15 列'):
            d.inspect_payload('adult.data', ADULT_ROW.replace('<=50K', '123456').encode())

    def test_adult_empty_after_comments(self):
        with self.assertRaisesRegex(d.DownloadError, '没有数据行'):
            d.inspect_payload('adult.test', b'| only a comment\n\n')

    def test_html_not_zip(self):
        path = self.root / 'html.zip'
        path.write_text('<html>Login required</html>')
        with self.assertRaisesRegex(d.DownloadError, '有效 ZIP'):
            d.inspect_archive(path, d.DATASETS['adult'])

    def test_missing_member(self):
        path = self.root / 'missing.zip'
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr('adult.data', ADULT_ROW)
        with self.assertRaisesRegex(d.DownloadError, 'adult.test'):
            d.inspect_archive(path, d.DATASETS['adult'])

    def test_duplicate_member_basename(self):
        path = self.root / 'duplicate.zip'
        make_zip(path, 'adult', {'nested/adult.data': ADULT_ROW.encode()})
        with self.assertRaisesRegex(d.DownloadError, '2 个'):
            d.inspect_archive(path, d.DATASETS['adult'])

    def test_archive_traversal_rejected(self):
        path = self.root / 'unsafe.zip'
        make_zip(path, 'adult', {'../outside': b'bad'})
        with self.assertRaisesRegex(d.DownloadError, '不安全'):
            d.inspect_archive(path, d.DATASETS['adult'])

    def test_archive_size_limit(self):
        path = self.root / 'size.zip'
        make_zip(path, 'adult')
        with patch.object(d, 'MAX_BYTES', 10), self.assertRaisesRegex(d.DownloadError, '100 MB'):
            d.inspect_archive(path, d.DATASETS['adult'])

    def test_duplicate_columns(self):
        with self.assertRaisesRegex(d.DownloadError, '重复字段'):
            d.inspect_payload('train.csv', b'Id,Id\n1,2\n')

    def test_missing_required_column(self):
        with self.assertRaisesRegex(d.DownloadError, '缺少所需字段'):
            d.inspect_payload('StudentsPerformance.csv', b'gender,score\nfemale,90\n')

    def test_uneven_csv_row(self):
        body = payloads('house')['train.csv'] + b'1,2\n'
        with self.assertRaisesRegex(d.DownloadError, '列数'):
            d.inspect_payload('train.csv', body)

    def test_empty_data_rows(self):
        body = ','.join(sorted(d.REQUIRED_COLUMNS['train.csv'])) + '\n'
        with self.assertRaisesRegex(d.DownloadError, '没有数据行'):
            d.inspect_payload('train.csv', body.encode())

    def test_house_test_cannot_contain_target(self):
        cols = sorted(d.REQUIRED_COLUMNS['test.csv'] | {'SalePrice'})
        body = ','.join(cols) + '\n' + ','.join('1' for _ in cols) + '\n'
        with self.assertRaisesRegex(d.DownloadError, '意外含 SalePrice'):
            d.inspect_payload('test.csv', body.encode())

    def test_packaging_preserves_file_bytes(self):
        source = fake_kaggle(d.DATASETS['house'], self.root / 'source')
        dest = self.root / 'house.zip'
        d.package_kaggle_files(source, dest, d.DATASETS['house'])
        with zipfile.ZipFile(dest) as archive:
            for name, body in payloads('house').items():
                self.assertEqual(archive.read(name), body)

    def test_packaging_excludes_credentials(self):
        source = fake_kaggle(d.DATASETS['nba'], self.root / 'source')
        (source / 'kaggle.json').write_text('{"key":"FAKE_SECRET"}')
        (source / '.env').write_text('FAKE_SECRET=anything')
        dest = self.root / 'nba.zip'
        d.package_kaggle_files(source, dest, d.DATASETS['nba'])
        with zipfile.ZipFile(dest) as archive:
            self.assertEqual(archive.namelist(), ['all_seasons.csv'])

    def test_packaging_rejects_symbolic_link(self):
        source = self.root / 'source'
        source.mkdir()
        external = self.root / 'external.csv'
        external.write_bytes(payloads('nba')['all_seasons.csv'])
        (source / 'all_seasons.csv').symlink_to(external)
        with self.assertRaisesRegex(d.DownloadError, '目录之外'):
            d.package_kaggle_files(source, self.root / 'out.zip', d.DATASETS['nba'])

    def test_manifest_records_real_hash_not_credential(self):
        with patch.dict(os.environ, {'KAGGLE_API_TOKEN': 'FAKE_SECRET'}):
            self.prepare_fixture()
        content = (self.root / d.MANIFEST_NAME).read_text()
        self.assertNotIn('FAKE_SECRET', content)
        record = json.loads(content)['datasets']['adult']
        self.assertEqual(record['packaging'], 'original_source_zip')
        self.assertEqual(record['archive_sha256'], d.sha256_file(self.root / 'adult.zip'))

    def test_cached_archive_reused(self):
        self.prepare_fixture()
        with patch.object(d, 'download_adult', side_effect=AssertionError('network must not be called')), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(d.main(['--output', str(self.root), '--only', 'adult']), 0)

    def test_untracked_archive_not_claimed_as_downloaded(self):
        make_zip(self.root / 'adult.zip', 'adult')
        with self.assertRaisesRegex(d.DownloadError, '来源记录'):
            d.cached_record(self.root, d.DATASETS['adult'], self.manifest)

    def test_modified_archive_rejected(self):
        self.prepare_fixture()
        with zipfile.ZipFile(self.root / 'adult.zip', 'a') as archive:
            archive.writestr('added.txt', 'changed')
        with self.assertRaisesRegex(d.DownloadError, 'SHA-256'):
            d.cached_record(self.root, d.DATASETS['adult'], self.manifest)

    def test_wrong_source_record_rejected(self):
        self.prepare_fixture()
        self.manifest['datasets']['adult']['source_page'] = 'https://example.invalid/'
        with self.assertRaisesRegex(d.DownloadError, '来源记录'):
            d.cached_record(self.root, d.DATASETS['adult'], self.manifest)

    def test_failure_keeps_previous_valid_archive(self):
        self.prepare_fixture()
        old = (self.root / 'adult.zip').read_bytes()
        with patch.object(d, 'download_adult', side_effect=d.DownloadError('simulated network failure')):
            with self.assertRaisesRegex(d.DownloadError, 'simulated'):
                d.prepare_dataset(self.root, d.DATASETS['adult'], self.manifest)
        self.assertEqual((self.root / 'adult.zip').read_bytes(), old)
        self.assertFalse(list(self.root.glob('.adult-*')))

    def test_bundle_has_only_four_archives_and_manifest(self):
        for key in d.DATASETS:
            self.prepare_fixture(key)
        (self.root / 'kaggle.json').write_text('FAKE_SECRET')
        (self.root / '.env').write_text('FAKE_SECRET')
        bundle = d.make_bundle(self.root, self.manifest)
        with zipfile.ZipFile(bundle) as archive:
            expected = {s.archive_name for s in d.DATASETS.values()} | {d.MANIFEST_NAME}
            self.assertEqual(set(archive.namelist()), expected)
            self.assertNotIn(b'FAKE_SECRET', archive.read(d.MANIFEST_NAME))
            self.assertEqual(set(json.loads(archive.read(d.MANIFEST_NAME))['datasets']), set(d.DATASETS))

    def test_incomplete_bundle_rejected(self):
        self.prepare_fixture()
        with self.assertRaisesRegex(d.DownloadError, '缺少 nba.zip'):
            d.make_bundle(self.root, self.manifest)
        self.assertFalse((self.root / d.BUNDLE_NAME).exists())

    def test_force_replaces_and_records_again(self):
        self.prepare_fixture()
        with patch.object(d, 'download_adult', side_effect=fake_adult) as mock, \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(d.main(['--output', str(self.root), '--only', 'adult', '--force']), 0)
            mock.assert_called_once()

    def test_full_main_with_mock_downloads(self):
        with patch.object(d, 'download_adult', side_effect=fake_adult), \
             patch.object(d, 'download_kaggle', side_effect=fake_kaggle), \
             patch.object(d.importlib.metadata, 'version', return_value='1.0.2'), \
             patch.object(d.importlib.util, 'find_spec', return_value=object()), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(d.main(['--output', str(self.root)]), 0)
        self.assertTrue((self.root / d.BUNDLE_NAME).is_file())

    def test_uci_dns_failure_is_explicit(self):
        with patch.object(d, 'urlopen', side_effect=URLError('mock DNS failure')):
            with self.assertRaisesRegex(d.DownloadError, 'DNS'):
                d.download_adult(self.root / 'out.zip')

    def test_uci_http_failure_is_explicit(self):
        error = HTTPError(d.ADULT_URL, 503, 'test', {}, None)
        with patch.object(d, 'urlopen', side_effect=error):
            with self.assertRaisesRegex(d.DownloadError, 'HTTP 503'):
                d.download_adult(self.root / 'out.zip')

    def test_token_not_written_to_disk(self):
        with patch.object(d.sys.stdin, 'isatty', return_value=True), \
             patch.object(d.getpass, 'getpass', return_value='FAKE_TEST_TOKEN'), \
             patch.dict(os.environ, {}, clear=True), contextlib.redirect_stdout(io.StringIO()):
            d.prompt_token()
            self.assertEqual(os.environ['KAGGLE_API_TOKEN'], 'FAKE_TEST_TOKEN')
        self.assertEqual(list(self.root.iterdir()), [])

    def test_nonterminal_token_prompt_rejected(self):
        with patch.object(d.sys.stdin, 'isatty', return_value=False):
            with self.assertRaisesRegex(d.DownloadError, '终端'):
                d.prompt_token()

    def test_invalid_token_rejected(self):
        with patch.object(d.sys.stdin, 'isatty', return_value=True), \
             patch.object(d.getpass, 'getpass', return_value='bad token'), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(d.DownloadError, '空白'):
                d.prompt_token()


if __name__ == '__main__':
    unittest.main(verbosity=2)
