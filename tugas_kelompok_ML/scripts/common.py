"""Fungsi bersama untuk skrip peran C. Hanya membaca train.csv, TIDAK pernah test.csv."""
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import build_preprocessor  # noqa: E402  (buatan B)

TRAIN_CSV = ROOT / 'data' / 'processed' / 'train.csv'
KONTRAK_JSON = ROOT / 'data' / 'processed' / 'columns.json'


def _perbaiki_jenis_kolom(train, kontrak):
    """Tentukan kolom numerik vs kategorik langsung dari data.

    Alasan: di pandas 3, kolom teks berjenis 'str', bukan 'object'. Kode A yang memakai
    dtype == 'object' lalu salah menaruh kolom teks (school, sex, Mjob, ...) di num_cols,
    sehingga imputer median error ("could not convert string to float: 'GP'").
    File columns.json milik A TIDAK diubah; perbaikan hanya berlaku di memori.
    """
    fitur = kontrak['fitur_dengan_g']
    num = [c for c in fitur if pd.api.types.is_numeric_dtype(train[c])]
    cat = [c for c in fitur if c not in num]
    if set(cat) != set(kontrak['cat_cols']):
        print(f"PERINGATAN: cat_cols di columns.json berisi {len(kontrak['cat_cols'])} kolom, "
              f"padahal data punya {len(cat)} kolom teks. Dipakai hasil deteksi dari data "
              f"(file columns.json tidak diubah). Kabari A.")
    kontrak = dict(kontrak)
    kontrak['num_cols'], kontrak['cat_cols'] = num, cat
    return kontrak


def muat_data(train_csv=TRAIN_CSV, kontrak_json=KONTRAK_JSON):
    """Muat data latih + kontrak kolom dari A, dengan pengaman anti-kebocoran."""
    train_csv = Path(train_csv)
    assert 'test' not in train_csv.name.lower(), 'Jangan memuat test.csv di tahap ini!'
    train = pd.read_csv(train_csv)
    kontrak = json.load(open(kontrak_json))
    kontrak = _perbaiki_jenis_kolom(train, kontrak)
    target = kontrak['target']
    for nama in ('fitur_dengan_g', 'fitur_tanpa_g'):
        assert 'G3' not in kontrak[nama] and target not in kontrak[nama], \
            f'G3/target bocor ke {nama}'
    return train, kontrak


def skenario(kontrak):
    return {'dengan_G': kontrak['fitur_dengan_g'],
            'tanpa_G': kontrak['fitur_tanpa_g']}


def buat_pipeline(fitur, clf, kontrak):
    pre = build_preprocessor(fitur, kontrak['num_cols'], kontrak['cat_cols'])
    return Pipeline([('pre', pre), ('clf', clf)])