"""Langkah 2 Panduan C: validasi silang bersarang -> results/cv_results.csv

Lingkaran dalam (GridSearchCV, 3-fold)  : menyetel hiperparameter.
Lingkaran luar (5-fold x 3 ulangan)     : menilai, pada data yang TIDAK dipakai menyetel.

Pakai (dari folder utama repo):
  python scripts/run_cv.py            # penuh (lama; disarankan di Colab / laptop kuat)
  python scripts/run_cv.py --cepat    # uji coba kecil (3 fold x 1 ulangan)
  python scripts/run_cv.py --lanjut   # lanjutkan bila proses sebelumnya terputus
"""
import argparse
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold,
                                     StratifiedKFold, cross_validate)

from common import ROOT, buat_pipeline, muat_data, skenario
from src.models import MODELS

ap = argparse.ArgumentParser()
ap.add_argument('--cepat', action='store_true')
ap.add_argument('--lanjut', action='store_true')
ap.add_argument('--train', default=None, help='jalur train.csv (default: data/processed/train.csv)')
ap.add_argument('--kontrak', default=None, help='jalur columns.json')
ap.add_argument('--out', default=str(ROOT / 'results' / 'cv_results.csv'))
args = ap.parse_args()

kw = {}
if args.train:
    kw['train_csv'] = args.train
if args.kontrak:
    kw['kontrak_json'] = args.kontrak
train, kontrak = muat_data(**kw)
y = train[kontrak['target']]

n_splits, n_repeats = (3, 1) if args.cepat else (5, 3)
outer = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Pembagian fold luar dibuat SEKALI lalu dipakai ulang untuk SEMUA model dan skenario,
# jadi D bisa membandingkan model berpasangan fold demi fold.
fold_luar = list(outer.split(train, y))

# zero_division=0: Baseline tidak pernah menebak "tidak lulus", jadi precision/F1-nya
# 0 (bukan error/peringatan). Kelas positif = 1 = tidak lulus.
metrik = {
    'f1': make_scorer(f1_score, zero_division=0),
    'recall': make_scorer(recall_score, zero_division=0),
    'precision': make_scorer(precision_score, zero_division=0),
    'roc_auc': 'roc_auc',
}

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
baris, selesai = [], set()
if args.lanjut and out.exists():
    lama = pd.read_csv(out)
    baris = lama.to_dict('records')
    selesai = set(zip(lama['skenario'], lama['model']))
    print(f'Melanjutkan: {len(selesai)} kombinasi sudah selesai')

mulai = time.time()
for nama_skenario, fitur in skenario(kontrak).items():
    X = train[fitur]
    for nama, (clf, grid) in MODELS.items():
        if (nama_skenario, nama) in selesai:
            continue
        t0 = time.time()
        pipe = buat_pipeline(fitur, clf, kontrak)
        model = (GridSearchCV(pipe, grid, cv=inner, scoring='f1', n_jobs=1)
                 if grid else pipe)
        hasil = cross_validate(model, X, y, cv=fold_luar, scoring=metrik, n_jobs=-1)
        for i in range(len(fold_luar)):
            baris.append({'skenario': nama_skenario, 'model': nama, 'fold': i,
                          **{m: hasil['test_' + m][i] for m in metrik}})
        # simpan tiap selesai satu model, supaya kalau terputus tidak mulai dari nol
        pd.DataFrame(baris).to_csv(out, index=False)
        f1_rata = pd.Series(hasil['test_f1']).mean()
        print(f'[{nama_skenario}] {nama:<13} F1 rata-rata={f1_rata:.3f} '
              f'({time.time() - t0:.0f} dtk)', flush=True)

print(f'Selesai dalam {(time.time() - mulai) / 60:.1f} menit -> {out}')
