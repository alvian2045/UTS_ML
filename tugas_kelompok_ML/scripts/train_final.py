"""Langkah 3 Panduan C: latih model final pada SELURUH data latih dan simpan.

Hasil: results/models/{skenario}_{model}.joblib dan results/best_params.json
"""
import argparse
import json

import joblib
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from common import ROOT, buat_pipeline, muat_data, skenario
from src.models import MODELS

ap = argparse.ArgumentParser()
ap.add_argument('--train', default=None)
ap.add_argument('--kontrak', default=None)
ap.add_argument('--outdir', default=str(ROOT / 'results'))
args = ap.parse_args()

kw = {}
if args.train:
    kw['train_csv'] = args.train
if args.kontrak:
    kw['kontrak_json'] = args.kontrak
train, kontrak = muat_data(**kw)
y = train[kontrak['target']]

from pathlib import Path  # noqa: E402
outdir = Path(args.outdir)
(outdir / 'models').mkdir(parents=True, exist_ok=True)

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
param_terbaik, skor_cv = {}, {}

for nama_skenario, fitur in skenario(kontrak).items():
    X = train[fitur]
    for nama, (clf, grid) in MODELS.items():
        pipe = buat_pipeline(fitur, clf, kontrak)
        model = GridSearchCV(pipe, grid, cv=cv5, scoring='f1', n_jobs=-1) if grid else pipe
        model.fit(X, y)
        kunci = f'{nama_skenario}_{nama}'
        joblib.dump(model, outdir / 'models' / f'{kunci}.joblib')
        param_terbaik[kunci] = model.best_params_ if grid else {}
        skor_cv[kunci] = float(model.best_score_) if grid else None
        print(f'{kunci:<26} {param_terbaik[kunci]}', flush=True)

json.dump(param_terbaik, open(outdir / 'best_params.json', 'w'), indent=2, default=str)
# F1 dalam (penyetelan) hanya untuk info; BUKAN estimasi performa (itu di cv_results.csv)
json.dump(skor_cv, open(outdir / 'best_f1_penyetelan.json', 'w'), indent=2)
print('Selesai.')
