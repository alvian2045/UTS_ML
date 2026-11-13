"""Angka untuk Bab 3 (Praproses): kualitas data, jumlah fitur, dan demo anti-kebocoran.

Hanya memakai data/processed/train.csv dan src/pipeline.py (buatan B).
Jalankan dari folder utama repo:  python scripts/praproses_bab3.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.pipeline import build_preprocessor  # noqa: E402

train = pd.read_csv(ROOT / 'data' / 'processed' / 'train.csv')   # TIDAK memuat test.csv
kontrak = json.load(open(ROOT / 'data' / 'processed' / 'columns.json'))
y = train[kontrak['target']]

# jenis kolom dicek langsung dari data agar aman terhadap tipe dtype
num_cols = [c for c in kontrak['fitur_dengan_g'] if pd.api.types.is_numeric_dtype(train[c])]
cat_cols = [c for c in kontrak['fitur_dengan_g'] if c not in num_cols]
SKENARIO = {'dengan_G': kontrak['fitur_dengan_g'], 'tanpa_G': kontrak['fitur_tanpa_g']}
print(f'Kolom numerik: {len(num_cols)} | kolom kategorik: {len(cat_cols)}')

# ---------- 1. kualitas data ----------
print('\n== Kualitas data ==')
print('Ukuran:', train.shape, '| nilai hilang:', int(train.isna().sum().sum()),
      '| duplikat:', int(train.duplicated().sum()))
a = train['absences']
q1, q3 = a.quantile([0.25, 0.75])
batas = q3 + 1.5 * (q3 - q1)
print(f'absences: median={a.median():.0f}, rata-rata={a.mean():.2f}, maks={a.max()}, '
      f'skewness={a.skew():.2f}, batas IQR={batas:.0f}, jumlah di atas batas={(a > batas).sum()}')
print('Kategori jarang (< 10% data):')
for c in cat_cols:
    vc = train[c].value_counts()
    jarang = {k: int(v) for k, v in vc.items() if v / len(train) < 0.10}
    if jarang:
        print(f'  {c}: {jarang}')

# ---------- 2. jumlah fitur setelah encoding ----------
print('\n== Jumlah fitur setelah praproses ==')
for nama, fitur in SKENARIO.items():
    Xt = build_preprocessor(fitur, num_cols, cat_cols).fit_transform(train[fitur])
    print(f'{nama}: {len(fitur)} kolom asli -> {Xt.shape[1]} fitur setelah one-hot (bentuk {Xt.shape})')

# ---------- 3. demo anti-kebocoran ----------
print('\n== Demo anti-kebocoran (F1 kelas tidak lulus, 5-fold x 3 ulangan, random_state=42) ==')
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
MODEL = {
    'kNN (k=5)': lambda: KNeighborsClassifier(5),
    'SVM (RBF)': lambda: SVC(class_weight='balanced', random_state=42),
    'Decision Tree (depth 3)': lambda: DecisionTreeClassifier(max_depth=3, class_weight='balanced',
                                                              random_state=42),
}
baris = []
for sk, fitur in SKENARIO.items():
    X = train[fitur]
    # (1) SALAH: praproses di-fit pada SELURUH data, baru validasi silang
    X_bocor = build_preprocessor(fitur, num_cols, cat_cols).fit_transform(X)
    for nama, buat in MODEL.items():
        s_bocor = cross_val_score(buat(), X_bocor, y, cv=cv, scoring='f1').mean()
        # (2) BENAR: praproses di dalam pipeline, di-fit hanya pada data latih tiap fold
        pipa = Pipeline([('pre', build_preprocessor(fitur, num_cols, cat_cols)), ('clf', buat())])
        s_benar = cross_val_score(pipa, X, y, cv=cv, scoring='f1').mean()
        baris.append((sk, nama, round(s_bocor, 4), round(s_benar, 4), round(s_bocor - s_benar, 4)))
tabel = pd.DataFrame(baris, columns=['skenario', 'model', 'F1_di_luar_pipeline', 'F1_di_dalam_pipeline', 'selisih'])
print(tabel.to_string(index=False))
print('\nCatatan: selisih pada data kecil bisa kecil atau bahkan negatif; yang dinilai adalah prinsipnya.')
