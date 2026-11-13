"""Langkah 4 Panduan C: bahan analisis dan grafik untuk laporan dan slide.

Keluaran:
  results/ringkasan_cv_c.csv        rata-rata dan simpangan baku per model (ringkasan cepat)
  results/feature_importance_top5.csv  5 fitur teratas DecisionTree dan RandomForest
  slides/grafik_f1_cv.png           F1 kelas tidak lulus per model (slide 8)
  slides/grafik_pohon.png           pohon keputusan dangkal (slide 7)
Dijalankan SETELAH run_cv.py dan train_final.py. Tidak menyentuh test.csv.
"""
import joblib
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.tree import plot_tree  # noqa: E402

from common import ROOT  # noqa: E402

RES, SLIDES = ROOT / 'results', ROOT / 'slides'
SLIDES.mkdir(exist_ok=True)

# warna: palet referensi (biru = tanpa G1/G2, oranye = dengan G1/G2); teks pakai tinta netral
INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e3e2de', '#fcfcfb'
WARNA = {'tanpa_G': '#2a78d6', 'dengan_G': '#eb6834'}
LABEL = {'tanpa_G': 'Tanpa G1/G2', 'dengan_G': 'Dengan G1/G2'}
URUTAN = ['Baseline', 'NaiveBayes', 'kNN', 'DecisionTree', 'SVM', 'RandomForest']

# ---------- 1. ringkasan validasi silang ----------
cv = pd.read_csv(RES / 'cv_results.csv')
metrik = ['f1', 'recall', 'precision', 'roc_auc']
ring = cv.groupby(['skenario', 'model'])[metrik].agg(['mean', 'std']).round(3)
ring.to_csv(RES / 'ringkasan_cv_c.csv')
print(ring.to_string())

# ---------- 2. grafik F1 per model ----------
model_ada = [m for m in URUTAN if m in cv['model'].unique()]
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200, facecolor=SURFACE)
ax.set_facecolor(SURFACE)
y0 = np.arange(len(model_ada))
tinggi = 0.34
for k, sk in enumerate(['tanpa_G', 'dengan_G']):
    d = cv[cv.skenario == sk].groupby('model')['f1'].agg(['mean', 'std']).reindex(model_ada)
    pos = y0 + (k - 0.5) * (tinggi + 0.04)
    ax.barh(pos, d['mean'], height=tinggi, xerr=d['std'], color=WARNA[sk], label=LABEL[sk],
            error_kw=dict(ecolor=INK2, elinewidth=1, capsize=3))
    for p, v, s in zip(pos, d['mean'], d['std'].fillna(0)):
        ax.text(v + s + 0.02, p, f'{v:.2f}', va='center', fontsize=8, color=INK2)
ax.set_yticks(y0, model_ada, color=INK)
ax.invert_yaxis()
ax.set_xlim(0, 1.05)
ax.set_xlabel('F1 kelas "tidak lulus" (rata-rata fold luar, garis = simpangan baku)', color=INK2)
ax.set_title('Perbandingan model pada validasi silang bersarang', color=INK, loc='left')
ax.xaxis.grid(True, color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
for s in ('top', 'right', 'left'):
    ax.spines[s].set_visible(False)
ax.spines['bottom'].set_color(GRID)
ax.tick_params(colors=INK2, length=0)
ax.legend(frameon=False, loc='lower right', labelcolor=INK)
fig.tight_layout()
fig.savefig(SLIDES / 'grafik_f1_cv.png')
plt.close(fig)


# ---------- 3. pentingnya fitur ----------
def nama_fitur(model):
    pre = model.best_estimator_.named_steps['pre'] if hasattr(model, 'best_estimator_') \
        else model.named_steps['pre']
    return [n.split('__', 1)[1] for n in pre.get_feature_names_out()]


def estimator(model):
    return model.best_estimator_ if hasattr(model, 'best_estimator_') else model


baris = []
for sk in ['tanpa_G', 'dengan_G']:
    for nama in ['DecisionTree', 'RandomForest']:
        m = joblib.load(RES / 'models' / f'{sk}_{nama}.joblib')
        imp = estimator(m).named_steps['clf'].feature_importances_
        s = pd.Series(imp, index=nama_fitur(m)).sort_values(ascending=False).head(5)
        for peringkat, (f, v) in enumerate(s.items(), 1):
            baris.append({'skenario': sk, 'model': nama, 'peringkat': peringkat,
                          'fitur': f, 'importance': round(float(v), 3)})
fi = pd.DataFrame(baris)
fi.to_csv(RES / 'feature_importance_top5.csv', index=False)
print(fi.to_string(index=False))

# ---------- 4. pohon keputusan dangkal ----------
m = joblib.load(RES / 'models' / 'tanpa_G_DecisionTree.joblib')
fig, ax = plt.subplots(figsize=(14, 7), dpi=150)
plot_tree(estimator(m).named_steps['clf'], feature_names=nama_fitur(m),
          class_names=['lulus', 'tidak lulus'], max_depth=3, filled=True,
          rounded=True, fontsize=8, ax=ax, impurity=False)
fig.tight_layout()
fig.savefig(SLIDES / 'grafik_pohon.png')
plt.close(fig)
print('Grafik tersimpan di', SLIDES)
