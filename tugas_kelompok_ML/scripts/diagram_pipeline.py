"""Diagram alur pipeline untuk Bab 3 -> report/gambar/gambar_3_1_pipeline.png"""
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / 'report' / 'gambar'
OUT.mkdir(parents=True, exist_ok=True)
INK, INK2, SURFACE = '#0b0b0b', '#52514e', '#fcfcfb'
BIRU, ORANYE, ABU = '#cfe0f7', '#fbd9cb', '#ececea'

fig, ax = plt.subplots(figsize=(10, 4.2), dpi=200, facecolor=SURFACE)
ax.set_xlim(0, 100)
ax.set_ylim(0, 42)
ax.axis('off')


def kotak(x, y, w, h, teks, warna):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4,rounding_size=1.2',
                                fc=warna, ec=INK2, lw=0.9))
    ax.text(x + w / 2, y + h / 2, teks, ha='center', va='center', fontsize=8.5, color=INK, linespacing=1.35)


def panah(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=10, color=INK2, lw=1))


kotak(1, 15, 16, 12, 'Data latih\n(train.csv)\nfitur: 32 kolom\n(30 tanpa G1/G2)', ABU)
kotak(27, 28, 22, 10, 'Kolom numerik (15)\nImputer median', BIRU)
kotak(27, 4, 22, 10, 'Kolom kategorik (17)\nImputer modus', ORANYE)
kotak(55, 28, 18, 10, 'StandardScaler\n(rata-rata 0, simpangan 1)', BIRU)
kotak(55, 4, 18, 10, 'One-hot encoding\nhandle_unknown="ignore"\nkeluaran padat', ORANYE)
kotak(79, 15, 20, 12, 'Penggabung\n(ColumnTransformer)\n58 fitur (56 tanpa G)\n-> model (Bab 4)', ABU)

panah(17.6, 23, 26.6, 31)
panah(17.6, 19, 26.6, 11)
panah(49.6, 33, 54.6, 33)
panah(49.6, 9, 54.6, 9)
panah(73.6, 33, 78.6, 25)
panah(73.6, 9, 78.6, 17)
ax.text(50, 0.5, 'Seluruh langkah di dalam Pipeline: dipelajari hanya dari data latih tiap fold, '
        'lalu diterapkan ke data validasi.', ha='center', fontsize=8, color=INK2)
fig.tight_layout()
fig.savefig(OUT / 'gambar_3_1_pipeline.png')
print('tersimpan:', OUT / 'gambar_3_1_pipeline.png')
