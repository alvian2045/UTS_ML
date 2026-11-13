"""Angka dan grafik untuk Bab 2 (Data). Hanya memakai data/processed/train.csv.

Jalankan dari folder utama repo:  python scripts/eda_bab2.py
Keluaran: angka di layar + report/gambar/gambar_2_1_kelas.png, gambar_2_2_g3.png, gambar_2_3_fitur.png
"""
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'report' / 'gambar'
OUT.mkdir(parents=True, exist_ok=True)

t = pd.read_csv(ROOT / 'data' / 'processed' / 'train.csv')   # TIDAK memuat test.csv

# warna: biru = lulus, oranye = tidak lulus; teks netral
INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e3e2de', '#fcfcfb'
C_LULUS, C_TIDAK = '#2a78d6', '#eb6834'


def rapikan(ax):
    ax.set_facecolor(SURFACE)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


# ---------- angka ----------
n = len(t)
kelas = t['tidak_lulus'].value_counts().sort_index()
print(f'Bentuk data latih: {t.shape} | nilai hilang: {t.isna().sum().sum()} | duplikat: {t.duplicated().sum()}')
print(f'Lulus: {kelas[0]} ({kelas[0] / n:.1%}) | Tidak lulus: {kelas[1]} ({kelas[1] / n:.1%})')
print('Korelasi G1, G2, G3:\n', t[['G1', 'G2', 'G3']].corr().round(3))
print(f'G3 = 0: {(t.G3 == 0).sum()} siswa, semuanya tidak lulus? '
      f'{((t.G3 == 0) & (t.tidak_lulus == 1)).sum() == (t.G3 == 0).sum()}')
print('Tingkat tidak lulus menurut failures:\n', t.groupby('failures')['tidak_lulus'].agg(['count', 'mean']).round(3))

# ---------- Gambar 2.1: kelas target ----------
fig, ax = plt.subplots(figsize=(5.5, 4), dpi=200, facecolor=SURFACE)
rapikan(ax)
b = ax.bar(['Lulus', 'Tidak lulus'], [kelas[0], kelas[1]], color=[C_LULUS, C_TIDAK], width=0.5)
for r, v in zip(b, [kelas[0], kelas[1]]):
    ax.text(r.get_x() + r.get_width() / 2, v + 4, f'{v} ({v / n:.1%})', ha='center', color=INK, fontsize=10)
ax.set_ylabel('Jumlah siswa (data latih)', color=INK2)
ax.set_title('Distribusi kelas target', color=INK, loc='left')
ax.set_ylim(0, kelas.max() * 1.15)
fig.tight_layout()
fig.savefig(OUT / 'gambar_2_1_kelas.png')
plt.close(fig)

# ---------- Gambar 2.2: histogram G3 ----------
fig, ax = plt.subplots(figsize=(7, 4), dpi=200, facecolor=SURFACE)
rapikan(ax)
bins = range(0, 21)
ax.hist(t.loc[t.G3 >= 10, 'G3'], bins=bins, color=C_LULUS, label='Lulus (G3 ≥ 10)', rwidth=0.9)
ax.hist(t.loc[t.G3 < 10, 'G3'], bins=bins, color=C_TIDAK, label='Tidak lulus (G3 < 10)', rwidth=0.9)
ax.axvline(10, color=INK2, linestyle='--', linewidth=1)
ax.text(9.85, ax.get_ylim()[1] * 0.92, 'ambang 10', color=INK2, fontsize=9, ha='right')
n0 = int((t.G3 == 0).sum())
ax.annotate(f'{n0} siswa bernilai 0', xy=(0.5, n0), xytext=(2.2, n0 + 3),
            color=INK2, fontsize=9, arrowprops=dict(arrowstyle='-', color=INK2, lw=0.8))
ax.set_xlabel('Nilai akhir G3 (skala 0 sampai 20)', color=INK2)
ax.set_ylabel('Jumlah siswa', color=INK2)
ax.set_title('Distribusi nilai akhir G3 pada data latih', color=INK, loc='left')
ax.legend(frameon=False, labelcolor=INK)
fig.tight_layout()
fig.savefig(OUT / 'gambar_2_2_g3.png')
plt.close(fig)

# ---------- Gambar 2.3: failures dan absences ----------
fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=200, facecolor=SURFACE)
g = t.groupby('failures')['tidak_lulus'].agg(['count', 'mean'])
rapikan(axes[0])
bars = axes[0].bar(g.index.astype(str), g['mean'] * 100, color=C_TIDAK, width=0.55)
for r, (_, row) in zip(bars, g.iterrows()):
    axes[0].text(r.get_x() + r.get_width() / 2, row['mean'] * 100 + 2,
                 f"{row['mean']:.0%}\n(n={int(row['count'])})", ha='center', color=INK, fontsize=8)
axes[0].set_ylim(0, 115)
axes[0].set_xlabel('Jumlah kegagalan kelas sebelumnya (failures)', color=INK2)
axes[0].set_ylabel('% siswa tidak lulus', color=INK2)
axes[0].set_title('Tingkat tidak lulus menurut riwayat kegagalan', color=INK, loc='left', fontsize=10)

rapikan(axes[1])
data = [t.loc[t.tidak_lulus == 0, 'absences'], t.loc[t.tidak_lulus == 1, 'absences']]
bp = axes[1].boxplot(data, tick_labels=['Lulus', 'Tidak lulus'], patch_artist=True, widths=0.5,
                     medianprops=dict(color=INK), flierprops=dict(markersize=3, markeredgecolor=INK2))
for patch, c in zip(bp['boxes'], [C_LULUS, C_TIDAK]):
    patch.set_facecolor(c)
    patch.set_alpha(0.85)
axes[1].set_ylabel('Jumlah ketidakhadiran (absences)', color=INK2)
axes[1].set_title('Ketidakhadiran menurut kelas', color=INK, loc='left', fontsize=10)
fig.tight_layout()
fig.savefig(OUT / 'gambar_2_3_fitur.png')
plt.close(fig)
# ---------- Gambar 2.4: G2 terhadap G3 ----------
import numpy as np  # noqa: E402

fig, ax = plt.subplots(figsize=(5.5, 4.2), dpi=200, facecolor=SURFACE)
rapikan(ax)
ax.xaxis.grid(True, color=GRID, linewidth=0.8)
rng = np.random.default_rng(1)          # sedikit getaran agar titik kembar tidak menumpuk
for k, (lab, c) in {0: ('Lulus', C_LULUS), 1: ('Tidak lulus', C_TIDAK)}.items():
    d = t[t.tidak_lulus == k]
    ax.scatter(d.G2 + rng.normal(0, .12, len(d)), d.G3 + rng.normal(0, .12, len(d)), s=14,
               color=c, alpha=.75, edgecolor=SURFACE, linewidth=.4, label=lab)
ax.axhline(9.5, color=INK2, linewidth=.9, linestyle=':')
r = t[['G2', 'G3']].corr().iloc[0, 1]
ax.set_xlabel('Nilai periode kedua (G2)', color=INK2)
ax.set_ylabel('Nilai akhir (G3)', color=INK2)
ax.set_title(f'G2 dan G3 (korelasi = {r:.2f})', color=INK, loc='left')
ax.legend(frameon=False, loc='lower right', labelcolor=INK)
fig.tight_layout()
fig.savefig(OUT / 'gambar_2_4_g2_g3.png')
plt.close(fig)
print('Grafik tersimpan di', OUT)
