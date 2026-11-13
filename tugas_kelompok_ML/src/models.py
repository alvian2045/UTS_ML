"""Daftar model dan grid hiperparameter (Peran C).

Dipakai oleh scripts/run_cv.py dan scripts/train_final.py.
Awalan 'clf__' karena di pipeline, model diberi nama 'clf' (lihat buat_pipeline).

ALASAN TIAP RENTANG HIPERPARAMETER (wajib ada di laporan, Bab 4):

DecisionTree (Pertemuan 3)
  - max_depth [2,3,4,5,None]   : membatasi kedalaman = pra-pemangkasan. Pohon dalam
                                 menghafal data latih (overfitting); data kita kecil.
  - min_samples_leaf [1,5,10]  : daun harus berisi cukup siswa, supaya tidak ada
                                 aturan yang hanya berlaku untuk 1-2 siswa.
  - ccp_alpha [0,.005,.01,.02] : pemangkasan pasca-pembentukan (cost-complexity).
                                 Makin besar alpha, makin banyak cabang dipotong.

RandomForest (Pertemuan 4)
  - n_estimators [200,500]     : lebih banyak pohon = rata-rata lebih stabil;
                                 di atas ~500 hampir tidak ada tambahan manfaat.
  - max_depth [None,5,10]      : mengendalikan kerumitan tiap pohon.
  - min_samples_leaf [1,3,5]   : regularisasi, sama alasannya dengan pohon tunggal.

NaiveBayes (Pertemuan 6)
  - var_smoothing [1e-9..1e-1] : menambah varians kecil agar tidak ada varians nol
                                 (fitur one-hot sering konstan dalam satu kelas).
                                 Nilai besar = model lebih "halus".

SVM (Pertemuan 6)
  - kernel rbf / linear        : membandingkan batas keputusan lengkung vs lurus.
  - C [0.1,1,10]               : C kecil = margin lebar, toleran salah (bias tinggi);
                                 C besar = margin sempit, ketat (risiko overfitting).
  - gamma ['scale',0.01,0.1]   : hanya untuk rbf. Gamma besar = pengaruh tiap titik
                                 sempit (batas berkelok-kelok, overfitting).
  Grid ditulis sebagai dua kelompok supaya 'gamma' tidak dicoba pada kernel linear
  (gamma tidak dipakai di sana, jadi hanya membuang waktu).

kNN (Pertemuan 7)
  - n_neighbors [3..15]        : k kecil = varians tinggi (ikut noise); k besar =
                                 bias tinggi (terlalu halus). Bias-varians.
  - weights uniform/distance   : tetangga dekat boleh dihitung lebih berpengaruh.

CATATAN KETIDAKSEIMBANGAN KELAS:
  class_weight='balanced' dipakai pada model yang mendukungnya (DecisionTree,
  RandomForest, SVM). GaussianNB dan kNN tidak punya opsi itu: catat di laporan
  sebagai keterbatasan.
"""
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RS = 42

MODELS = {
    'Baseline': (DummyClassifier(strategy='most_frequent'), {}),
    'DecisionTree': (
        DecisionTreeClassifier(class_weight='balanced', random_state=RS),
        {'clf__max_depth': [2, 3, 4, 5, None],
         'clf__min_samples_leaf': [1, 5, 10],
         'clf__ccp_alpha': [0.0, 0.005, 0.01, 0.02]}),
    'RandomForest': (
        RandomForestClassifier(class_weight='balanced', random_state=RS),
        {'clf__n_estimators': [200, 500],
         'clf__max_depth': [None, 5, 10],
         'clf__min_samples_leaf': [1, 3, 5]}),
    'NaiveBayes': (
        GaussianNB(),
        {'clf__var_smoothing': [1e-9, 1e-6, 1e-3, 1e-1]}),
    'SVM': (
        SVC(class_weight='balanced', random_state=RS),
        [{'clf__kernel': ['rbf'],
          'clf__C': [0.1, 1, 10],
          'clf__gamma': ['scale', 0.01, 0.1]},
         {'clf__kernel': ['linear'],
          'clf__C': [0.1, 1, 10]}]),
    'kNN': (
        KNeighborsClassifier(),
        {'clf__n_neighbors': [3, 5, 7, 11, 15],
         'clf__weights': ['uniform', 'distance']}),
}
