# Bab 1. Pendahuluan dan Perumusan Masalah

## 1.1 Latar belakang

Kegagalan siswa dalam suatu mata pelajaran biasanya baru diketahui setelah nilai akhir keluar, ketika sekolah sudah tidak punya banyak waktu untuk membantu. Jika siswa yang berisiko tidak lulus dapat dikenali lebih awal, guru dan sekolah dapat memberi dukungan tambahan, seperti bimbingan belajar, pendampingan, atau komunikasi dengan orang tua, selagi masih ada kesempatan memperbaiki hasil. Pendekatan ini sejalan dengan tujuan penelitian asal dataset yang kami pakai, yaitu memanfaatkan data siswa sekolah menengah untuk memprediksi prestasi belajar (Cortez dan Silva, 2008).

Pembelajaran mesin cocok untuk persoalan ini karena hubungan antara latar belakang siswa (misalnya riwayat kegagalan, ketidakhadiran, waktu belajar, dan pendidikan orang tua) dan hasil akhirnya sulit dituliskan sebagai aturan manual. Model dapat mempelajari pola tersebut dari data historis. Namun hasilnya harus dinilai dengan jujur, terutama karena data yang tersedia kecil dan kelas yang kita pedulikan (siswa tidak lulus) biasanya lebih sedikit daripada kelas lulus.

Pengembangan sistem peringatan dini (*Early Warning Systems* atau EWS) berbasis data di institusi pendidikan telah banyak diteliti, antara lain oleh Balfanz dkk. (2007) serta Heppen dan Therriault (2008), yang menunjukkan bahwa indikator kinerja akademik awal, tingkat ketidakhadiran, dan riwayat kegagalan merupakan prediktor kunci kegagalan akademik. Pemanfaatan model klasifikasi prediktif memungkinkan intervensi dan bimbingan belajar dilakukan secara proaktif sebelum siswa mencapai kegagalan permanen pada evaluasi akhir.

## 1.2 Rumusan masalah dan tujuan

**Rumusan masalah.**
1. Seberapa baik model klasifikasi klasik dari Pertemuan 2 sampai 7 dapat memprediksi apakah seorang siswa tidak lulus mata pelajaran Matematika berdasarkan data latar belakang dan nilai periode sebelumnya?
2. Seberapa besar nilai periode pertama dan kedua (G1 dan G2) memengaruhi kemampuan prediksi, dan seberapa baik model bekerja tanpa keduanya, yaitu pada situasi peringatan dini?
3. Bagaimana cara menilai model secara jujur ketika data berukuran kecil dan kelasnya tidak seimbang?

**Tujuan.**
1. Membangun dan membandingkan beberapa model klasifikasi (Decision Tree, Random Forest, Naive Bayes, SVM, dan kNN) dengan baseline sebagai pembanding.
2. Membandingkan dua skenario fitur: dengan dan tanpa G1 dan G2.
3. Menerapkan protokol evaluasi yang mencegah kebocoran data: pemisahan data uji di awal, praproses di dalam pipeline, validasi silang bersarang, dan penggunaan data uji satu kali.

## 1.3 Perumusan masalah pembelajaran (T, P, E)

Mengikuti definisi Mitchell (1997), sebuah program dikatakan belajar dari pengalaman E terhadap tugas T dengan ukuran performa P, jika performanya pada T yang diukur dengan P membaik karena E.

| Unsur | Isi dalam proyek ini |
|---|---|
| **T** (tugas) | Klasifikasi biner: memprediksi apakah siswa **tidak lulus** (nilai akhir G3 di bawah 10) atau lulus. |
| **P** (ukuran performa) | F1, recall, dan precision untuk kelas "tidak lulus", serta ROC-AUC. Akurasi saja tidak dipakai karena menyesatkan pada data yang tidak seimbang. |
| **E** (pengalaman) | Data historis siswa (atribut demografis, sosial, dan sekolah, serta nilai periode sebelumnya) beserta hasil akhirnya. |

Kelas positif adalah **tidak lulus**. Alasannya, tujuan kita menangkap siswa yang berisiko, sehingga recall dan F1 pada kelas ini langsung mengukur hal yang kita pedulikan: berapa banyak siswa berisiko yang berhasil dikenali (recall) dan seberapa tepat penandaannya (precision).

## 1.4 Ruang lingkup dan batasan

- Data berasal dari satu mata pelajaran, yaitu **Matematika** (`student-mat.csv`), dari dua sekolah menengah di Portugal pada satu periode pengumpulan data.
- Model yang dipakai terbatas pada model klasik dari Pertemuan 2 sampai 7.
- Ambang kelulusan ditetapkan 10 pada skala 0 sampai 20. Ambang lain akan mengubah proporsi kelas dan hasil, dan ini dibahas sebagai keterbatasan.
- Hasil tidak dimaksudkan untuk diterapkan langsung pada sekolah lain tanpa validasi tambahan, karena data berasal dari satu konteks dan berukuran kecil.

## 1.5 Sistematika laporan

Bab 2 membahas data dan analisis eksplorasi. Bab 3 membahas praproses dan pipeline. Bab 4 membahas model dan pelatihan. Bab 5 menjelaskan protokol evaluasi, Bab 6 menyajikan hasil, Bab 7 membahas analisis kesalahan, dan Bab 8 membahas keterbatasan dan saran.

## Daftar sumber untuk bab ini

- Balfanz, R., Herzog, L., & Mac Iver, D. J. (2007). Preventing student disengagement and keeping students on the graduation path in urban middle-grades schools: Early identification and effective student support. *Educational Psychologist, 42*(4), 223–235.
- Cortez, P., & Silva, A. (2008). Using data mining to predict secondary school student performance. *Proceedings of FUBUTEC 2008*, 5–12. Dataset: UCI Machine Learning Repository, Student Performance.
- Heppen, J. B., & Therriault, S. B. (2008). *Developing Early Warning Systems to Identify Potential High School Dropouts*. National High School Center, American Institutes for Research.
- Mitchell, T. M. (1997). *Machine Learning*. McGraw-Hill.
