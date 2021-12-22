# kelvin_portofolio
Data science portofolio

# Project 1: [Predictive Analytics](https://github.com/kola283/Predictive-Analysis)
## Domain Project
Domain yang saya ambil di sini adalah pada bidang Ekonomi dan Bisnis. Saya akan mengambil topik tentang harga emas. Karena harga emas sering digunakan sebagai standar nilai mata uang dunia serta harga yang fluktuatif mengikuti kondisi pasar, maka prediksi harga emas sangat dibutuhkan bagi banyak orang yang sering berinvestasi pada instrumen ini. Proyek ini bertujuan untuk memprediksi harga optimal emas berdasarkan data-data pada tahun sebelumnya.

Artikel yang diterbitkan oleh Kompas.com mengenai penyebab harga emas sering naik dan turun. [link](https://money.kompas.com/read/2021/06/12/110000026/ini-penyebab-harga-emas-sering-naik-dan-turun?page=all)

## Business Understanding
Problem Statement
Dengan harga emas yang fluktuatif setiap saat. Bagaimana kita menentukan harga optimalnya dalam kasus ingin berinvestasi emas di harga yang sesuai atau lebih rendah dan menghindari berinvestasi saat harga jauh di atas harga normal.

## Goals
Maka diperlukan suatu sistem pembelajaran mesin yang berguna untuk membantu dalam memprediksi harga emas di kemudian hari.

## Solution Statements
Dengan menggunakan pendekatan model machine learning dalam 3 algoritma yang berbeda yaitu:

* KNN Algoritma KNN menggunakan ‘kesamaan fitur’ untuk memprediksi nilai dari setiap data yang baru. KNN bekerja dengan membandingkan jarak satu sampel ke sampel pelatihan lain dengan memilih sejumlah k-tetangga terdekat.
* RF Random forest pada dasarnya adalah versi bagging dari algoritma decision tree. Jadi dalam algoritma RF kita memiliki banyak algoritma decision tree di dalamnya.
* Boosting Algorithm Algoritma boosting bekerja dengan membangun model dari data latih. Kemudian ia membuat model kedua yang bertugas memperbaiki kesalahan dari model pertama. Model ditambahkan sampai data latih terprediksi dengan baik atau telah mencapai jumlah maksimum model untuk ditambahkan.
Data Understanding
Dataset yang saya pakai adalah dataset Gold Price yang berisi harga emas dari Januari 2011 sampai dengan September 2020. Dataset ini berisi 2531 baris data dan 6 kolom. Linknya adalah berikut [Dataset](https://www.kaggle.com/shikhnu/gold-price).

Variabel-variabel pada Gold Price dataset adalah sebagai berikut.

* Date : Tanggal
* Price : Harga penutupan di hari itu
* Open : Harga pembukaan di hari itu
* High : Harga tertinggi di hari itu
* Low : Harga terendah di hari itu
* Chg% : Perubahan persentase dari harga penutupan hari sebelumnya

Melakukan gold.info() untuk mengecek jenis tipe data apa saja yang ada pada dataset. Dan hasilnya adalah 5 data numerik yang dipakai adalah

![image](https://github.com/kola283/portofolio/blob/main/images/infodata.JPG?raw=true)

Lalu melakukan teknik pairplot() untuk melihat keterikatan tiap fitur yang ada terhadap target price dan hasilnya adalah 3 fitur yaitu Open, High dan Low memiliki korelasi positif. Sementara fitur Chg% memiliki pola persebaran yang acak.

![image](https://github.com/kola283/portofolio/blob/main/images/korelasi.png?raw=true)

## Data Preparation
Disini dilakukan pembagian dataset menjadi data training dan data test dengan proporsi 80% data training dan 20% data latih dengan menggunakan teknik train_test_split. Hal ini sangat penting dilakukan untuk melakukan evaluasi model kedepannya. Selain menyiapkan data dengan membagi datanya, dilakukan satu teknik lagi yaitu Standarisasi. Dengan menggunakan teknik StandardScaler dari library Scikitlearn. Untuk menghindari kebocoran data maka kita melakukan teknik standarisasi pada data training terlebih dahulu dan melakukan standarisasi pada data test saat akan evaluasi model.

## Modeling
Data yang siap uji akan dicoba dengan 3 model yang berbeda.
1. KNN
Parameter yang digunakan (n_neighbors = 3)
Dengan menggunakan model KNN regresi, model bekerja dengan cara membandingkan jarak satu sampel ke sampel pelatihan lain dengan memilih sejumlah k-tetangga terdekat. Dalam kasus ini jumlah k-tetangga nya adalah 3.

2. RF
Parameter yang digunakan (n_estimators = 50, max_depth = 15, random_state = 55, n_jobs = -1)
Dengan menggunakan model RF regresi, model bekerja dengan cara kumpulan algoritma Decision Tree yang terkumpul pada RF melakukan prediksi secara independen lalu hasil dari prediksi tersebut akan digabungkan untuk dijadikan prediksi akhir.

3. Boosting Algorithm
Parameter yang digunakan (n_estimators = 50, learning_rate = 0.05, random_state = 55)
Dengan menggunakan model Boosting Algorithm, model bekerja dengan membangun model dari data latih. Kemudian ia membuat model kedua yang bertugas memperbaiki kesalahan dari model pertama. Model ditambahkan sampai data latih terprediksi dengan baik atau telah mencapai jumlah maksimum model untuk ditambahkan. Lalu menggabungkan beberapa model sederhana dan dianggap lemah (weak learners) sehingga membentuk suatu model yang kuat (strong ensemble learner).


* Data dari dataset : 1317.7
* Prediksi_KNN : 1368.7
* Prediksi _RF : 1320.9
* Prediksi_Boosting : 1349.3

Dapat dilihat prediksi model yang sangat mendekati nilai dari dataset adalah model RF. Maka model yang digunakan untuk pengembangan lebih lanjut adalah model RF.

## Evaluation
Metrik yang akan kita gunakan pada prediksi ini adalah MSE atau Mean Squared Error yang menghitung selisih rata-rata nilai sebenarnya dengan nilai prediksi. MSE didefinisikan dalam persamaan berikut
![Image of rumus mse](https://d17ivq9b7rppb3.cloudfront.net/original/academy/2021071619431112f1106e20559e77c855cea11d1b1479.jpeg)

Keterangan:
- N = jumlah dataset
- yi = nilai sebenarnya
- y_pred = nilai prediksi

Dengan penggunaan metrik di atas hasil yang didapat adalah saat pengujian dengan dataset bernilai 1317.7 prediksi yang didapat dari model adalah 1320.9
Ini sudah sangat mendekati dibandingkan dua model yang lainnya.
