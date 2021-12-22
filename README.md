# Portofolio
Data science portofolio

# Project 1: Predictive Analytics
# Domain Project
Domain yang saya ambil di sini adalah pada bidang Ekonomi dan Bisnis. Saya akan mengambil topik tentang harga emas. Karena harga emas sering digunakan sebagai standar nilai mata uang dunia serta harga yang fluktuatif mengikuti kondisi pasar, maka prediksi harga emas sangat dibutuhkan bagi banyak orang yang sering berinvestasi pada instrumen ini. Proyek ini bertujuan untuk memprediksi harga optimal emas berdasarkan data-data pada tahun sebelumnya.

Artikel yang diterbitkan oleh Kompas.com mengenai penyebab harga emas sering naik dan turun. (https://money.kompas.com/read/2021/06/12/110000026/ini-penyebab-harga-emas-sering-naik-dan-turun?page=all)

# Business Understanding
Problem Statement
Dengan harga emas yang fluktuatif setiap saat. Bagaimana kita menentukan harga optimalnya dalam kasus ingin berinvestasi emas di harga yang sesuai atau lebih rendah dan menghindari berinvestasi saat harga jauh di atas harga normal.

# Goals
Maka diperlukan suatu sistem pembelajaran mesin yang berguna untuk membantu dalam memprediksi harga emas di kemudian hari.

# Solution Statements
Dengan menggunakan pendekatan model machine learning dalam 3 algoritma yang berbeda yaitu:

* KNN Algoritma KNN menggunakan ‘kesamaan fitur’ untuk memprediksi nilai dari setiap data yang baru. KNN bekerja dengan membandingkan jarak satu sampel ke sampel pelatihan lain dengan memilih sejumlah k-tetangga terdekat.
* RF Random forest pada dasarnya adalah versi bagging dari algoritma decision tree. Jadi dalam algoritma RF kita memiliki banyak algoritma decision tree di dalamnya.
* Boosting Algorithm Algoritma boosting bekerja dengan membangun model dari data latih. Kemudian ia membuat model kedua yang bertugas memperbaiki kesalahan dari model pertama. Model ditambahkan sampai data latih terprediksi dengan baik atau telah mencapai jumlah maksimum model untuk ditambahkan.
Data Understanding
Dataset yang saya pakai adalah dataset Gold Price yang berisi harga emas dari Januari 2011 sampai dengan September 2020. Dataset ini berisi 2531 baris data dan 6 kolom. Linknya adalah berikut (https://www.kaggle.com/shikhnu/gold-price).

Variabel-variabel pada Gold Price dataset adalah sebagai berikut.

* Date : Tanggal
* Price : Harga penutupan di hari itu
* Open : Harga pembukaan di hari itu
* High : Harga tertinggi di hari itu
* Low : Harga terendah di hari itu
* Chg% : Perubahan persentase dari harga penutupan hari sebelumnya

Melakukan gold.info() untuk mengecek jenis tipe data apa saja yang ada pada dataset. Dan hasilnya adalah 5 data numerik yang dipakai adalah
