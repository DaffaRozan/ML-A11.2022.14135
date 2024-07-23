# ML-A11.2022.14135
## Daffa Rozan F
## A11.2022.14135

# Penggunaan machine learning untuk klasifikasi gangguan mental pada mahasiswa menggunakan metode Naive Bayes

## Ringkasan dan Permasalahan

Penggunaan machine learning dengan metode Naive Bayes untuk klasifikasi gangguan mental pada mahasiswa menangani masalah yang berkaitan dengan kesejahteraan mental di lingkungan perguruan tinggi. Kesehatan mental mahasiswa menjadi semakin penting dengan tingginya tingkat stres, depresi, dan kecemasan di kalangan mereka.

## Tujuan

Tujuan dari penggunaan machine learning dengan metode Naive Bayes untuk klasifikasi gangguan mental pada mahasiswa adalah untuk membantu dalam identifikasi dini gangguan mental dengan mengenali pola-pola dalam data mahasiswa, sehingga memungkinkan pemberian dukungan dan intervensi yang tepat waktu. Dengan demikian, kesejahteraan mental mahasiswa dapat didukung secara optimal melalui tindakan yang didasarkan pada hasil klasifikasi yang akurat.

## Model

Model yang digunakan dalam penelitian ini adalah model Naive Bayes. Model ini dilatih menggunakan data pelatihan yang mencakup riwayat medis, perilaku, dan faktor-faktor lingkungan mahasiswa
## Alur Project

Pengumpulan dan Ekstraksi Data

    File zip berisi data Student mental health.

Import dan Pembersihan Data

    Data diimport menggunakan pandas dan berbagai library lainnya, kemudian dilakukan pembersihan data, termasuk konversi Student mental health menjadi numerik.

Analisis Data Awal

    Data awal dianalisis dengan visualisasi menggunakan seaborn untuk melihat Kesehatan mental pada siswa dan mahasiswa.

Prediksi Kesehatan Masa Depan

    Prediksi mental health untuk  lima tahun ke depan dilakukan menggunakan model Naive Bayes yang telah dibuat.
    Data hasil prediksi digabungkan dengan data historis untuk analisis lebih lanjut.

Visualisasi Prediksi

    Pertumbuhan rata-rata kesehatan dari tahun 2020 hingga 2024 divisualisasikan.
    Perubahan Kesehatan mental dari tahun 2020 hingga 2024 divisualisasikan dalam bentuk diagram batang.
## Dataset

Dataset berasal dari Kaggle
Dataset didapatkan dari Kaggle (https://www.kaggle.com/datasets/shariful07/student-mental-health, diakses 29 April 2024) berisi Timestamp, Umur, Pilih Gender, Tahun keberapa Pembelajaran, Jurusan yang dipilih, Target IPK, Status sudah menikah atau belum.



### EDA (Exploratory Data Analysis)??

Import Libraries dan Data
Preview Data
![App Screenshot](./image/image2.png)
Distribusi Nilai UMP
![App Screenshot](./image/image3.png)
Rata-rata UMP per Tahun
![App Screenshot](./image/image4.png)
Top 5 Provinsi dengan UMP Tertinggi
![App Screenshot](./image/image.png)

### Proses features Dataset
Preprocessing Data
categorical_col = []
for column in data.columns:
    if data[column].dtype == object and len(data[column].unique()) <= 50:
        categorical_col.append(column)
        print(f"{column} : {data[column].unique()}")
        print("====================================") 
models = {}
X = data.drop('Depresi', axis=1)
y = data['Depresi']

X.head()

## Proses Learning/Modeling

1. Prediksi Depresi
sns.set(font_scale=1.4)
data['Depresi'].value_counts().plot(kind='bar', figsize=(10, 6), rot=0)
plt.xlabel("Status Depresi", labelpad=14)
plt.ylabel("Jumlah", labelpad=14)
plt.title("Depresot", y=1.02);

plt.subplots(figsize = (25,5))
sns.countplot(x=data['Umur'],order=data['Umur'].value_counts().index.sort_values(ascending=True),hue=data['Depresi'])
plt.show()

plt.subplots(figsize = (25,5))
sns.countplot(x=data['Umur'],order=data['Umur'].value_counts().index.sort_values(ascending=True),hue=data['Kecemasan'])
plt.show()

plt.subplots(figsize = (25,5))
sns.countplot(x=data['Umur'],order=data['Umur'].value_counts().index.sort_values(ascending=True),hue=data['Panic attack'])
plt.show()

plt.subplots(figsize = (25,5))
sns.countplot(x=data['Umur'],order=data['Umur'].value_counts().index.sort_values(ascending=True),hue=data['Konsultasi'])
plt.show()



## Perfoma Model

Koefisien Determinasi (R²):
    R² berkisar 0.29999999999999993 untuk berbagai mental health, yang menunjukkan bahwa model Anda mampu menjelaskan sebagian besar variabilitas data.

Mean Absolute Error (MAE):
    MAE dari sekitar 0.14285714285714285. Nilai MAE yang lebih rendah menunjukkan bahwa prediksi model Anda lebih akurat, dan MAE di bawah 200,000 pada sebagian besar menunjukkan bahwa kesalahan prediksi relatif kecil.

Mean Squared Error (MSE):
    MSE bervariasi cukup luas, tetapi sebagian besar nilai MSE menunjukkan bahwa model Anda dapat memprediksi dengan baik. MSE yang lebih tinggi menunjukkan variabilitas yang lebih besar dalam prediksi model.

## Diskusi Hasil dan Kesimpulan

### Hasil
Prediksi :

Tren Kenaikan Gaji:
Terdapat tren kenaikan gaji rata-rata di Indonesia dari tahun ke tahun.
Visualisasi menunjukkan peningkatan yang konsisten, dengan beberapa tahun mengalami pertumbuhan yang lebih signifikan dibandingkan tahun lainnya.

Pertumbuhan Gaji Rata-rata:
Grafik batang menunjukkan pertumbuhan gaji rata-rata tahunan dari 2022 hingga 2032.
Pertumbuhan gaji diprediksi akan terus meningkat, namun dengan laju yang bervariasi setiap tahunnya.

Lima Wilayah dengan Gaji Tertinggi:
Analisis menunjukkan lima wilayah dengan gaji tertinggi di Indonesia.
DKI Jakarta konsisten menjadi wilayah dengan gaji tertinggi, diikuti oleh provinsi-provinsi lain yang kemungkinan besar merupakan daerah industri atau pusat ekonomi.

Variasi Gaji antar Wilayah:
Terdapat kesenjangan gaji yang signifikan antar wilayah di Indonesia.
Beberapa wilayah menunjukkan pertumbuhan gaji yang lebih cepat dibandingkan wilayah lain.

Prediksi Gaji Masa Depan:
Model regresi linear digunakan untuk memprediksi gaji hingga tahun 2032.
Prediksi menunjukkan bahwa gaji akan terus meningkat di semua wilayah, namun dengan laju yang berbeda-beda.

Visualisasi Dinamis:
Grafik animasi menunjukkan perubahan gaji di berbagai wilayah dari tahun ke tahun.
Visualisasi ini memperlihatkan dinamika perubahan peringkat wilayah berdasarkan tingkat gaji.

## Kesimpulan
Hasil yang ditargetkan adalah pengembangan model klasifikasi yang dapat mengidentifikasi kemungkinan adanya gangguan mental pada mahasiswa dengan tingkat akurasi yang tinggi. Model ini diharapkan dapat memberikan landasan untuk intervensi yang tepat waktu dan dukungan yang sesuai untuk meningkatkan kesejahteraan mental mahasiswa secara keseluruhan.

