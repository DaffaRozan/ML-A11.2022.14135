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
![App Screenshot](./image/image1)
Distribusi depresi
![App Screenshot](./image/image2)
pengelompokan depresi
![App Screenshot](./image/image3)


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
1. Status Depresi
kita dapat melihat bahwa mayoritas mahasiswa tidak mengalami depresi (kode 0), sementara sebagian kecil mengalami depresi (kode 1). Jumlah mahasiswa yang tidak mengalami depresi jauh lebih banyak dibandingkan dengan yang mengalami depresi.

2. Depresi Berdasarkan Umur
yang menggambarkan distribusi depresi berdasarkan umur mahasiswa, terlihat bahwa:
Pada usia 18 tahun, jumlah mahasiswa yang tidak mengalami depresi lebih banyak daripada yang mengalami depresi.
Pada usia 19 dan 20 tahun, jumlah mahasiswa yang tidak mengalami depresi juga lebih banyak, meskipun ada sedikit peningkatan pada jumlah mahasiswa yang mengalami depresi.
Pada usia 21 tahun, jumlah mahasiswa yang mengalami depresi lebih sedikit.
Pada usia 22 dan 23 tahun, tidak ada data mengenai mahasiswa yang mengalami depresi.
Pada usia 24 tahun, terdapat peningkatan jumlah mahasiswa yang mengalami depresi.

3. Kecemasan Berdasarkan Umur
yang menggambarkan distribusi kecemasan berdasarkan umur mahasiswa, terlihat bahwa:
Pada usia 18 tahun, jumlah mahasiswa yang mengalami kecemasan lebih banyak daripada yang tidak mengalami kecemasan.
Pada usia 19 dan 20 tahun, jumlah mahasiswa yang mengalami kecemasan sedikit lebih tinggi.
Pada usia 21 dan 22 tahun, tidak ada data mengenai mahasiswa yang mengalami kecemasan.
Pada usia 23 dan 24 tahun, jumlah mahasiswa yang mengalami kecemasan lebih banyak daripada yang tidak mengalami kecemasan.

4. Serangan Panik Berdasarkan Umur
yang menggambarkan distribusi serangan panik berdasarkan umur mahasiswa, terlihat bahwa:
Pada usia 18 tahun, jumlah mahasiswa yang tidak mengalami serangan panik lebih banyak daripada yang mengalami serangan panik.
Pada usia 19 dan 20 tahun, jumlah mahasiswa yang mengalami serangan panik sedikit lebih tinggi.
Pada usia 21 dan 22 tahun, tidak ada data mengenai mahasiswa yang mengalami serangan panik.
Pada usia 23 dan 24 tahun, jumlah mahasiswa yang mengalami serangan panik lebih sedikit.

5. Konsultasi Berdasarkan Umur
Pada grafik kelima, yang menggambarkan distribusi konsultasi berdasarkan umur mahasiswa, terlihat bahwa:
Pada usia 18 tahun, jumlah mahasiswa yang tidak melakukan konsultasi lebih banyak daripada yang melakukan konsultasi.
Pada usia 19 dan 20 tahun, jumlah mahasiswa yang melakukan konsultasi sedikit lebih tinggi.
Pada usia 21 dan 22 tahun, tidak ada data mengenai mahasiswa yang melakukan konsultasi.
Pada usia 23 dan 24 tahun, jumlah mahasiswa yang melakukan konsultasi lebih banyak.

## Kesimpulan
Depresi: Lebih banyak mahasiswa yang tidak mengalami depresi dibandingkan yang mengalami, dengan peningkatan pada usia 24 tahun.
Kecemasan: Lebih banyak mahasiswa yang mengalami kecemasan, terutama pada usia 18, 19, dan 24 tahun.
Serangan Panik: Lebih banyak mahasiswa yang tidak mengalami serangan panik, dengan sedikit peningkatan pada usia 19 dan 20 tahun.
Konsultasi: Lebih banyak mahasiswa yang tidak melakukan konsultasi, dengan peningkatan pada usia 23 dan 24 tahun.

