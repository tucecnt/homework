
📌 **UYARI:**  Python dosyalarını çalıştırırken, terminalde projenin kök dizini olan `homework/` klasörünün içinde olun. Aksi halde `"data/hamveri/store_zara.csv"` gibi göreli yollar hata verebilir.
---

# 👗 Moda Kombin Uyumu Analizi — NLP Projesi

Bu proje, Zara ürünlerinin açıklamalarını kullanarak, **moda kombin uyumu analizi** yapmayı amaçlar. Doğal Dil İşleme (NLP) teknikleri ile ürünler arasında **anlamsal benzerlik** hesaplanarak kombin önerileri yapılabilir.

---

## 🎯 Projenin Amacı

Zara mağazasına ait ürünlerin açıklamaları analiz edilerek, stil olarak birbiriyle **uyumlu ürün çiftleri** tespit edilmiştir. Böylece, görsel bilgiye gerek kalmadan yalnızca açıklamalara dayanarak kullanıcıya kombin önerileri yapılabilir.

---

## 📊 Kullanılan Veri Seti

* **Veri Seti Adı:** [Zara Sales Products Analysis](https://www.kaggle.com/datasets/kingabzpro/zara-sales-products-analysis)
* **Dosya:** `data/hamveri/store_zara.csv`
* **İçerik:** Ürün adı, açıklaması, fiyat, kategori ve renk bilgileri
* **Amaç:**

  * Ürün açıklamalarından vektör temsilleri (TF-IDF, Word2Vec) üretmek
  * Ürünler arası anlamsal benzerlikleri tespit ederek kombin önerileri oluşturmak

---

## ⚙️ Projenin Çalıştırılması ve Modelin Oluşturulması

Aşağıdaki adımlar, repo üzerinden modeli nasıl oluşturacağınızı adım adım açıklar:

### 1. Veri Ön İşleme

```bash
src/data_process.py
```

* Veri temizleme (noktalama, küçük harfe çevirme)
* Lemmatization & Stemming
* Stopwords temizleme

### 2. TF-IDF Vektörleme

```bash
src/tf_idf.py
```

* TF-IDF matrisleri oluşturulur (`data/processveri/tfidf_*.csv`)
* Farklı ön işleme türleriyle (lemma / stem) varyasyonlar kaydedilir

### 3. Word2Vec Model Eğitimi

```bash
src/word2.py
```

* CBOW ve Skip-gram ile 4 farklı pencere ve boyut kombinasyonunda modeller üretilir
* Modeller `models/` klasörüne `.model` olarak kaydedilir

### 4. Benzer Ürün Analizi & Raporlama

```bash
report/similar_words_report.csv
```

* TF-IDF ve Word2Vec ile oluşturulan vektör temsilleri ile benzer ürünler bulunur
* Elde edilen sonuçlar CSV dosyasında raporlanır

---

## 🛠️ Gerekli Kütüphaneler ve Kurulum

Python ortamınıza aşağıdaki kütüphaneleri kurmanız gerekmektedir:

### 📦 Python Paketleri

```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn tqdm
```

### 📥 NLTK Veri Setleri

Projede kullanılan NLTK verilerini indirmeniz gerekir. Terminalde Python'u açıp şunları yazın:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

---

## 📁 Dosya Yapısı ve Önemli Not

Bu repo içerisindeki dosyalar şu yapıda organize edilmiştir:

```
homework/
├── README.md
├── data/
│   ├── hamveri/store_zara.csv
│   └── processveri/...
├── models/
│   └── .model dosyaları
├── report/
│   └── similar_words_report.csv
├── src/
│   ├── data_process.py
│   ├── tf_idf.py
│   ├── word2.py
│   └── diğer script dosyaları
```

Final klasörü için yazılan kodlar,
## Çalıştırma Talimatları

Projede kullanılan final dersi için gereken hesaplamalar `final/` klasörü altında yer almaktadır. Aşağıda her bir işlem için nasıl çalıştırma yapılacağı adım adım açıklanmıştır.

### 1. TF-IDF Benzerlik Hesaplama

- TF-IDF modellerinin benzerlik skorları ve en benzer 5 metin çıkarımı için:
```bash
python final/tf_model_benzerlik.py
```
### 2. Word2Vec Benzerlik Hesaplama

* Word2Vec modelleri (CBOW ve SkipGram, farklı pencere ve vektör boyutları ile) kullanılarak benzerlik skorları hesaplanır ve en benzer 5 metin çıkarılır.

**Çalıştırma komutu:**

```bash
python final/word2_model_benzerlik.py
```

* Çıktı dosyası:
  `results_word2vec_top5_with_scores.csv`

---

## 3. Manuel Anlamsal Puanların Eklenmesi

* Elde edilen benzerlik sonuçlarına manuel olarak verilen anlamsal benzerlik puanları eklenir.

**Çalıştırma komutu:**

```bash
python final/manuel_puan_ekleme.py
```

* Bu işlem sonrası dosyalar güncellenir ve ortalama anlamsal puanlar hesaplanır.

---

## 4. Model Bazında Ortalama Anlamsal Puan Hesaplama

* Her modelin önerdiği 5 metnin anlamsal puanları üzerinden ortalama puan hesaplanır.

**Çalıştırma komutu:**

```bash
python final/model_ortalama_puani.py
```

* Ortalama puanlar, model karşılaştırması ve değerlendirmesi için kullanılır.

---

## 5. Jaccard Benzerlik Matrisi Hesaplama

* Farklı modellerin sıraladığı ilk 5 metinler arasındaki örtüşme, Jaccard benzerlik ölçütü ile hesaplanır.

**Çalıştırma komutu:**

```bash
python final/jaccard.py
```

* Çıktı dosyası:
  `all_models_jaccard_matrix.csv`

---

## 6. Jaccard Benzerlik Matrisinin Görselleştirilmesi

* Hesaplanan Jaccard benzerlik matrisi, ısı haritası (heatmap) şeklinde görselleştirilir.

**Çalıştırma komutu:**

```bash
python final/jaccard_matris_gorsellestirme.py
```

* Görsel çıktı, rapor için görselleştirme sağlar.

---

# Genel Notlar

* Python 3.7 veya üzeri versiyonlar kullanılmalıdır.
* Gerekli kütüphaneler:
  `pandas`, `numpy`, `gensim`, `matplotlib`, `seaborn`
* Veriler `data/` klasöründe, eğitim modelleri `models/` klasöründedir.
* Final Projesi için yazılan scriptler `final/` klasöründe yer alır.
* Çalıştırma sırasını takip etmek önerilir:

  1. TF-IDF benzerlik → 2. Word2Vec benzerlik → 3. Manuel puan ekleme → 4. Ortalama puan hesaplama → 5. Jaccard matrisi → 6. Görselleştirme

---

# Örnek Çalıştırma

```bash
cd homework-main
python final/tf_model_benzerlik.py
python final/word2_model_benzerlik.py
python final/manuel_puan_ekleme.py
python final/model_ortalama_puani.py
python final/jaccard.py
python final/jaccard_matris_gorsellestirme.py
```


Gülşen Çintuğlu
