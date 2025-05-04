
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


Gülşen Çintuğlu
