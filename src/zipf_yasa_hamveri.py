import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

# Veriyi yükle
df = pd.read_csv("data/hamveri/store_zara.csv")

# Stopword listesini al
stop_words = set(stopwords.words('english'))

# Metni işlemek için fonksiyon
def preprocess_text(text):
    if pd.isnull(text):
        return []  # NaN varsa, boş liste döndür
    tokens = word_tokenize(text.lower())  # Kelimelere ayır ve küçük harfe çevir
    filtered_tokens = [
        token for token in tokens if token.isalpha() and token not in stop_words
    ]  # Sadece alfabetik kelimeleri al
    return filtered_tokens

# Veri setindeki her açıklamaya uygula
df['tokens'] = df['description'].apply(preprocess_text)

# Tüm kelimeleri birleştir
all_tokens = [token for sublist in df['tokens'] for token in sublist]

# Kelimelerin frekansını hesapla
fdist = FreqDist(all_tokens)

# Kelimeleri sıklıklarına göre sırala
sorted_frequencies = fdist.most_common()

# X ve Y eksenlerini oluştur
ranks = np.arange(1, len(sorted_frequencies) + 1)
frequencies = np.array([freq for _, freq in sorted_frequencies])

# Grafik çizimi (Log-log çizimi için plt.loglog)
plt.figure(figsize=(10, 6))
plt.loglog(ranks, frequencies, 'bo-', markersize=5)
plt.title("Zipf Yasası Log Log Grafiği Ham Veri", fontsize=14)
plt.xlabel('Rank', fontsize=12)
plt.ylabel('Frekans', fontsize=12)
plt.grid(True)

plt.show()
