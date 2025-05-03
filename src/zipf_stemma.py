import pandas as pd
import ast
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

# Veriyi oku
df = pd.read_csv("data/processveri/stemmed_data.csv")

# Toplam satır sayısı
total_rows = len(df)

# 'stemmed_tokens' sütunundaki string ifadeleri listeye dönüştür
df['stemmed_tokens'] = df['stemmed_tokens'].apply(ast.literal_eval)

# Tüm kelimeleri birleştir
all_stemmed_tokens = [token for tokens in df['stemmed_tokens'] for token in tokens]

# Toplam kelime sayısı ve benzersiz kelime sayısı
total_tokens = len(all_stemmed_tokens)
unique_tokens = len(set(all_stemmed_tokens))

# Kelime frekanslarını hesapla
token_counts = Counter(all_stemmed_tokens)

# En sık 10 kelimeyi göster (isteğe bağlı)
print("\nEn sık geçen 10 kelime: Stemma")
for word, freq in token_counts.most_common(10):
    print(f"{word}: {freq}")

# Zipf grafiği için veri hazırla
frequencies = sorted(token_counts.values(), reverse=True)
ranks = np.arange(1, len(frequencies) + 1)

# Özet bilgileri yazdır
print("\n--- Veri Özeti ---")
print(f"Toplam satır sayısı       : {total_rows}")
print(f"Toplam kelime sayısı      : {total_tokens}")
print(f"Benzersiz kelime sayısı   : {unique_tokens}")

# Zipf log-log grafiği
plt.figure(figsize=(10, 6))
plt.loglog(ranks, frequencies, marker="o", linestyle='None', markersize=5, color='darkblue')
plt.title("Zipf Yasası - Stemmed Veriler")
plt.xlabel("Kelime Sırası (Rank)")
plt.ylabel("Frekans (Frequency)")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.show()
