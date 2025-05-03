import pandas as pd
import ast
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

# Veriyi oku
df = pd.read_csv("data/processveri/lemmatized_data.csv")

# Toplam satır sayısı
total_rows = len(df)

# String olarak kaydedilmiş listeyi gerçek listeye çevir
df['lemmatized_tokens'] = df['lemmatized_tokens'].apply(ast.literal_eval)

# Tüm lemmatize edilmiş kelimeleri birleştir
all_lemmatized_tokens = [token for tokens in df['lemmatized_tokens'] for token in tokens]

# Toplam ve benzersiz kelime sayısı
total_tokens = len(all_lemmatized_tokens)
unique_tokens = len(set(all_lemmatized_tokens))

# Kelime frekanslarını hesapla
token_counts = Counter(all_lemmatized_tokens)

# En sık geçen ilk 10 kelimeyi yazdır
print("\nEn sık geçen 10 kelime: Lemma")
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

# Zipf yasası log-log grafiği
plt.figure(figsize=(10, 6))
plt.loglog(ranks, frequencies, marker="o", linestyle='None', markersize=5, color='darkgreen')
plt.title("Zipf Yasası - Lemmatized Veriler")
plt.xlabel("Kelime Sırası (Rank)")
plt.ylabel("Frekans (Frequency)")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.show()
