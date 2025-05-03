import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import ast
import numpy as np

def get_token_stats(csv_file, column_name):
    df = pd.read_csv(csv_file)

    all_tokens = []
    for tokens_str in df[column_name]:
        tokens = ast.literal_eval(tokens_str)
        all_tokens.extend(tokens)

    word_freq = Counter(all_tokens)
    total_word_count = len(all_tokens)
    unique_word_count = len(word_freq)

    return df, word_freq, total_word_count, unique_word_count

def print_top_words(title, word_freq, df, total_word_count, unique_word_count):
    print(f"\n--- {title.upper()} VERİ ANALİZİ ---")
    print("\nEn sık geçen 10 kelime:")
    for word, freq in word_freq.most_common(10):
        print(f"{word}: {freq}")

    print("\n--- Veri Özeti ---")
    print(f"Toplam satır sayısı       : {len(df)}")
    print(f"Toplam kelime sayısı      : {total_word_count}")
    print(f"Benzersiz kelime sayısı   : {unique_word_count}")

# Verileri al
df_stemmed, freq_stemmed, total_stemmed, unique_stemmed = get_token_stats("data/processveri/stemmed_data.csv", "stemmed_tokens")
df_lemmatized, freq_lemmatized, total_lemmatized, unique_lemmatized = get_token_stats("data/processveri/lemmatized_data.csv", "lemmatized_tokens")

# Bilgileri yazdır
print_top_words("Stemmed", freq_stemmed, df_stemmed, total_stemmed, unique_stemmed)
print_top_words("Lemmatized", freq_lemmatized, df_lemmatized, total_lemmatized, unique_lemmatized)

# Grafik verileri hazırla
freqs_stemmed = np.array(sorted(freq_stemmed.values(), reverse=True))
ranks_stemmed = np.arange(1, len(freqs_stemmed) + 1)

freqs_lemmatized = np.array(sorted(freq_lemmatized.values(), reverse=True))
ranks_lemmatized = np.arange(1, len(freqs_lemmatized) + 1)

# Subplot oluştur
plt.figure(figsize=(14, 6))

# Stemmed plot
plt.subplot(1, 2, 1)
plt.loglog(ranks_stemmed, freqs_stemmed, marker='o', linestyle='none', color='blue')
plt.title("Zipf Yasası - Stemmed")
plt.xlabel("Kelime Sırası (log)")
plt.ylabel("Frekans (log)")
plt.grid(True)

# Lemmatized plot
plt.subplot(1, 2, 2)
plt.loglog(ranks_lemmatized, freqs_lemmatized, marker='o', linestyle='none', color='green')
plt.title("Zipf Yasası - Lemmatized")
plt.xlabel("Kelime Sırası (log)")
plt.ylabel("Frekans (log)")
plt.grid(True)

plt.tight_layout()
plt.show()
