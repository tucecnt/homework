import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

df_lemmatized = pd.read_csv("data/processveri/lemmatized_data.csv")  
df_stemmed = pd.read_csv("data/processveri/stemmed_data.csv")  

lemmatized_texts = df_lemmatized['lemmatized_tokens'].apply(lambda x: ' '.join(eval(x)))
stemmed_texts = df_stemmed['stemmed_tokens'].apply(lambda x: ' '.join(eval(x)))
print(lemmatized_texts[:3])

tfidf_vectorizer = TfidfVectorizer()

# Lemmatized veri TF-IDF 
tfidf_lemmatized = tfidf_vectorizer.fit_transform(lemmatized_texts)
df_tfidf_lemmatized = pd.DataFrame(tfidf_lemmatized.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
print(df_tfidf_lemmatized.head())

# Stemmed veri TF-IDF 
tfidf_stemmed = tfidf_vectorizer.fit_transform(stemmed_texts)
df_tfidf_stemmed = pd.DataFrame(tfidf_stemmed.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Sonuçları CSV olarak kaydet
df_tfidf_lemmatized.to_csv("data/processveri/tfidf_lemmatized.csv", index=False)
df_tfidf_stemmed.to_csv("data/processveri/tfidf_stemmed.csv", index=False)

print("TF-IDF işlemi tamamlandı ve dosyalar kaydedildi.")
first_sentence_vector = df_tfidf_lemmatized.iloc[0]

# Skorlara göre sırala (yüksekten düşüğe) ve ilk 5 kelimeyi al
top_5_words = first_sentence_vector.sort_values(ascending=False).head(5)

# Sonuçları yazdır
print("İlk cümlede en yüksek TF-IDF skoruna sahip 5 kelime:")
print(top_5_words)