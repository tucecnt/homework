import pandas as pd
from gensim.models import Word2Vec
import os

if not os.path.exists('models'):
    os.makedirs('models')

# Veriyi oku
df_lemmatized = pd.read_csv("data/processveri/lemmatized_data.csv")
df_stemmed = pd.read_csv("data/processveri/stemmed_data.csv")

lemmatized_texts = df_lemmatized['lemmatized_tokens'].apply(lambda x: eval(x))
stemmed_texts = df_stemmed['stemmed_tokens'].apply(lambda x: eval(x))

parameters = [
    {'model_type': 'cbow', 'window': 2, 'vector_size': 100}, 
    {'model_type': 'skipgram', 'window': 2, 'vector_size': 100}, 
    {'model_type': 'cbow', 'window': 4, 'vector_size': 100}, 
    {'model_type': 'skipgram', 'window': 4, 'vector_size': 100}, 
    {'model_type': 'cbow', 'window': 2, 'vector_size': 300}, 
    {'model_type': 'skipgram', 'window': 2, 'vector_size': 300}, 
    {'model_type': 'cbow', 'window': 4, 'vector_size': 300}, 
    {'model_type': 'skipgram', 'window': 4, 'vector_size': 300}
]
all_results = []

# Model eğitimi ve benzer kelime analizi
for data_type, texts in [('lemmatized', lemmatized_texts), ('stemmed', stemmed_texts)]:
    for param in parameters:
        model_type = param['model_type']
        window = param['window']
        vector_size = param['vector_size']
        
        model = Word2Vec(
            sentences=texts, 
            vector_size=vector_size, 
            window=window, 
            sg=1 if model_type == 'skipgram' else 0,
            min_count=1, 
            workers=4
        )

        model_filename = f"models/word2vec_{data_type}_{model_type}_win{window}_dim{vector_size}.model"
        model.save(model_filename)
        print(f"{model_filename} kaydedildi.")

        word = "shoe"
        try:
            similar_words = model.wv.most_similar(word, topn=5)
            print(f"\nModel: {model_filename}")
            print(f"{word} kelimesine en yakın 5 kelime:")
            for sw in similar_words:
                print(sw)

            for sw in similar_words:
                all_results.append({
                    'Model': model_filename,
                    'Kelime': word,
                    'Benzer Kelime': sw[0],
                    'Benzerlik': round(sw[1], 4)
                })

        except KeyError:
            print(f"Kelime '{word}' modelde bulunamadı.")

result_df = pd.DataFrame(all_results)
result_df.to_csv("report/similar_words_report.csv", index=False)
print("\nTüm sonuçlar 'similar_words_report.csv' dosyasına kaydedildi.")
