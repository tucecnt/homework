import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from numpy.linalg import norm
from numpy import dot
import os

# Cosine similarity fonksiyonu
def cosine_similarity(vec1, vec2):
    if norm(vec1) == 0 or norm(vec2) == 0:
        return 0.0
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))

# Ortalama vektör hesaplama
def get_average_vector(text, model):
    words = text.lower().split()
    vectors = [model.wv[w] for w in words if w in model.wv]
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

import csv

def main():
    df = pd.read_csv('data/processveri/zara.csv')
    input_text = df.loc[5, 'description']
    print(f"Seçilen Girdi Metni: {input_text}\n")

    models_folder = 'models'
    model_files = [f for f in os.listdir(models_folder) if f.endswith('.model')]

    with open('results_word2vec_top5.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Model', 'Index', 'Similarity', 'Text'])

        for model_file in model_files:
            print(f"Model: {model_file}")
            model_path = os.path.join(models_folder, model_file)
            model = Word2Vec.load(model_path)

            input_vec = get_average_vector(input_text, model)

            similarities = []
            for idx, desc in enumerate(df['description']):
                vec = get_average_vector(desc, model)
                sim = cosine_similarity(input_vec, vec)
                similarities.append((idx, sim, df.loc[idx, 'description']))

            similarities.sort(key=lambda x: x[1], reverse=True)
            top5 = similarities[1:6]

            for idx, score, text in top5:
                writer.writerow([model_file, idx, score, text])
                print(f"Index: {idx}, Benzerlik: {score:.4f}, Metin: {text}")
            print('-' * 80)

if __name__ == "__main__":
    main()
