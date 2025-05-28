import pandas as pd
import numpy as np
from numpy.linalg import norm

def cosine_similarity(vec1, vec2):
    if norm(vec1) == 0 or norm(vec2) == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

def find_top5_for_input(tfidf_df, input_index):
    vectors = tfidf_df.values
    input_vec = vectors[input_index]
    sims = []
    n = len(vectors)
    for j in range(n):
        if j == input_index:
            continue
        sim = cosine_similarity(input_vec, vectors[j])
        sims.append((j, sim))
    sims.sort(key=lambda x: x[1], reverse=True)
    top5 = sims[:5]
    results = []
    for idx, score in top5:
        results.append({
            'Input_Index': input_index,
            'Similar_Index': idx,
            'Similarity': score
        })
    return pd.DataFrame(results)

lemmatized_file = 'data/processveri/tfidf_lemmatized.csv'
stemmed_file = 'data/processveri/tfidf_stemmed.csv'

df_lemmatized = pd.read_csv(lemmatized_file)
df_stemmed = pd.read_csv(stemmed_file)

df = pd.read_csv('data/processveri/zara.csv')

# Örnek metin index'i
input_index = 5
input_text = df.loc[input_index, 'description']
print(f"Seçilen Girdi Metni (index {input_index}):\n{input_text}\n")

# TF-IDF için top5 benzer metinler
result_lemmatized = find_top5_for_input(df_lemmatized, input_index)
result_stemmed = find_top5_for_input(df_stemmed, input_index)

result_lemmatized['Similar_Text'] = result_lemmatized['Similar_Index'].apply(lambda idx: df.loc[idx, 'description'])
result_stemmed['Similar_Text'] = result_stemmed['Similar_Index'].apply(lambda idx: df.loc[idx, 'description'])
result_lemmatized['Model'] = 'TF-IDF Lemmatized'
result_stemmed['Model'] = 'TF-IDF Stemmed'
result_all = pd.concat([result_lemmatized, result_stemmed], ignore_index=True)

semantic_scores = {
    ('TF-IDF Lemmatized', 67): 5,
    ('TF-IDF Lemmatized', 179): 4,
    ('TF-IDF Lemmatized', 155): 4,
    ('TF-IDF Lemmatized', 40): 4,
    ('TF-IDF Lemmatized', 60): 3,
    ('TF-IDF Stemmed', 67): 5,
    ('TF-IDF Stemmed', 179): 4,
    ('TF-IDF Stemmed', 359): 2,
    ('TF-IDF Stemmed', 155): 4,
    ('TF-IDF Stemmed', 40): 4,
}

def get_semantic_score(row):
    return semantic_scores.get((row['Model'], row['Similar_Index']), np.nan)

result_all['Semantic_Score'] = result_all.apply(get_semantic_score, axis=1)
cols = ['Model', 'Input_Index', 'Similar_Index', 'Similarity', 'Similar_Text', 'Semantic_Score']
result_all = result_all[cols]

for model_name, group in result_all.groupby('Model'):
    print(f"\n{model_name} En Benzer 5 Metin:")
    display_df = group.copy()
    display_df.index = range(1, len(display_df) + 1)
    print(display_df.to_string(index=True))

result_all.to_csv('tfidf_top5_for_input_all_models_with_semanticscore.csv', index=False)

avg_scores = result_all.groupby('Model')['Semantic_Score'].mean()
print("\nModel Bazında Ortalama Anlamsal Benzerlik Puanları:")
print(avg_scores)