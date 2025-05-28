import pandas as pd
import numpy as np

result_all = pd.read_csv('tfidf_top5_for_input_all_models_with_semanticscore.csv')
df_w2v = pd.read_csv('results_word2vec_top5_with_scores.csv')  # Dosya adını kendi dosyana göre değiştir
df_w2v.rename(columns={'Index': 'Similar_Index', 'Similarity': 'Similarity', 'Text': 'Similar_Text', 'Model': 'Model'}, inplace=True)
all_results = pd.concat([result_all, df_w2v], ignore_index=True)

all_results['Model'] = all_results['Model'].str.replace('.model', '', regex=False)

models = all_results['Model'].unique()

model_top5_sets = {
    model: set(all_results[all_results['Model'] == model]['Similar_Index'].values)
    for model in models
}

n_models = len(models)
jaccard_matrix = np.zeros((n_models, n_models))

for i, m1 in enumerate(models):
    for j, m2 in enumerate(models):
        set1 = model_top5_sets[m1]
        set2 = model_top5_sets[m2]
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        jaccard_matrix[i, j] = intersection / union if union > 0 else 0

jaccard_df = pd.DataFrame(jaccard_matrix, index=models, columns=models)

print("Modeller Arası Jaccard Benzerlik Matrisi:")
print(jaccard_df)
jaccard_df.to_csv('all_models_jaccard_matrix.csv')
