import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

jaccard_df = pd.read_csv('all_models_jaccard_matrix.csv', index_col=0)

plt.figure(figsize=(14, 12))

# Heatmap çiz (annot=True ile sayıları göster)
sns.heatmap(jaccard_df, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'Jaccard Similarity'})
plt.title('Model Jaccard Benzerlik Matrisi (Heatmap)', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
