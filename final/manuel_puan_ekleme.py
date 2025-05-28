import pandas as pd

input_path = 'results_word2vec_top5.csv'  
output_path = 'results_word2vec_top5_with_scores.csv'
df = pd.read_csv(input_path)
df['Semantic_Score'] = ''
df.to_csv(output_path, index=False)

