import pandas as pd

input_csv = 'results_word2vec_top5_with_scores.csv'

df = pd.read_csv(input_csv)
df['Semantic_Score'] = pd.to_numeric(df['Semantic_Score'], errors='coerce')
df = df.dropna(subset=['Semantic_Score'])

grouped = df.groupby('Model')

summary_rows = []

for model_name, group in grouped:
    avg_score = group['Semantic_Score'].mean()
    texts = group['Text'].tolist()
    scores = group['Semantic_Score'].tolist()
    
    # Metinleri ve puanları virgülle ayırarak string yap
    texts_str = ', '.join([f"\"{t[:30]}...\"" for t in texts])  # İlk 30 karakterle özet
    scores_str = ', '.join(map(str, scores))
    
    print(f"Model: {model_name}")
    print(f"Ortalama Puan: {avg_score:.2f}")
    print(f"5 Benzer Metin: {texts_str}")
    print(f"Puanlar: {scores_str}")
    print("-" * 80)
    
    summary_rows.append({
        'Model': model_name,
        '5 Benzer Metin': texts_str,
        'Puanlar': scores_str,
        'Ortalama Puan': round(avg_score, 2)
    })

summary_df = pd.DataFrame(summary_rows)
print(summary_df)
