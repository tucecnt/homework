import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
 
df = pd.read_csv("data/processveri/zara.csv")  

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if pd.isnull(text):  
        return [], []  
    sentences = sent_tokenize(text)  # Cümlelere ayır
    all_lemmatized_tokens = []
    all_stemmed_tokens = []
    for sentence in sentences:
        tokens = word_tokenize(sentence)  # Kelimelere ayır
        filtered_tokens = [
            token.lower() for token in tokens if token.isalpha() and token.lower() not in stop_words
        ]
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]  # Lemmatize 
        stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]  # Stemleme
        all_lemmatized_tokens.extend(lemmatized_tokens)
        all_stemmed_tokens.extend(stemmed_tokens)
    return all_lemmatized_tokens, all_stemmed_tokens

# Veri setindeki her açıklamaya uygula
df['lemmatized_tokens'], df['stemmed_tokens'] = zip(*df['description'].apply(preprocess_text))
print(df[['description', 'lemmatized_tokens', 'stemmed_tokens']].head())

# İlk 5 cümle
for i in range(5):
    print(f"Cümle {i+1} - Lemmatized: {df['lemmatized_tokens'][i]}")
    print(" \n")
    print(f"Cümle {i+1} - Stemmed: {df['stemmed_tokens'][i]}")
    print(" \n")
    
df['lemmatized_tokens'], df['stemmed_tokens'] = zip(*df['description'].apply(preprocess_text))

df[['lemmatized_tokens']].to_csv("data/processveri/lemmatized_data.csv", index=False)
df[['stemmed_tokens']].to_csv("data/processveri/stemmed_data.csv", index=False)

