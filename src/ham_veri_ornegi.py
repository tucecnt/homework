import pandas as pd

df = pd.read_csv("data/hamveri/store_zara.csv")

# İlk birkaç satır
print(df.head())

# Veri kümesinin genel yapısı
print(df.info())

# Eksik veri kontrolü
print(df.isnull().sum())

# "description" sütununun ilk 5 satırı
print(df["description"].head())

