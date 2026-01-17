from wordfreq import word_frequency
import math
import pandas as pd
import string

df = pd.read_csv('deceptive-opinion.csv')

total_len = 0
total_len2 = 0
rarity = 0
rarity2 = 0

sentences = df['text'].tolist()
labels = df['deceptive'].tolist() # 각각 리스트가 됨
table = str.maketrans('', '', string.punctuation)

for i in range(len(sentences)):
    if labels[i] == 'truthful':
        total_len = total_len + len(sentences[i].split())
        for w in sentences[i].split():
            if 0 < word_frequency(w.translate(table),'en') < 0.00001:
                rarity = rarity + 1

    else:
        total_len2 = total_len2 + len(sentences[i].split())
        for w in sentences[i].split():
            if 0 < word_frequency(w.translate(table), 'en') < 0.00001:
                rarity2 = rarity2 + 1

print((rarity/total_len),(rarity2/total_len2))
