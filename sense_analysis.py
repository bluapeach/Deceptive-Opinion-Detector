from wordfreq import word_frequency
import math
import pandas as pd
import string
from scipy import stats
import numpy as np
import gensim.downloader as api

model = api.load("glove-wiki-gigaword-100")
sensory = {
    'visual': ['bright', 'dark', 'colorful', 'shining', 'vivid', 'visual'],
    'auditory': ['loud', 'quiet', 'noisy', 'echoing', 'silent', 'sound'],
    'olfactory': ['stink', 'fragrant', 'scented', 'musty', 'aroma', 'smell'],
    'tactile': ['soft', 'rough', 'cold', 'hot', 'humid', 'sticky', 'touch'],
    'gustatory': ['sweet', 'salty', 'bitter', 'sour', 'delicious', 'taste']
}
df = pd.read_csv('C:/Users/bsj32/jupyter/deceptive-opinion.csv')

sentences = df['text'].tolist()
labels = df['deceptive'].tolist() # 각각 리스트가 됨
table = str.maketrans('', '', string.punctuation)
allsense = [word for words in sensory.values() for word in words]
rate = 0
rate2 = 0
wordtot = 0
wordtot2 = 0
#sense_list = []
#sense_list2 = []
for i in range(len(sentences)):
    if labels[i] == 'truthful':
        clean_words = sentences[i].translate(table).lower().split()
        for w in clean_words:
            if w in model.key_to_index:
                wordtot = wordtot + 1
                if max([model.similarity(w,word) for word in allsense]) >= 0.6:
                    rate = rate + 1
        #sense_list.append(rate) = rate / len(sentences[i].translate(table).split())   
    else:
        clean_words = sentences[i].translate(table).lower().split()
        for w in clean_words:
            if w in model.key_to_index:
                wordtot2 = wordtot2 + 1
                if max([model.similarity(w,word) for word in allsense]) >= 0.6:
                    rate2 = rate2 + 1
        #sense_list2[i] = rate2 / len(sentences[i].translate(table).split())
print(rate/wordtot,rate2/wordtot2)
