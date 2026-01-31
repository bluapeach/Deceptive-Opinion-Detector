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







from wordfreq import word_frequency
import math
import pandas as pd
import string
from scipy import stats
import numpy as np
import gensim.downloader as api
import spacy


def cohen_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std


nlp = spacy.load("en_core_web_sm")
model = api.load("glove-wiki-gigaword-100")

sensory = {
    'visual': ['bright', 'dark', 'colorful', 'shining', 'vivid', 'visual'],
    'auditory': ['loud', 'quiet', 'noisy', 'echoing', 'silent', 'sound'],
    'olfactory': ['stink', 'fragrant', 'scented', 'musty', 'aroma', 'smell'],
    'tactile': ['soft', 'rough', 'cold', 'hot', 'humid', 'sticky', 'touch'],
    'gustatory': ['sweet', 'salty', 'bitter', 'sour', 'delicious', 'taste']
}
allsense = [word for words in sensory.values() for word in words]


df = pd.read_csv('C:/Users/bsj32/jupyter/deceptive-opinion.csv')

sentences = df['text'].tolist()
labels = df['deceptive'].tolist()
table = str.maketrans('', '', string.punctuation)


sense_list = []  # truthful 용
sense_list2 = []  # deceptive 용

for i in range(len(sentences)):
    rate_cnt = 0
    doc = nlp(sentences[i])
    length = sum(1 for w in doc if w.pos_ != 'PUNCT')

    if length == 0: continue  


    clean_words = sentences[i].translate(table).lower().split()
    for w in clean_words:
        if w in model.key_to_index:
            if max([model.similarity(w, target) for target in allsense]) >= 0.6:
                rate_cnt = rate_cnt + 1

    if labels[i] == 'truthful':
        sense_list.append(rate_cnt / length)
    else:
        sense_list2.append(rate_cnt / length)


t_stat_abs, p_val_abs = stats.ttest_ind(sense_list, sense_list2)
eff_size_abs = cohen_d(sense_list, sense_list2)

print(np.mean(sense_list), np.mean(sense_list2))
print(f"t-statistic: {t_stat_abs:.4f}")
print(f"p-value: {p_val_abs:.4e}")
print(f"Cohen's d: {eff_size_abs:.4f}")
