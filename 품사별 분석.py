import spacy
from wordfreq import word_frequency
import math
import pandas as pd
import string
from scipy import stats
import numpy as np
import gensim.downloader as api
from textblob import TextBlob
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import wordnet as wn

df = pd.read_csv('C:/Users/bsj32/jupyter/deceptive-opinion.csv')
def cohen_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    # 합산 표준편차(Pooled Standard Deviation) 계산
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std

sentences = df['text'].tolist()
labels = df['deceptive'].tolist()  # 각각 리스트가 됨
table = str.maketrans('', '', string.punctuation)

nlp = spacy.load("en_core_web_sm")

pos_list = []
pos_list2 = []
score = 0
for i in range(len(sentences)):
    cnt = 0
    doc = nlp(sentences[i])
    length = sum(1 for w in doc if w.pos_ != 'PUNCT')
    
    if labels[i] == 'truthful':
        for w in doc:
            if w.pos_ == 'ADJ':
                cnt  = cnt + 1
            #이건 ADJ대신 측정하고싶은 품사 집어넣어서 돌리면 됨
        pos_list.append(cnt/length)
    else:
        for w in doc:
            if w.pos_ == 'ADJ':
                cnt  = cnt + 1
            #이건 ADJ대신 측정하고싶은 품사 집어넣어서 돌리면 됨
        pos_list2.append(cnt/length)

print(np.mean(pos_list), np.mean(pos_list2))
t_stat, p_val = stats.ttest_ind(pos_list, pos_list2)
eff_size = cohen_d(pos_list, pos_list2)

print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_val:.4e}")
print(f"Cohen's d: {eff_size:.4f}")
