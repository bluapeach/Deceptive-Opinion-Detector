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


def get_concreteness(word):
    # 단어의 명사뜻을 모두 가져옴
    synsets = wn.synsets(word, pos=wn.NOUN)# 리스트 형태로 나옴

    if not synsets:
        return None  # 데이터에 없는 단어는 패스

    # 가장 대표적인 의미의 카테고리를 가져옴 단어마다, 의미마다 카테고리있음
    category = synsets[0].lexname()

    # 추상적 명사들의 대표 카테고리 리스트
    categories = [
        'noun.attribute', 'noun.feeling', 'noun.cognition',
        'noun.state', 'noun.communication', 'noun.relation', 'noun.motive'
    ]

    if category in categories:
        return "Abstract"
    else:
        return "Concrete"

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
    if labels[i] == 'truthful':
        score = [TextBlob(w.text).sentiment.subjectivity for w in doc if w.pos_ == 'ADJ']#이건 ADJ대신 측정하고싶은 품사 집어넣어서 돌리면 됨
        if score:
            pos_list.append(np.mean(score))
        else:
            pos_list.append(0)
    else:
        score = [TextBlob(w.text).sentiment.subjectivity for w in doc if w.pos_ == 'ADJ']
        if score:
            pos_list2.append(np.mean(score))
        else:
            pos_list2.append(0)

print(np.mean(pos_list), np.mean(pos_list2))
t_stat, p_val = stats.ttest_ind(pos_list, pos_list2)
eff_size = cohen_d(pos_list, pos_list2)

print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_val:.4e}")
print(f"Cohen's d: {eff_size:.4f}")
