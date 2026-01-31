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
abs_list = []
abs_list2 = []
con_list = []
con_list2 = []
for i in range(len(sentences)):
    abs_cnt = 0
    con_cnt = 0
    doc = nlp(sentences[i])
    if labels[i] == 'truthful':
        for w in doc:
            if w.pos_ == 'NOUN':
               if get_concreteness(w.text) == 'Abstract':
                   abs_cnt += 1
               else:
                   con_cnt += 1
        abs_list.append(abs_cnt)
        con_list.append(con_cnt)
    else:
        for w in doc:
            if w.pos_ == 'NOUN':
                if get_concreteness(w.text) == 'Abstract':
                    abs_cnt += 1
                else:
                    con_cnt += 1
        abs_list2.append(abs_cnt)
        con_list2.append(con_cnt)

print(f"진짜 리뷰 추상명사 평균: {np.mean(abs_list):.4f}")
print(f"가짜 리뷰 추상명사평균: {np.mean(abs_list2):.4f}")

t_stat_abs, p_val_abs = stats.ttest_ind(abs_list, abs_list2)
eff_size_abs = cohen_d(abs_list, abs_list2)

print(f"t-statistic: {t_stat_abs:.4f}")
print(f"p-value: {p_val_abs:.4e}")
print(f"Cohen's d: {eff_size_abs:.4f}")
print("-" * 40)

print(f"진짜 리뷰 구체명사 평균: {np.mean(con_list):.4f}")
print(f"가짜 리뷰 구체명사 평균: {np.mean(con_list2):.4f}")

t_stat_con, p_val_con = stats.ttest_ind(con_list, con_list2)
eff_size_con = cohen_d(con_list, con_list2)

print(f"t-statistic: {t_stat_con:.4f}")
print(f"p-value: {p_val_con:.4e}")
print(f"Cohen's d: {eff_size_con:.4f}")
