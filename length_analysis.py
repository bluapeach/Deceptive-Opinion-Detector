import math
import pandas as pd
import string
from scipy import stats
import numpy as np
import gensim.downloader as api

df = pd.read_csv('C:/Users/bsj32/jupyter/deceptive-opinion.csv')
def cohen_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std

sentences = df['text'].tolist()
labels = df['deceptive'].tolist()  # 각각 리스트가 됨
table = str.maketrans('', '', string.punctuation)

length_list = []
length_list2 = []

for i in range(len(sentences)):
    if labels[i] == 'truthful':
        clean_words = sentences[i].translate(table).split()
        length_list.append(len(clean_words))
    else:
        clean_words = sentences[i].translate(table).split()
        length_list2.append(len(clean_words))

print(np.mean(length_list),np.std(length_list),np.mean(length_list2),np.std(length_list2))
t_stat, p_val = stats.ttest_ind(length_list, length_list2)
eff_size = cohen_d(length_list, length_list2)

print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_val:.4e}")
print(f"Cohen's d: {eff_size:.4f}")
