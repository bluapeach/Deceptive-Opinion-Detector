from wordfreq import word_frequency
from empath import Empath
import pandas as pd
import string
import numpy as np
from textblob import TextBlob
from scipy import stats

df = pd.read_csv('deceptive-opinion.csv')

def cohen_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std
    
sentences = df['text'].tolist()
labels = df['deceptive'].tolist()

table = str.maketrans('', '', string.punctuation)

emolist = []
emolist2 = []
score = 0
for i in range(len(sentences)):
    words = sentences[i].translate(table).lower()
    blob = TextBlob(words)
    score = blob.sentiment.subjectivity
    if labels[i] == 'truthful':
        emolist.append(score)
    else:
        emolist2.append(score)

mean1 = np.mean(emolist)
mean2 = np.mean(emolist2)
t_stat, p_val = stats.ttest_ind(emolist, emolist2)
eff_size = cohen_d(emolist, emolist2)

print(f"평균 (Truthful): {mean1:.4f}, 평균 (Deceptive): {mean2:.4f}")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_val:.4e}")
print(f"Cohen's d: {eff_size:.4f}")
