import math
import pandas as pd

df = pd.read_csv('deceptive-opinion.csv')

sentences = df['text'].tolist()
labels = df['deceptive'].tolist() # 각각 리스트가 됨

pronoun = ['i', 'my', 'me', 'mine']
def pronounrate():
    updated_pronoun = 0
    updated_pronoun2 = 0
    a = 0
    b = 0
    for i in range(len(sentences)):
        if labels[i] == 'truthful':
            a = a + 1
            cnt = 0
            clean_sentence = sentences[i].lower()
            for w in clean_sentence.split():
                if w in pronoun:
                    cnt += 1
            pronoun_rate = cnt / len(sentences[i].split())
            updated_pronoun += pronoun_rate
        else:
            b = b + 1
            cnt = 0
            clean_sentence = sentences[i].lower()
            for w in clean_sentence.split():
                if w in pronoun:
                    cnt += 1
            pronoun_rate = cnt / len(sentences[i].split())
            updated_pronoun2 += pronoun_rate
    return updated_pronoun/a, updated_pronoun2/b

avg1, avg2 = pronounrate()
print(avg1,avg2)





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
#nltk.download('wordnet')
#nltk.download('omw-1.4')
#from nltk.corpus import wordnet as wn



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

length = 0

pos_list = []
pos_list2 = []
score = 0
for i in range(len(sentences)):
    cnt = 0
    doc = nlp(sentences[i])
    if labels[i] == 'truthful':
        length = sum(1 for w in doc if w.pos_ != 'PUNCT')
        first_person = ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']
        cnt = sum(1 for w in doc if w.text.lower() in first_person and w.pos_ == 'PRON')
        pos_list.append(cnt/length)
    else:
        length = sum(1 for w in doc if w.pos_ != 'PUNCT')
        first_person = ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']
        cnt = sum(1 for w in doc if w.text.lower() in first_person and w.pos_ == 'PRON')
        pos_list2.append(cnt / length)

print(np.mean(pos_list), np.mean(pos_list2))
t_stat, p_val = stats.ttest_ind(pos_list, pos_list2)
eff_size = cohen_d(pos_list, pos_list2)

print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_val:.4e}")
print(f"Cohen's d: {eff_size:.4f}")
