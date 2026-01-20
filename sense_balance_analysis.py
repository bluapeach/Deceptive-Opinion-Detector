from wordfreq import word_frequency
import math
import pandas as pd
import string
from scipy import stats
import numpy as np
import gensim.downloader as api

model = api.load("glove-wiki-gigaword-100")
sensory = {
    'visual': [
        'bright', 'dark', 'colorful', 'shining', 'vivid', 'visual', 'view', 'lobby',
        'clean', 'dirty', 'decor', 'modern', 'old', 'lighting', 'beautiful', 'shabby',
        'spacious', 'cramped', 'dusty', 'spotless', 'elegant', 'panorama', 'scenery'
    ],
    'auditory': [
        'loud', 'quiet', 'noisy', 'echoing', 'silent', 'sound', 'walls', 'traffic',
        'voices', 'street', 'night', 'floor', 'music', 'alarm', 'humming', 'silent',
        'thumping', 'whisper', 'screech', 'clanking', 'ventilation', 'ac'
    ],
    'olfactory': [
        'stink', 'fragrant', 'scented', 'musty', 'aroma', 'smell', 'odor', 'perfume',
        'freshener', 'mildew', 'smoke', 'cigarette', 'damp', 'floral', 'bleach', 'sewage',
        'reeking', 'stale', 'fresh', 'unpleasant'
    ],
    'tactile': [
        'soft', 'rough', 'cold', 'hot', 'humid', 'sticky', 'touch', 'bed', 'pillows',
        'sheet', 'shower', 'water', 'temperature', 'hard', 'fluffy', 'stiff', 'pressure',
        'lukewarm', 'freezing', 'cozy', 'smooth', 'prickly'
    ],
    'gustatory': [
        'sweet', 'salty', 'bitter', 'sour', 'delicious', 'taste', 'flavor', 'yummy',
        'breakfast', 'buffet', 'coffee', 'tea', 'spicy', 'bland', 'savory', 'refreshing',
        'greasy', 'stale', 'cooked', 'menu'
    ]
}
df = pd.read_csv('C:/Users/bsj32/jupyter/deceptive-opinion.csv')
def cohen_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std
def get_entropy(counts):
    probs = np.array(counts) / (sum(counts) if sum(counts) > 0 else 1)
    return -np.sum([p * np.log2(p) for p in probs if p > 0])

sentences = df['text'].tolist()
labels = df['deceptive'].tolist() # 각각 리스트가 됨
table = str.maketrans('', '', string.punctuation)
allsense = [word for words in sensory.values() for word in words]#감각 딕셔너리에 있는 모든 단어의 리스트
rate = 0
rate2 = 0
wordtot = 0
wordtot2 = 0
sense_list = []
entropy_list = []
sense_list2 = []
entropy_list2 = []
for i in range(len(sentences)):
    if labels[i] == 'truthful':
        clean_words = sentences[i].translate(table).lower().split()
        sense_list = []
        for key in sensory.keys():
            cnt = 0
            for w in clean_words:
                for word in sensory[key]:
                    try:
                        if model.similarity(word, w) > 0.6:
                            cnt += 1
                            break
                    except KeyError:
                        continue
            sense_list.append(cnt)
        if sum(sense_list) > 2:
            entropy_list.append(get_entropy(sense_list))
    else:
        clean_words = sentences[i].translate(table).lower().split()
        sense_list2 = []
        for key in sensory.keys():
            cnt = 0 #감각별 해당단어 갯수 카운트
            for w in clean_words:
                for word in sensory[key]:
                    try:
                        if model.similarity(word, w) > 0.6:
                            cnt += 1
                            break
                    except KeyError:
                        continue
            sense_list2.append(cnt)
        if sum(sense_list2) > 2:
            entropy_list2.append(get_entropy(sense_list2))

print(np.mean(entropy_list), np.mean(entropy_list2))

t_stat, p_val = stats.ttest_ind(entropy_list, entropy_list2)
eff_size = cohen_d(entropy_list, entropy_list2)

print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_val:.4e}")
print(f"Cohen's d: {eff_size:.4f}")
