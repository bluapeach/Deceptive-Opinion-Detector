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
