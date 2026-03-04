import pandas as pd
import numpy as np
import sympy as sp
from sympy import Matrix
import math
df = pd.read_csv('deceptive_analysis_matrix.csv')

beta = np.full((7,1),0.5) # 2d
def sigmoid(x):
    return 1/(1 + np.exp(-x))

X = df.iloc[:,1:-1].to_numpy()
y = df[['Is_Deceptive']].to_numpy()

probability = np.full((1600,1),0)
learningrate = 0.001
canary = 0

def update():
    global beta, probability
    z = X @ beta
    probability = sigmoid(z)

    if np.linalg.norm(X.T @ (y - probability)) < 1e-6:
        return True
    else:
        beta = beta + learningrate * (X.T @ (y - probability))
        return False

for k in range(32431):
    if update():
        break
    else:
        continue

