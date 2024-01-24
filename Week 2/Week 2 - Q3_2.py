import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
k = 0
# read in the dataset
mnist = pd.read_csv("mnist.csv", header=None)
mnist = mnist.to_numpy()

# split into X and y
X = mnist[:, 1:]
y = mnist[:, 0]

# keep 4s and 9s only - 12,665 examples in total
included_examples = np.isin(y, [4, 9])
X = X[included_examples]
y = y[included_examples]

# convert to [0,1] from grayscale
X = X / 255.0

### LOGISTIC REGRESSION ###


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=0)

model = LogisticRegression(penalty = "l2", solver = "liblinear").fit(X_train, y_train)
pred = model.predict(X_test)
print(sum(pred == y_test)/len(y_test))
