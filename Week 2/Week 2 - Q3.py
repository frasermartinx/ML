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

def make_square(vec):
    image_size = int(np.sqrt(len(vec)))
    tmp = np.reshape(vec, (image_size, image_size))
    return tmp

fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
for i in range(3):
    for j in range(3):
        ax[i, j].imshow(make_square(X[3*i + j]))
        ax[i, j].set_title(f"label: {y[3*i + j]}")

#plt.show()

#change y to be 0 for 4 and 1 for 9
y = np.array([0 if i==4 else 1 for i in y])


import scipy


class binary_LDA:
    def __init__(self,X_train: np.array, y_train:np.array):
        #X should have rows as observations
        #y should be the class vector of 0 and 1
        self.cov = np.cov(X_train.T)
        self.xtrain = X_train
        self.ytrain = y_train
        #find priors:
        self.pi0 = len(np.where(self.ytrain == 0)[0]) / len(self.ytrain)
        self.pi1 = 1 - self.pi0
        #find means:
        self.mu0 = self.find_mean(0)
        self.mu1 = self.find_mean(1)

    def find_mean(self,k):
        X_k = self.xtrain[np.where(self.ytrain == k)[0],:]
        return np.mean(X_k,axis = 0)

    #it is faster to use discriminant as no exp
    #def posterior(self,x):

        #posterior for 1

        #return self.pi1 * scipy.stats.multivariate_normal.pdf(x,self.mu1,self.cov) / (self.pi0 * scipy.stats.multivariate_normal.pdf(x,self.mu0,self.cov) + self.pi1 * scipy.stats.multivariate_normal.pdf(x,self.mu1,self.cov))

    def posterior(self,x):
        d = np.log(self.pi0) - np.log(self.pi1) + x.T @ np.linalg.inv(self.cov) @ self.mu0 - x.T @ np.linalg.inv(self.cov) @ self.mu1 - 0.5 * self.mu0.T @ np.linalg.inv(self.cov) @ self.mu0 + 0.5 * self.mu1.T @ np.linalg.inv(self.cov) @ self.mu1
        return 0 if d > 0 else 1

    def classify(self,X_test):
        pred = np.array([self.posterior(i) for i in X_test])
        return pred

#pred = c.classify(X)
#we remove any variables that have variance of 0:
#indices = [i for i in range(X.shape[1]) if sum(X.T[i]) != 0]
#X = X[:,indices]
#y = y[indices]
c = binary_LDA(X,y)

#print(c.classify(X))

#the problem is that the covariance matrix is giving eigenvalues that are classed as negative even though they are 0

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

diag = np.diagonal(c.cov)
#print(max(diag))
#remove 0 variance ones
fig = plt.figure()
plt.boxplot(diag)
#plt.show()
indices = [i for i in range(X.shape[1]) if diag[i] > np.mean(diag)]
X = X[:,indices]

#classifier = binary_LDA(X,y)
#pred = classifier.classify(X)
#print(sum(pred==y)/len(y))

#now with test train

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=5)
classifier_train = binary_LDA(X_train,y_train)
pred_test = classifier_train.classify(X_test)
print(sum(pred_test==y_test)/len(y_test))
