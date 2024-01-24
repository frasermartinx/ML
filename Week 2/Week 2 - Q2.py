import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd

### a) - cleaning ###

#we need to do some data cleaning
iris = datasets.load_iris()
iris_final = pd.DataFrame(iris.data, columns = iris.feature_names)
#now need to make the target column, but we are doing binary, so compress 0 and 1 to 0 and 2 to 1
#print(iris.target)
iris.target = np.array([1 if i == 2 else 0 for i in iris.target])
#print(iris.target)
#now map the names:
iris_final["class"] = iris.target
target_names = {0: "Not Virginica", 1: "Virginica"}
iris_final["class_name"] = iris_final["class"].map(target_names)
#print(iris_final)

#now we are ready to commence

### b)- training test ###
from sklearn.model_selection import train_test_split

train, test = train_test_split(iris_final, test_size = 0.3, random_state = 2)

### c) - LDA ###

#for LDA we assume that the class conditional densities are gaussian with different means
#but the same covariance matrix

#we will assume equal class probabilities

#as for the mean for each class, we will use the training set and use prior mean

#find means:


train_0 = train.loc[train["class"] == 0]
train_1 = train.loc[train["class"] == 1]
train_0_x = train_0.iloc[:,0:4]
train_1_x = train_1.iloc[:,0:4]
mean_0 = np.array(np.mean(train_0_x,axis = 0))
mean_1 = np.array(np.mean(train_1_x,axis = 0))

#we will use the covariance matrix for both classes:
train_x = train.iloc[:,0:4]
cov = np.cov(train_x)
print(cov)



