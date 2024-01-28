import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd
import scipy

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

train, test = train_test_split(iris_final, test_size = 0.3, random_state = 3)

### c) - LDA ###


#for LDA we assume that the class conditional densities are gaussian with different means
#but the same covariance matrix

#empirical prior probabilities:
train_0 = train.loc[train["class"] == 0]
train_1 = train.loc[train["class"] == 1]

pi0 = train_0.shape[0]/(train_0.shape[0] + train_1.shape[0])
pi1 = 1 - pi0
#as for the mean for each class, we will use the training set and use prior mean
print(pi0)
print(pi1)
#find means:



train_0_x = train_0.iloc[:,0:4]
train_1_x = train_1.iloc[:,0:4]
mean_0 = np.array(np.mean(train_0_x,axis = 0))
mean_1 = np.array(np.mean(train_1_x,axis = 0))

#we will use the covariance matrix for both classes:
train_x = train.iloc[:,0:4].to_numpy()
cov = np.cov(train_x.T)



#now we can do the actual classifier

def LDA_discriminant(x,mu0,mu1,sigma,pi0,pi1):
    d = np.log(pi0) - np.log(pi1) + x.T @ np.linalg.inv(sigma) @ mu0 - x.T @ np.linalg.inv(sigma) @ mu1 - 0.5* mu0.T @ np.linalg.inv(sigma) @ mu0 + 0.5 * mu1.T @ np.linalg.inv(sigma) @ mu1
    return 0 if d > 0 else 1
test_x = test.iloc[:,0:4].to_numpy()

#test
test_prediction = np.array([LDA_discriminant(i,mean_0,mean_1, cov, pi0, pi1) for i in test_x])
test_actual = test["class"].to_numpy()
err_test = sum(test_actual == test_prediction) / len(test_prediction)
#train:
train_prediction = np.array([LDA_discriminant(i,mean_0,mean_1, cov, pi0, pi1) for i in train_x])
train_actual = train["class"].to_numpy()
err_train = sum(train_actual == train_prediction) / len(train_prediction)



### d) - confusion matrix and ROC ###
def confusion_matrix(pred:np.array, actual: np.array):
    TP = sum((pred == 1) & (actual == 1))
    TN = sum((pred == 0) & (actual == 0))
    FP = sum((pred == 1) & (actual == 0))
    FN = sum((pred == 0) & (actual == 1))
    return np.array([[TP,FP],[FN,TN]])

conf_train = confusion_matrix(train_prediction, train_actual)
conf_test = confusion_matrix(test_prediction, test_actual)

print("Accuracy for training set:")
print(err_train)

print("Confusion matrix for training set:")
print(conf_train)

print("Accuracy for test set:")
print(err_test)

print("Confusion matrix for test set:")
print(conf_test)


#ROC

#we didnt have to do discriminant above, could have just done this and set boundary to 0.5 without the transformation
def posterior(x,mu0,mu1,cov,pi0,pi1):
    p =  pi1 * scipy.stats.multivariate_normal.pdf(x,mu1,cov) / (pi0 * scipy.stats.multivariate_normal.pdf(x,mu0,cov) + pi1 * scipy.stats.multivariate_normal.pdf(x,mu1,cov))
    return p
train_p = [posterior(i,mean_0, mean_1, cov, pi0, pi1) for i in train_x]
test_p = [posterior(i,mean_0, mean_1, cov, pi0, pi1) for i in test_x]

from sklearn import metrics
fpr_train, tpr_train, _ = metrics.roc_curve(train_actual,  train_p)
plt.plot(fpr_train, tpr_train)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
fpr_test, tpr_test, _ = metrics.roc_curve(test_actual,  test_p)
plt.plot(fpr_test, tpr_test)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


#sick