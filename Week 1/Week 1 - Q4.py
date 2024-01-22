import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("dataQ4.csv")
data = data.to_numpy()
x = data[:,0][:,None]
z = data[:,1][:,None]
z = np.round(z)
y = data[:,2][:,None]
#create the design matrix:
X = np.concatenate([z,np.cos(x),z*np.cos(x),x,x*z,x**2,z*(x**2),x**3,z*(x**3)],axis = 1)

print(X.shape)
#now infer the parameters using ridge regression

def get_MSE_ridge(X,y,lamb):

    #now do LOO-CV
    CV = np.zeros([X.shape[0],1])
    for i in range(X.shape[0]):
        X_train, X_test = np.delete(X,i,axis = 0), X[i,:]
        y_train, y_test = np.delete(y, i, axis=0), y[i, :]
    #solve
        coeffs = np.linalg.solve((X_train.T @ X_train) + lamb * np.eye(X_train.shape[1]), X_train.T @ y_train)
        y_pred_test = X_test @ coeffs
        mse_test = np.mean((y_pred_test - y_test) ** 2)
        CV[i] = mse_test
    CV_mean = np.mean(CV)
    CV_sd = np.std(CV)

    return CV_mean,CV_sd

lamb = np.concatenate(([0], 2 ** np.arange(-10, 0.1, 0.1)))
ridge_mses = [get_MSE_ridge(X, y, l)[0] for l in lamb]
plt.plot(lamb, ridge_mses, label="Test")
plt.legend()
plt.ylabel("Averaged MSE")
plt.xlabel("$\lambda$")
plt.show()

lambda_best = lamb[np.argmin(ridge_mses)]
print("Lambda minimising LOOCV MSE=", lambda_best)

Paras_hat = np.linalg.solve(X.T @ X + lambda_best * np.eye(9), X.T @ y)

# Define the names
names = ["a", "b1", "b2", "c1", "c2", "d1", "d2", "e1", "e2"]
# Print the names and values
for name, value in zip(names, Paras_hat):
    print(f"{name}: {value}")


### We now do prediction ###

xtest = np.arange(-7,7,0.1)
Xalltest1 = np.column_stack([np.ones_like(xtest), np.cos(xtest), np.cos(xtest), xtest, xtest, xtest**2, xtest**2, xtest**3, xtest**3])
Xalltest0 = np.column_stack([np.zeros_like(xtest), np.cos(xtest), np.zeros_like(xtest), xtest, np.zeros_like(xtest), xtest**2, np.zeros_like(xtest), xtest**3, np.zeros_like(xtest)])

#y predictions

y_1 = Xalltest1 @ Paras_hat
y_0 = Xalltest0 @ Paras_hat


# Plot the predictions for z=0
plt.figure()
plt.plot(xtest, y_0, color="red")
plt.scatter(x[z==0], y[z==0])
plt.xlabel("x")
plt.ylabel("y")
plt.title("z=0")

# Plot the predictions for z=1
plt.figure()
plt.plot(xtest, y_1, color="red")
plt.scatter(x[z==1], y[z==1])
plt.xlabel("x")
plt.ylabel("y")
plt.title("z=1")

plt.show()