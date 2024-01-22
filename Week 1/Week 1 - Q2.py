import numpy as np
import matplotlib.pyplot as plt

def generate_data() -> np.array:
    #std of noise
    noise_std = 100
    #input variable
    x = np.linspace(-5,5,100)
    #x is currently a (100,) vector, or a list

    #get as column vector
    x = x[:,None]
    #x is now (100,1) this is the form we want a vector in

    #true function
    y = 5 * x ** 3 - x ** 2 + x + noise_std * np.random.normal(size = [x.shape[0],1])

    #return data together
    data = np.concatenate([x,y],axis = 1)
    return data

data = generate_data()
print(data.shape)
#plot
#plt.scatter(data[:,0],data[:,1])
#plt.show()

### b) ###
#MSE of a polynomial model

#we can write what data type each variable should be
def mse(x: np.ndarray,y: np.ndarray,polyorder: int) -> float:
    #x is input
    #y is output
    #polyorder is the order of the polynomial we are fitting

    ### creating the design matrix:
    #pad the first column with ones for intercept term
    X = np.ones([x.shape[0],1])
    #then we iteratively add polynomial order terms using concatenate
    for p in range(1, polyorder + 1):
        X = np.concatenate([X,x**p],axis = 1)
    #solve for coeffs, noting this is still just a linear model
    coeffs = np.linalg.solve(X.T @ X, X.T @ y)
    y_pred = X @ coeffs
    mse = np.mean((y_pred - y)**2)
    return mse


#now we test

orders = [i for i in range(1,9)]
plt.plot(orders, [mse(data[:,:1],data[:,1:],i) for i in orders])
plt.ylabel("MSE")
plt.xlabel("Order")
plt.show()

#this looks good but we predicted on the training set, we did not see how it generalises to test sets.

### c)
from sklearn.model_selection import train_test_split

def get_mse_ranodmsplit(x, y, polyorder):

    #again we construct the design matrix:
    X = np.ones([x.shape[0],1])
    #then we iteratively add polynomial order terms using concatenate
    for p in range(1, polyorder + 1):
        X = np.concatenate([X,x**p],axis = 1)

    #we then split the data into the test and the training set
    #the test size parameter controls the split ratio
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 2)
    #solve
    coeffs = np.linalg.solve(X_train.T @ X_train, X_train.T @ y_train)
    y_pred_test = X_test @ coeffs
    y_pred_train = X_train @ coeffs
    mse_test = np.mean((y_pred_test - y_test) ** 2)
    mse_train = np.mean((y_pred_train - y_train) ** 2)
    return mse_test,mse_train

orders = [i for i in range(1,9)]
MSE = [get_mse_ranodmsplit(data[:,:1],data[:,1:],i) for i in orders]

plt.plot(orders, [list(MSE[i])[0] for i in range(len(orders))],label = "test")
plt.plot(orders, [list(MSE[i])[1] for i in range(len(orders))],label = "train")
plt.legend()
plt.ylabel("MSE")
plt.xlabel("Order")
plt.show()

#we see degree three is the best
#on the training set it tends to decrease, whereas on the test set it increases as overfitting begins


### d) LOO-CV ###

def get_MSE_LOO(x,y,polyorder):
    #again we construct the design matrix:
    X = np.ones([x.shape[0],1])
    #then we iteratively add polynomial order terms using concatenate
    for p in range(1, polyorder + 1):
        X = np.concatenate([X,x**p],axis = 1)



    #we then do the LOO-CV bit
    CV = np.zeros([x.shape[0],1])
    for i in range(x.shape[0]):
        X_train, X_test = np.delete(X,i,axis = 0), X[i,:]
        y_train, y_test = np.delete(y, i, axis=0), y[i, :]
    #solve
        coeffs = np.linalg.solve(X_train.T @ X_train, X_train.T @ y_train)
        y_pred_test = X_test @ coeffs
        mse_test = np.mean((y_pred_test - y_test) ** 2)
        CV[i] = mse_test
    CV_mean = np.mean(CV)
    CV_sd = np.std(CV)
    return CV_mean, CV_sd

cv_means = []
cv_sds = []

#for each order, run the above and get the values:
for p in orders:
    cv_mean, cv_sd = get_MSE_LOO(data[:,:1], data[:,1:], p)
    cv_means.append(cv_mean)
    cv_sds.append(cv_sd)

#plot

plt.errorbar(orders,cv_means,yerr = cv_sds, fmt = 'o-')
plt.xlabel("order")
plt.ylabel("LOO-CV")
plt.show()


### e) - Ridge ###
#here we are told the degree is 5
def get_MSE_ridge(x,y,lamb):

    #again we construct the design matrix:
    X = np.ones([x.shape[0],1])
    #then we iteratively add polynomial order terms using concatenate
    for p in range(1, 6):
        X = np.concatenate([X,x**p],axis = 1)

    #now do LOO-CV
    CV = np.zeros([x.shape[0],1])
    for i in range(x.shape[0]):
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

lamb = 2 ** np.linspace(0, 15, 151)
ridge_mses = [get_MSE_ridge(data[:, :1], data[:, 1:], l)[0] for l in lamb]
plt.plot(lamb, ridge_mses, label="Test")
plt.legend()
plt.ylabel("Averaged MSE")
plt.xlabel("$\lambda$")
plt.show()

#we can then find the minimum mse lambda value:
lmbda = lamb[np.argmin(ridge_mses)]
print("Lambda minimising LOOCV MSE=", lmbda)


# now compare on the same test train split, using the value of lambda we just found. with degree 5
x = data[:, :1]
y = data[:, 1:]
X = np.ones((x.shape[0], 1))
for p in range(1, 5 + 1):
    X = np.concatenate([X, x ** p], axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=2
)

# obtain ridge solution
coefs = np.linalg.solve(
    X_train.T @ X_train + lmbda * np.eye(X_train.shape[1]), X_train.T @ y_train
)
pred_y = X_test @ coefs
mse_ridge = np.mean((pred_y - y_test) ** 2)

# obtain OLS solution
coefs = np.linalg.solve(X_train.T @ X_train, X_train.T @ y_train)
pred_y = X_test @ coefs
mse_OLS = np.mean((pred_y - y_test) ** 2)

print(f"MSE_ridge: {mse_ridge}, MSE_OLS: {mse_OLS}")
#so we see ridge outperforms OLS