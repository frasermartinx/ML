
import numpy as np
import matplotlib.pyplot as plt
beta_0 = 1
beta_1 = 1

def SLM_generate(b0,b1,n,sigma_2):
    #generate x from normal
    x = np.random.randn(n)
    #get as column vector
    x = x[:,None]
    #now generate the actual model
    y = b0+b1*x
    #add noise
    y = y + np.sqrt(sigma_2)*(np.random.randn(100)[:,None])
    return x,y

sigma_2 = 3
x,y = SLM_generate(1,1,100,sigma_2)
plt.scatter(x,y)
plt.show()

### b) - Bayesian ###

def posterior(x,y,alpha,sigma_2):
    X = np.concatenate([np.ones([x.shape[0],1]),x],axis = 1)
    #alpha is the prior variance for the parameter vector
    mu = np.linalg.inv(X.T @ X + sigma_2/alpha * np.eye(X.shape[1])) @ (X.T @ y)
    Sigma = sigma_2 * np.linalg.inv(X.T @ X + (sigma_2/alpha) * np.eye(X.shape[1]))
    return mu, Sigma


### c) - Plotting posterior ###

#i)

#first do function to plot bivariate normal

from scipy.stats import multivariate_normal
def plot_biv_normal(
    mu: np.ndarray, Sigma: np.ndarray, beta0, beta1, label: str = "num_points"
) -> None:
    beta = np.array([[beta0],[beta1]])
    x1 = np.linspace(-2, 2, 101)
    x2 = np.linspace(-2, 2, 101)
    xx, yy = np.meshgrid(x1, x2)

    eval_points = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
    d_mvn = multivariate_normal(mu[:, 0], Sigma)
    density = d_mvn.pdf(eval_points).reshape(xx.shape)
    plt.figure(figsize=(5, 3))
    contours = plt.contourf(xx, yy, density, levels=100, cmap="viridis")
    plt.colorbar()
    contours = plt.contour(
        xx, yy, density, levels=[0.5, 0.67, 0.95], colors="black", alpha=1
    )
    fmt = {}
    indicated_levels = [0.95, 0.67, 0.5]
    for l, s in zip(contours.levels, indicated_levels):
        fmt[l] = s
    plt.clabel(contours, contours.levels, fontsize=8, fmt=fmt)

    # add the points to show where the MAP estimate and true parameters are
    # red star = True parameters
    # black dot = MAP estimate
    plt.scatter(
        beta[0], beta[1], color="red", label="True Parameter", marker=(5, 2), s=40
    )
    plt.scatter(mu[0], mu[1], color="black", label="MAP Estimate", s=40)
    # adjust these if you can't see the contours
    plt.xlim((-1, 2))
    plt.ylim((0, 2))
    plt.xlabel(r" $\beta_1$")
    plt.ylabel(r" $\beta_2$")
    plt.title(label)
    plt.legend()
    plt.show()

n_list = [10, 30, 100]
alpha = 0.5
for n in n_list:
    mu, Sigma = posterior(x[:n], y[:n], sigma_2, alpha)
    plot_biv_normal(mu, Sigma, beta_0,beta_1, label=f"$n={n}$")

#posterior becomes more concentrated as the number of samples increase
#the MAP estimate becomes increasingly closer to the actual value of beta