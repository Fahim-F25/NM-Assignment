import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

# 1. Generate synthetic dataset
x, y = make_classidication( n_samples = 200, n_features = 2, n_classes = 2, n_redundant = 0,random_state = 42)

# Add intercept term (bias)
X = np.hstack([np.ones((X.shape[0], 1)), X]) # shape (m,n+1)
m, n = X.shape

# Reshape y to column vector
y = y.reshape(-1,1)

#2.Define logistic function
def sigmoid(z):
    return 1/(1 + np.exp(-z))

def cost_function(theta, X, y):
    h = sigmoid(X @ theta)
    return -(1/m) * (y.T @ np.log(h) + (1-y).T @ np.log(1-h))


def gradient(theta, X, y):
    h = sigmoid(X @ theta)
    return (1/m) * (X.T @ (h - y))


def hessian(theta, X, y):
    h = sigmoid(X @ theta)
    R = np.diag((h * (1-h)).flatten())
    return (1/m) * (X.T @ R @ X)

# 3. Gradiant Descent
def logistic_regression_gd(X, y , alpha=0.1, num_iter=1000):
    theta = np.zeros((X.shape[1],1))
    costs = []

    for i in range(num_iter):
        grad = gradient(theta, X, y)
        theta -= alpha * grad
        costs.append(cost_function(theta, X, y).item())
    return theta, costs 


# 4. Newton's Method
def logistic_regression_newton(X , y , num_iter=10):
    theta = np.zeros((X.shape[1], 1))
    costs = []

    for i in range(num_iter):
        grad = gradient(theta,X ,y)
        H = hessian(theta, X , y)
        theta -= np.linalg.inv(H) @ grad
        costs.append(cost_function(theta, X, y).item())
    return theta, costs


# 5. Run both methods
theta_gd, costs_gd = logistic_regressio_gd(X, y, alpha=0.1, num_iter=300)
theta_newton, costs_newton = logistic_regression_newton(X, y, num_iter=10)

print("Final theta (Gradient Descent):", theta_gd.ravel())
print("Final theta (Newton's Method):", theta_newton.ravel())

#6.Plot convergence
plt.plot(costs_gd,label="Gradient Descent")
plt.plot(costs_newton,label="Newton's Method")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Convergence Comparison")
plt.legend()
plt.show()




    
