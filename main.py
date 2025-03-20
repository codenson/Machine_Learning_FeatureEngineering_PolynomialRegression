import numpy as np
import matplotlib.pyplot as plt
from lab_utils_multi import zscore_normalize_features, run_gradient_descent_feng
np.set_printoptions(precision=2)  # reduced display precision on numpy arrays

# create target data
x = np.arange(0, 20, 1)
y = 1 + x**2
X = x.reshape(-1, 1)

model_w,model_b = run_gradient_descent_feng(X,y,iterations=1000, alpha = 1e-2)

def print_linear_graph():
    plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("no feature engineering")
    plt.plot(x,X@model_w + model_b, label="Predicted Value");  plt.xlabel("X"); plt.ylabel("y"); plt.legend(); plt.show()

print_linear_graph()

# create target data
x = np.arange(0, 20, 1)
y = 1 + x**2

# Engineer features  for swquared term
X = x**2      #<-- added engineered feature

X = X.reshape(-1, 1)  #X should be a 2-D Matrix


def print_quadratic_graph_1():
    model_w,model_b = run_gradient_descent_feng(X, y, iterations=10000, alpha = 1e-5)
    plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("Added x**2 feature")
    plt.plot(x, np.dot(X,model_w) + model_b, label="Predicted Value"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()

print_quadratic_graph_1()

# create target data for cubic term
x = np.arange(0, 20, 1)
y = x**2

# engineer features .
X = np.c_[x, x**2, x**3]   #<-- added engineered feature

def print_polynomial_graph():
    model_w,model_b = run_gradient_descent_feng(X, y, iterations=10000, alpha=1e-7)
    plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("x, x**2, x**3 features")
    plt.plot(x, X@model_w + model_b, label="Predicted Value"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()

print_polynomial_graph()

print(X)
