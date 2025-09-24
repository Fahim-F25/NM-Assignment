# NM-Assignment
<br>
# 📘 README Description

-----01-----

## Logistic Regression Optimization using Gradient Descent and Newton’s Method

This project demonstrates the implementation and comparison of two popular optimization techniques — **Gradient Descent** and **Newton’s Method** — for minimizing the cost function of a **Logistic Regression** model.

-----02-----
### 🔹 Key Features

* Generates a *synthetic binary classification dataset* using scikit-learn.
* Implements logistic regression cost function with:

  * *Sigmoid function*
  * *Cost calculation (Negative Log-Likelihood)*
  * *Gradient computation*
  * *Hessian computation* (for Newton’s method)
* Optimizes logistic regression parameters (θ) using:


  -----03-----

  1. *Gradient Descent* – iterative updates with a learning rate.
  2. *Newton’s Method* – updates using both gradient and Hessian (second-order derivative).
* Compares *convergence speed* of both methods by plotting cost vs iteration.


-----04-----
### 🔹 Results

* *Gradient Descent* takes many small steps to converge, depending on the learning rate.
* *Newton’s Method* converges much faster in fewer iterations, but requires matrix inversion, which is computationally expensive for high-dimensional data.
* A *comparison plot* shows the difference in convergence behavior.
  

-----05-----

### 🔹 Technologies Used

* *Python*
* *NumPy* – for mathematical operations
* *Scikit-learn* – for dataset generation
* *Matplotlib* – for visualization

