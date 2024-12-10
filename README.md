# Machine Learning

**Machine learning** refers to a computer's ability to learn without being explicitly programmed.  
It was defined by **Arthur Samuel** in the 1950s. He demonstrated an example of a computer becoming a better checkers player than himself after playing against itself 10,000 times.

## Types of Machine Learning
- [ ] **Supervised Learning** (most used)
  - [ ] **Regression**: Example - fitting, infinite outputs
  - [ ] **Classification**: Finite outputs
- [ ] **Unsupervised Learning**
  - [ ] **Clustering**
  - [ ] **Anomaly Detection**
  - [ ] **Dimensionality Reduction**

### Supervised Learning
Learn the mapping from `X` to `Y` using given data to predict the correct values of `Y`.

**Example**: Regression (fitting a curve or line to data points).

### Unsupervised Learning
Identify patterns and structures within the given data.

---

## Regression
Regression estimates the relationships between a dependent variable and one or more independent variables.

### Notations:
- A training set consists of input variables (`x`) and output or target variables (`y`), with `m` training examples.
- The `i`-th training example is denoted as `(x(i), y(i))`.
- The predicted value of the target variable is denoted as `y^`.

---

## Cost Function (`J`)
The cost function compares how well the prediction matches the true value.  
**Example**: Squared Error Cost Function

<p align="center">
    J(w, b) = (1 / 2m) Σ<sub>i=1</sub><sup>m</sup> (f<sub>w,b</sub>(x<sup>(i)</sup>) - y<sup>(i)</sup>)²
</p>


The values of `w` and `b` that minimize `J` give the best-fit function `f_w,b(x)`.

---

## Gradient Descent
Gradient Descent is a method to minimize a function. It iteratively updates the values of parameters until the derivative (slope) approaches zero.

### Key Steps:
1. Use the partial derivatives with respect to `w` and `b` to find the slope direction.
2. Adjust the parameters using a learning rate (`α`) to define the step size.

tmp_w = w - α * ∂J(w, b) / ∂w  
tmp_b = b - α * ∂J(w, b) / ∂b  
w = tmp_w  
b = tmp_b  

