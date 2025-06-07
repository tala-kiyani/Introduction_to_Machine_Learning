# Introduction to Machine Learning: Coursework & Notes

This repository contains all the code, exercises, and notes from my journey through the "Introduction to Machine Learning" course. Each chapter and project is organized into its own directory.

---
## Chapter 01: Supervised Learning

### 📝 Exercise 1: Simple Linear Regression from Scratch

This project implements a simple linear regression model from scratch using Python. The goal is to calculate the regression coefficients (slope and intercept) for a given set of data points without relying on machine learning libraries like Scikit-learn.

**The final regression equation is: $$ y = 3.35x - 2.85 $$**

#### 📊 Dataset
The model is built using the following sample data points:
* **x**: `[5, 3, -1, 2, 6]`
* **y**: `[14, 6, -5.5, 3.5, 18]`

#### ➗ Methodology
The regression line is of the form **$y = w_1x + w_0$**, where:
* $w_1$ is the slope of the line.
* $w_0$ is the y-intercept.

These coefficients are calculated using the following formulas:

1.  **Slope ($w_1$)**:
    $$ w_1 = \frac{n(\sum x_i y_i) - (\sum x_i)(\sum y_i)}{n(\sum x_i^2) - (\sum x_i)^2} $$

2.  **Y-Intercept ($w_0$)**:
    $$ w_0 = \frac{\sum y_i - w_1 \sum x_i}{n} $$

#### 🔢 Results
Based on the provided dataset, the calculated components are:
* **Number of data points (n)**: 5
* **Sum of x ($\sum x_i$)**: 15
* **Sum of y ($\sum y_i$)**: 36.0
* **Sum of xy ($\sum x_i y_i$)**: 208.5
* **Sum of x squared ($\sum x_i^2$)**: 75

Plugging these values into the formulas gives the final coefficients:
* **Slope ($w_1$)**: `3.35`
* **Y-Intercept ($w_0$)**: `-2.85`

#### ▶️ How to Run
The code is in the `Simple_Regression.ipynb` Jupyter Notebook. You can open it and run the cells sequentially to see the calculations.

---

### 📝 Exercise 2: Analytical Linear Regression & Loss Surface Visualization

This exercise expands on simple linear regression by first simulating data with a known relationship and then finding the line of best fit analytically. It also visualizes the **Mean Squared Error (MSE) loss surface** to show how the error changes with different coefficient values.

#### 📊 Data Simulation
The data is simulated based on the linear equation **$y = -15x + 20$**, with Gaussian noise added to the `y` values to make the task more realistic. A total of 1000 data points were generated.

#### ➗ Methodology
**1. Analytical Solution**

The regression coefficients ($b_1$ for the slope and $b_0$ for the intercept) are calculated directly using the same formulas as in Exercise 1.

**2. Loss Function Visualization**

The **Mean Squared Error (MSE)** is used as the loss function to measure the model's error. The formula is:

$$ MSE = \frac{1}{n} \sum_{i=1}^{n}(y_{pred_i} - y_i)^2 = \frac{1}{n} \sum_{i=1}^{n}((b_1x_i + b_0) - y_i)^2 $$

A 3D surface plot is generated to show the MSE loss (on a log scale for better visualization) for a grid of different `b0` and `b1` values. The lowest point on this surface represents the optimal coefficients that minimize the error.

#### 🔢 Results
The analytical calculation produced the following regression equation:

**$$ y = -15.02x + 20.18 $$**

This is extremely close to the true relationship of $y = -15x + 20$, demonstrating that the analytical method successfully recovered the underlying pattern from the noisy data.

#### 📈 Visualization
The 3D plot of the MSE loss surface shows a clear convex bowl shape. The bottom of the bowl is the point of minimum loss, which corresponds to the analytically solved coefficients ($b_0 \approx 20$, $b_1 \approx -15$).


### 📝 Exercise 3: Linear Regression with Closed-Form (Analytical) Solution

This notebook demonstrates finding the optimal parameters for a linear regression model using the analytical closed-form solution, also known as the **Normal Equation**. The process involves generating synthetic data, applying the formula to find the coefficients, and plotting the resulting regression line.

#### 📊 Data Generation
Synthetic data is generated based on the true linear relationship **$y = 3x + 8$**, with random noise added to simulate a real-world dataset. Fifty data points are created for this purpose.

#### ➗ Methodology
The linear hypothesis is defined as **$h_w(x) = w_0 + w_1 x_1$**. To solve for the parameter vector **$w$** (containing $w_0$ and $w_1$), the Normal Equation is used:

$$ w = (X^T X)^{-1} X^T y $$

Where:
* **$w$** is the vector of model parameters (intercept and slope).
* **$X$** is the input feature matrix (with a bias term of $x_0 = 1$ added).
* **$y$** is the vector of target values.

This formula directly calculates the optimal **$w$** that minimizes the cost function without requiring an iterative optimization process like gradient descent.

#### 🔢 Results
The model's calculated parameters are very close to the true values used to generate the data:
* **True Slope**: 3
* **Calculated Slope ($w_1$)**: 2.86
<br>
* **True Intercept**: 8
* **Calculated Intercept ($w_0$)**: 6.87

#### 📈 Visualization
The notebook includes two plots:
1.  A scatter plot of the generated data points.
2.  A final plot showing the original data points along with the fitted linear regression line calculated by the model.


## Chapter 02: Optimization

### 📝 Exercise 1: Gradient Descent on a Convex Function

This notebook implements the gradient descent algorithm from scratch to find the minimum of a convex function. The optimization path is visualized on a contour plot to show how the algorithm iteratively converges to the optimal solution.

#### ➗ Methodology

**1. Cost Function**

A simple convex cost function, **$J(w)$**, is defined with two parameters, $w_0$ and $w_1$:

$$ J(w) = w_0^2 + 2w_1^2 $$

The global minimum of this function is at the point (0, 0).

**2. Gradient Descent Algorithm**

Gradient descent is an iterative optimization algorithm used to find the minimum of a function. It works by repeatedly taking steps in the direction opposite to the gradient of the function at the current point. The update rule is:

$$ w_{next} = w_{current} - \eta \nabla J(w_{current}) $$

Where:
* **$w$** is the vector of parameters.
* **$\eta$** (eta) is the learning rate, which controls the step size.
* **$\nabla J(w)$** is the gradient of the cost function.

#### 🔢 Execution Details
The algorithm was run with the following hyperparameters:
* **Initial Point ($w_{init}$)**: `[4, 4]`
* **Learning Rate ($\eta$)**: `0.01`
* **Number of Steps**: `180`

#### 📈 Visualization
The result is visualized as a contour plot, where each line represents a constant value of the cost function. The path taken by gradient descent is plotted on top, clearly showing the steps from the initial point `[4, 4]` converging toward the minimum at `[0, 0]`.

