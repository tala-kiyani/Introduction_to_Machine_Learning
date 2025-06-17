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

---

### 📝 Exercise 4: Polynomial Regression and Model Complexity (Underfitting vs. Overfitting)

This notebook implements polynomial regression to fit a model to data with non-linear patterns. It uses the analytical closed-form solution on engineered polynomial features.

Crucially, it explores the relationship between **model complexity** (the degree of the polynomial) and model performance by analyzing the Root Mean Square Error (RMSE) on both training and testing datasets. This helps visualize the concepts of **underfitting**, **good fit**, and **overfitting**.

#### ➗ Methodology

**1. Feature Engineering**
To fit a non-linear model, the original feature `x` is transformed into polynomial features. For a degree `m`, the input matrix `X` is expanded to include columns for `x`, `x²`, ..., `x^m`. The model's hypothesis then becomes:

$$ h_w(x) = w_0 + w_1 x + w_2 x^2 + \dots + w_m x^m $$

**2. Error Metric**
The **Root Mean Square Error (RMSE)** is used to evaluate the model's performance on both the training and test sets.

$$ RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n}(y_{pred_i} - y_i)^2} $$

#### 📈 Analysis: Underfitting vs. Overfitting

By training models with polynomial degrees from 0 to 8, we can observe the bias-variance tradeoff:

* **Underfitting (Low Degree, e.g., 0-1)**: The model is too simple to capture the underlying trend. Both **training and testing errors are high**.
* **Good Fit (Optimal Degree, e.g., 2-3)**: The model generalizes well. Both **training and testing errors are low**, with the test error being at its minimum. In this case, a degree 3 polynomial seems optimal.
* **Overfitting (High Degree, e.g., 4+)**: The model becomes too complex and starts fitting the noise in the training data. The **training error continues to decrease**, but the **testing error starts to increase**, indicating poor generalization to unseen data.

This relationship is visualized in the RMSE vs. Polynomial Degree plot:
---

### 📝 Exercise 5: Generalization & Learning Curves (Underfitting vs. Overfitting)

This notebook dives deeper into model generalization by plotting **learning curves**. Learning curves are a crucial diagnostic tool used to understand how a model's performance on the training and testing sets changes as the number of training examples increases. This analysis is performed for three polynomial regression models to illustrate underfitting, a good fit, and overfitting.

#### ➗ Methodology

Learning curves are generated by training a model on progressively larger subsets of the training data. For each subset size, the model's error (RMSE) is calculated on both that training subset and the *entire* validation set.

#### 📈 Analysis of Learning Curves

**1. Underfitting (Degree `m=1`)**
The learning curves for the linear model show that both the training and validation errors are high and quickly plateau at a similar error value. This indicates that the model is too simple (high bias) and adding more data will not improve its performance, as it cannot capture the underlying complexity of the data.

**2. Good Fit (Degree `m=3`)**
For the cubic model, the training error starts low and increases, while the validation error starts high and decreases. The two curves converge to a low error value, with a small gap between them. This is the hallmark of a well-generalized model that performs well on both seen and unseen data.

**3. Overfitting (Degree `m=10`)**
The learning curves for the high-degree polynomial show a large and persistent gap between the training error (which is very low) and the validation error (which is high). This signifies that the model is too complex (high variance). It has "memorized" the training data, including its noise, and fails to generalize to new data.

#### 📊 Visualization

A final plot compares the three fitted polynomial curves (degrees 1, 3, and 10) against the dataset, providing a clear visual representation of underfitting, a good fit, and overfitting.
---

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
### 📝 Exercise 2: Applying Gradient Descent to Linear Regression

This exercise demonstrates how to use the **Gradient Descent** algorithm to find the optimal parameters for a linear regression model. This iterative approach serves as an alternative to the analytical closed-form solution.

#### ➗ Methodology

**1. Cost Function (MSE)**
The goal is to minimize the **Mean Squared Error (MSE)** cost function, which measures the average squared difference between the predicted ($h_w(x_i)$) and actual ($y_i$) values.

$$ J(w) = \frac{1}{2n} \sum_{i=1}^{n}(h_w(x_i) - y_i)^2 \quad \text{where} \quad h_w(x_i) = w_0 + w_1 x_i $$

**2. Algorithm**
Gradient descent iteratively adjusts the parameters $w_0$ and $w_1$ by moving in the direction of the negative gradient of the cost function. The algorithm is run for 1000 steps with a learning rate of 0.01.

#### 🔢 Results

The parameters found by the gradient descent algorithm are:
* **Calculated Slope ($w_1$)**: 2.86
* **Calculated Intercept ($w_0$)**: 6.87

These results are nearly identical to those found using the Normal Equation, demonstrating that for a convex problem like linear regression, both methods converge to the same optimal solution.

#### 📈 Visualization

---
The final plot shows the synthetic data points and the best-fit line discovered by the gradient descent algorithm.
## Chapter 03: Case Study - Regression

### 🏠 Project 1: Tehran House Price Prediction

This project is an end-to-end case study focused on predicting house prices in Tehran using a real-world dataset. The notebook covers data loading, cleaning, exploratory data analysis (EDA), feature engineering, model training, and evaluation.

#### ➗ Methodology
The project uses **Ridge Regression** (Linear Regression with L2 Regularization) to predict prices. Key preprocessing steps include one-hot encoding for the `Address` feature and `StandardScaler` for numerical features.

#### 📈 Evaluation
The model's performance is evaluated on a test set using **Mean Squared Error (MSE)** and the **R-squared (R²)** metric.
---


### 🏠 Project 2: California House Price Prediction with Linear Regression

This project uses a multiple linear regression model to predict median house values in California districts. It demonstrates a foundational machine learning workflow using the Scikit-learn library to load a dataset, split the data, train a model, and evaluate its performance.

#### 📊 Dataset
The project uses the classic **California Housing dataset**, loaded directly from Scikit-learn. The goal is to predict the `target` (median house value) based on 8 numerical features, including median income, house age, and location.

#### ➗ Methodology
The workflow is a straightforward implementation of a supervised learning task:
1.  **Load Data**: The dataset is loaded using `sklearn.datasets.fetch_california_housing`.
2.  **Data Splitting**: The data is split into a training set (80%) and a testing set (20%) to ensure the model can be evaluated on unseen data.
3.  **Model Training**: A `LinearRegression` model from Scikit-learn is initialized and trained on the training data (`X_train`, `y_train`).
4.  **Prediction**: The trained model is used to make predictions on the test set (`X_test`).

#### 📈 Evaluation
The model's accuracy is evaluated on the test set by comparing the predicted values (`y_pred`) against the actual values (`y_test`).
* **Metric**: Mean Squared Error (MSE) is calculated to quantify the average squared difference between the predicted and actual prices.

#### 📊 Visualization
A scatter plot is created to visually inspect the model's performance. It plots the actual house prices from the test set against the predicted prices. In a perfect model, all points would lie on a 45-degree diagonal line.
