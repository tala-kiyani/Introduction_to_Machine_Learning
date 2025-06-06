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
