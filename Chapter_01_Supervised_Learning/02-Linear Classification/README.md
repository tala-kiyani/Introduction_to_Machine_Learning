# Exercise: Linear Classification with the Perceptron

This notebook introduces the fundamentals of linear classification. It begins by demonstrating why linear regression is not suitable for classification tasks and then builds a **Perceptron**—one of the earliest and simplest algorithms for binary classification—from scratch.

The implemented Perceptron model is then trained and evaluated on the Breast Cancer dataset.

---

### 1. The Limitation of Linear Regression for Classification

The first part of the notebook uses the Iris dataset to illustrate a key conceptual point. By attempting to fit a `LinearRegression` model to a binary classification problem (e.g., 'Is this flower an Iris-versicolor?'), we can see that the resulting "line of best fit" does not produce a logical decision boundary for separating the classes. This motivates the need for algorithms specifically designed for classification, like Logistic Regression or the Perceptron.

---

### 2. The Perceptron Algorithm

The Perceptron is a simple algorithm for supervised learning of a binary classifier. It is a linear classifier, meaning it learns a linear boundary to separate two classes of data.

#### Methodology

The notebook implements a `Perceptron` class from scratch with the following core logic:

* **Net Input Function**: The model first computes a weighted sum of the input features, adding a bias term. This is the net input, `z`.

    $$ z = w \cdot x + b = w_1x_1 + w_2x_2 + \dots + w_nx_n + b $$

* **Step Function (Activation)**: The net input `z` is then passed to a step function (in this case, a Heaviside step function). If the output is greater than a threshold (e.g., 0), the model predicts one class (e.g., class 1); otherwise, it predicts the other (e.g., class 0).

* **Learning Rule**: The model learns its weights (`w`) and bias (`b`) iteratively. For each training example:
    1.  A prediction is made.
    2.  If the prediction is incorrect, the weights are updated to move the decision boundary closer to correctly classifying the example. The update rule is:

        `w_new = w_old + learning_rate * (y_true - y_pred) * x`
        
        `b_new = b_old + learning_rate * (y_true - y_pred)`

---

### 3. Case Study: Breast Cancer Dataset Classification

The custom-built Perceptron is applied to the real-world task of classifying breast cancer tumors as either malignant or benign.

#### 📊 Dataset

The project uses the **Breast Cancer Wisconsin** dataset from Scikit-learn. For visualization purposes, only two features are used for training the classifier:
* `mean radius`
* `mean texture`

#### 📈 Results & Evaluation

The Perceptron model is trained on a portion of the dataset and then evaluated on an unseen test set. The **accuracy** of the model—the percentage of correctly classified tumors—is calculated to measure its performance.

#### 📊 Visualization

The final output is a scatter plot that visualizes the model's performance on the test data. It shows:
* The data points for the two classes (malignant and benign).
* The linear **decision boundary** that the Perceptron has learned. This line separates the two-dimensional feature space into the two predicted classes.

**![Perceptron Decision Boundary](path/to/your/perceptron_boundary_plot.png)**

> **Note:** To make the image appear, save the plot from your notebook as an image file (e.g., `perceptron_boundary_plot.png`), add it to your repository, and update the path in the markdown line above.