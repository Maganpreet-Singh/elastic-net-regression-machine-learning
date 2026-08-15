# Elastic Net Regression - Machine Learning

A comprehensive repository for learning **Elastic Net Regression** from the fundamentals to practical implementation. This repository covers **L1 and L2 regularization**, feature selection, coefficient shrinkage, model evaluation, hyperparameter tuning, cross-validation, and implementation using Python and Scikit-learn.

---

## 📚 About the Project

Elastic Net Regression is a regularized form of Linear Regression that combines the strengths of **Ridge Regression (L2)** and **Lasso Regression (L1)**.

It is particularly useful when dealing with:

- Multicollinearity
- High-dimensional datasets
- Many correlated features
- Feature selection
- Overfitting

Elastic Net combines both regularization penalties to create a balance between coefficient shrinkage and feature selection.

### Elastic Net Equation

The model can be written as:

```text
ŷ = β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ
```

Elastic Net adds both L1 and L2 penalties:

```text
J(β) = MSE + λ₁Σ|βⱼ| + λ₂Σβⱼ²
```

Where:

- `λ₁` = L1 regularization strength
- `λ₂` = L2 regularization strength
- `βⱼ` = Model coefficients

In Scikit-learn, Elastic Net is controlled mainly using:

- `alpha`
- `l1_ratio`

---

## 🎯 Learning Objectives

By completing this repository, you will understand:

- What Elastic Net Regression is
- Why regularization is needed
- How L1 and L2 regularization work
- How Elastic Net combines Ridge and Lasso
- The mathematics behind Elastic Net
- The role of `alpha`
- The role of `l1_ratio`
- Feature selection
- Coefficient shrinkage
- Overfitting and underfitting
- Feature scaling
- Model evaluation
- Cross-validation
- Hyperparameter tuning
- Elastic Net using Scikit-learn
- Elastic Net using Pipelines
- Elastic Net from scratch

---

## 📖 Topics Covered

### 1. Introduction

- Introduction to Regression
- Linear Regression
- Limitations of Linear Regression
- Overfitting
- Multicollinearity
- Regularization
- Ridge Regression
- Lasso Regression
- Introduction to Elastic Net

---

### 2. Elastic Net Regression

- Definition
- Working principle
- L1 Regularization
- L2 Regularization
- Combined Regularization
- Feature Selection
- Coefficient Shrinkage
- Model Complexity

---

## 🔐 L1 + L2 Regularization

Elastic Net combines the penalties used by Lasso and Ridge.

### L1 Penalty

```text
λ₁ Σ|βⱼ|
```

L1 regularization can shrink some coefficients exactly to zero, making it useful for **feature selection**.

### L2 Penalty

```text
λ₂ Σβⱼ²
```

L2 regularization shrinks coefficients toward zero and helps with **multicollinearity and model stability**.

### Elastic Net

```text
L1 + L2
```

This combination allows Elastic Net to benefit from both techniques.

---

## 🎛️ Important Hyperparameters

### Alpha

`alpha` controls the overall strength of regularization.

```python
ElasticNet(alpha=1.0)
```

Higher `alpha` means stronger regularization.

| Alpha | Regularization | Model Complexity |
|---|---|---|
| Small | Weak | Higher |
| Medium | Moderate | Balanced |
| Large | Strong | Lower |

---

### L1 Ratio

`l1_ratio` controls the balance between L1 and L2 regularization.

```python
ElasticNet(
    alpha=1.0,
    l1_ratio=0.5
)
```

| `l1_ratio` | Behavior |
|---|---|
| `0` | Mostly Ridge / L2 |
| `0.25` | More L2 |
| `0.5` | Balanced L1 + L2 |
| `0.75` | More L1 |
| `1` | Lasso / L1 |

> In practice, `alpha` and `l1_ratio` are tuned together.

---

## ⚖️ Elastic Net vs Ridge vs Lasso

| Feature | Ridge | Lasso | Elastic Net |
|---|---|---|---|
| Regularization | L2 | L1 | L1 + L2 |
| Coefficient Shrinkage | Yes | Yes | Yes |
| Exact Zero Coefficients | Usually No | Yes | Yes |
| Feature Selection | No | Yes | Yes |
| Multicollinearity | Good | Can be unstable | Good |
| Correlated Features | Good | May select one | More balanced |
| Hyperparameters | `alpha` | `alpha` | `alpha`, `l1_ratio` |

---

## 🎯 Feature Selection

One of Elastic Net's important advantages is its ability to perform feature selection.

Example:

```text
Before:

X₁ → 5.4
X₂ → 3.2
X₃ → 1.1
X₄ → 2.7
X₅ → 0.6

After Elastic Net:

X₁ → 4.8
X₂ → 2.9
X₃ → 0
X₄ → 2.1
X₅ → 0
```

Features whose coefficients become zero can be removed from the effective model.

---

## 🔄 Elastic Net Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Train-Test Split
   ↓
Feature Scaling
   ↓
Elastic Net Regression
   ↓
Prediction
   ↓
Model Evaluation
   ↓
Cross-Validation
   ↓
Hyperparameter Tuning
   ↓
Final Model
```

---

## 📊 Data Preprocessing

Important preprocessing steps include:

- Handling missing values
- Removing duplicates
- Detecting outliers
- Encoding categorical variables
- Train-test splitting
- Feature scaling

> Feature scaling is important for Elastic Net because both regularization penalties depend on coefficient magnitudes.

---

## 📏 Feature Scaling

### Standardization

```text
X_scaled = (X - mean(X)) / std(X)
```

Example:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

## 🧪 Elastic Net Using Scikit-learn

Basic implementation:

```python
from sklearn.linear_model import ElasticNet

model = ElasticNet(
    alpha=1.0,
    l1_ratio=0.5
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

---

## 🔗 Elastic Net with Pipeline

A Pipeline combines preprocessing and model training into a single workflow.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet

model = Pipeline([
    ("scaler", StandardScaler()),
    ("elasticnet", ElasticNet(
        alpha=1.0,
        l1_ratio=0.5
    ))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

---

## 🎛️ Hyperparameter Tuning

The two main hyperparameters are:

```text
alpha
l1_ratio
```

Example parameter grid:

```python
param_grid = {
    "alpha": [0.001, 0.01, 0.1, 1, 10, 100],
    "l1_ratio": [0, 0.25, 0.5, 0.75, 1]
}
```

Using `GridSearchCV`:

```python
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV

model = ElasticNet(max_iter=10000)

param_grid = {
    "alpha": [0.001, 0.01, 0.1, 1, 10, 100],
    "l1_ratio": [0, 0.25, 0.5, 0.75, 1]
}

grid = GridSearchCV(
    model,
    param_grid,
    cv=5,
    scoring="r2"
)

grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)
```

---

## 🔄 Cross-Validation

Cross-validation helps determine which combination of:

- `alpha`
- `l1_ratio`

provides the best generalization.

Instead of evaluating the model on only one validation split, multiple folds are used to obtain a more reliable estimate of model performance.

---

## 📈 Model Evaluation

The following regression metrics can be used:

### Mean Absolute Error

```text
MAE = (1/n) Σ|yᵢ - ŷᵢ|
```

### Mean Squared Error

```text
MSE = (1/n) Σ(yᵢ - ŷᵢ)²
```

### Root Mean Squared Error

```text
RMSE = √MSE
```

### R² Score

```text
R² = 1 - (SSres / SStot)
```

Example:

```python
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R²:", r2)
```

---

## ⚠️ Overfitting and Underfitting

### Underfitting

Can occur when regularization is too strong.

Characteristics:

- High training error
- High testing error
- Coefficients heavily shrunk
- Model is too simple

### Good Fit

The model balances complexity, regularization, and generalization.

### Overfitting

Can occur when regularization is too weak.

Characteristics:

- Very low training error
- Higher test/validation error
- Unnecessary model complexity
- Large coefficients

```text
Weak Regularization
       ↓
Higher Complexity
       ↓
Possible Overfitting

Balanced Regularization
       ↓
Better Generalization

Strong Regularization
       ↓
Lower Complexity
       ↓
Possible Underfitting
```

---

## 🧠 Elastic Net and Multicollinearity

Elastic Net is particularly useful when several input features are strongly correlated.

For example:

```text
X₁ ↔ X₂
X₂ ↔ X₃
X₃ ↔ X₄
```

When features are highly correlated:

- Ordinary Linear Regression can produce unstable coefficients.
- Lasso may arbitrarily select one feature from a correlated group.
- Ridge handles correlated features well.
- Elastic Net provides a balance between feature selection and coefficient stability.

---

## 🧑‍💻 Elastic Net From Scratch

The repository can also explore the mathematical implementation of Elastic Net from scratch.

Important concepts include:

- L1 penalty
- L2 penalty
- Combined cost function
- Coefficient optimization
- Gradient-based optimization
- Feature selection
- Prediction
- Model evaluation

The from-scratch implementation helps build an understanding of how regularization affects model training internally.

---

## 💡 Advantages

- Combines Ridge and Lasso
- Helps reduce overfitting
- Performs feature selection
- Handles multicollinearity
- Works well with correlated features
- Useful for high-dimensional datasets
- Produces simpler models
- Flexible balance between L1 and L2 regularization

---

## ⚠️ Disadvantages

- More hyperparameters to tune
- Requires feature scaling
- Can be computationally more expensive than simple Linear Regression
- Poor parameter selection can cause underfitting
- Less interpretable than simple Linear Regression
- Requires cross-validation for reliable hyperparameter selection

---

## 🌍 Real-World Applications

Elastic Net Regression can be applied to:

- House Price Prediction
- Salary Prediction
- Medical Data Analysis
- Financial Modeling
- Marketing Analytics
- Sales Prediction
- Demand Forecasting
- High-dimensional datasets
- Feature selection problems
- Predictive modeling with correlated variables

---

## 🧠 Key Takeaways

- Elastic Net combines **L1 and L2 regularization**.
- L1 regularization helps with feature selection.
- L2 regularization helps with coefficient stability and multicollinearity.
- `alpha` controls the overall regularization strength.
- `l1_ratio` controls the balance between L1 and L2.
- `l1_ratio = 1` behaves like Lasso.
- `l1_ratio = 0` behaves like Ridge.
- Feature scaling is important.
- Cross-validation helps select good hyperparameters.
- Elastic Net is especially useful for datasets with many correlated features.
- Strong regularization can lead to underfitting.
- Weak regularization can lead to overfitting.

---

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

---

## 📚 Learning Resources

- *Introduction to Statistical Learning (ISLR)*
- *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow*
- Scikit-learn Documentation
- Andrew Ng Machine Learning Course
- StatQuest

---

## 🚀 Future Improvements

- Add more real-world datasets
- Compare Elastic Net with Linear Regression
- Compare Elastic Net with Ridge and Lasso
- Add coefficient visualization
- Add hyperparameter experiments
- Add Elastic Net from scratch using NumPy
- Add feature-selection experiments
- Add complete end-to-end projects
- Add model deployment examples

---

## 👨‍💻 Author

**Maganpreet Singh**

B.Tech Computer Science & Engineering

Machine Learning | Data Science | Python

---

## ⭐ Support

If you find this repository helpful for learning Elastic Net Regression, consider giving it a ⭐ on GitHub.

---

**Keep learning. Keep building. Keep experimenting. 🚀**
