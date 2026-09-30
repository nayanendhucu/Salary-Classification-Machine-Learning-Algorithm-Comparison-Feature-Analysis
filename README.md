# Salary Classification: Machine Learning Algorithm Comparison & Feature Analysis

## Overview

This project is a practical machine learning experimentation and algorithm-comparison project using the **Adult Census Income dataset**.

The objective is to predict whether an individual's annual income falls into the `<=50K` or `>50K` category while experimenting with multiple machine learning algorithms and preprocessing techniques.

Rather than focusing on a single model, the project explores how different classification algorithms behave on the same dataset and how techniques such as feature scaling, class balancing, feature selection, and feature analysis affect model performance.

The project includes:

* Data exploration
* Categorical feature encoding
* Train-test splitting
* Feature scaling
* K-Nearest Neighbors
* Decision Tree classification
* Support Vector Classification
* Random undersampling
* Random oversampling
* Tomek Links
* Chi-square feature selection
* Feature importance analysis
* Correlation analysis
* Classification reports
* Accuracy comparison
* Data visualization

---

# Dataset

The project uses the **Adult Census Income dataset**, commonly used for binary classification and machine learning experimentation.

The dataset contains demographic, employment, education, and financial-related attributes used to classify income level.

### Dataset Size

* Rows: **32,561**
* Columns: **15**
* Target variable: `salary`

### Target Classes

The target variable contains two classes:

```text
<=50K
>50K
```

### Class Distribution

```text
<=50K : 24,720
>50K  : 7,841
```

The target distribution is imbalanced, which makes class-balancing techniques relevant to the experimentation.

---

# Dataset Features

The dataset contains the following features:

| Feature          | Description                           |
| ---------------- | ------------------------------------- |
| `age`            | Age of the individual                 |
| `workclass`      | Type of employment                    |
| `fnlwgt`         | Final sampling weight                 |
| `education`      | Education category                    |
| `education-num`  | Numerical representation of education |
| `marital-status` | Marital status                        |
| `occupation`     | Occupation category                   |
| `relationship`   | Relationship status                   |
| `race`           | Race category                         |
| `sex`            | Sex category                          |
| `capital-gain`   | Capital gain                          |
| `capital-loss`   | Capital loss                          |
| `hours-per-week` | Weekly working hours                  |
| `native-country` | Native country                        |
| `salary`         | Income classification target          |

---

# Objective

The main objective is to build and compare different machine learning classification algorithms for salary classification.

The project also investigates:

* How preprocessing affects different algorithms
* How class imbalance affects classification
* Whether resampling techniques improve performance
* How feature selection affects SVC performance
* Which features contribute strongly to classification
* Relationships between numerical features

---

# Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Imbalanced-learn
* Jupyter Notebook / Google Colab

---

# Machine Learning Workflow

```text
Adult Census Income Dataset
            |
            v
       Data Exploration
            |
            v
      Missing Value Check
            |
            v
   Separate Features & Target
            |
            v
    Encode Categorical Data
            |
            v
      Train-Test Split
            |
            +----------------------+
            |                      |
            v                      v
      Standard Scaling       Original Features
            |                      |
            v                      v
           KNN              Decision Tree
            |
            v
      SVC Classification
            |
            v
   Class-Balancing Experiments
            |
            +----------------------+
            |          |           |
            v          v           v
      Undersampling  Oversampling  Tomek Links
            |
            v
      Feature Selection
            |
            v
       Chi-Square Test
            |
            v
      Model Evaluation
            |
            v
 Feature Importance & Correlation
```

---

# Data Exploration

The dataset is loaded using Pandas:

```python
import pandas as pd

df = pd.read_csv("/content/salary.csv")
```

The project examines:

* Dataset shape
* Data types
* Missing values
* Categorical variables
* Feature structure
* Target distribution

The dataset contains:

```text
32,561 rows
15 columns
```

---

# Feature and Target Separation

The target variable is separated from the input features:

```python
x = df.drop(['salary'], axis=1)
y = df['salary']
```

The target variable is then encoded into numerical values.

---

# Categorical Feature Encoding

Categorical columns are identified using Pandas:

```python
catogarical_columns = x.select_dtypes(
    include="object"
).columns
```

`LabelEncoder` is used to convert categorical variables into numerical representations.

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in catogarical_columns:
    x[col] = le.fit_transform(x[col])

y = le.fit_transform(y)
```

This converts the categorical variables into values that can be processed by the machine learning algorithms.

---

# Train-Test Split

The dataset is divided into training and testing sets.

```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)
```

### Split

* Training: 80%
* Testing: 20%
* Random state: 42

The resulting test set contains **6,513 observations**.

---

# Feature Scaling

StandardScaler is applied for algorithms that are sensitive to feature scale.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
```

The scaled features are primarily used with KNN and SVC.

---

# Algorithm 1: K-Nearest Neighbors

The first classification algorithm explored is **K-Nearest Neighbors (KNN)**.

```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(x_train_scaled, y_train)

y_pred = knn.predict(x_test_scaled)
```

### Configuration

* Algorithm: KNN
* Number of neighbors: 5
* Scaled features: Yes

### Result

**KNN Accuracy: 83.53%**

---

# KNN with Undersampling

Random undersampling is applied to reduce the majority class and create a more balanced training set.

```python
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=42)

X_res, y_res = rus.fit_resample(
    x_train_scaled,
    y_train
)
```

A new KNN model is then trained on the resampled data.

### Result

**KNN + Undersampling Accuracy: 77.86%**

---

# KNN with Oversampling

Random oversampling is also tested.

```python
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=42)

X_res, y_res = ros.fit_resample(
    x_train_scaled,
    y_train
)
```

### Result

**KNN + Oversampling Accuracy: 77.80%**

---

# KNN with Tomek Links

Tomek Links are used as an alternative resampling technique.

```python
from imblearn.under_sampling import TomekLinks

tl = TomekLinks()

X_res, y_res = tl.fit_resample(
    x_train_scaled,
    y_train
)
```

### Result

**KNN + Tomek Links Accuracy: 83.14%**

---

# Algorithm 2: Decision Tree

A Decision Tree classifier is used as another approach to the classification problem.

```python
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(
    random_state=42
)

dt.fit(x_train, y_train)

y_pred_dt = dt.predict(x_test)
```

Decision Trees do not require feature scaling in the same way as distance-based algorithms such as KNN.

### Result

**Decision Tree Accuracy: 81.25%**

---

# Decision Tree with Undersampling

Random undersampling is applied before training another Decision Tree.

### Result

**Decision Tree + Undersampling Accuracy: 75.93%**

---

# Decision Tree with Oversampling

Random oversampling is applied to the training data.

### Result

**Decision Tree + Oversampling Accuracy: 81.10%**

---

# Decision Tree with Tomek Links

Tomek Links are also evaluated with the Decision Tree model.

### Result

**Decision Tree + Tomek Links Accuracy: 80.85%**

---

# Algorithm 3: Support Vector Classification

Support Vector Classification (SVC) is the third major classification algorithm explored.

```python
from sklearn.svm import SVC

svc = SVC()

svc.fit(x_train_scaled, y_train)

y_pred_svc = svc.predict(x_test_scaled)
```

### Result

**SVC Accuracy: 85.20%**

The notebook also generates a classification report containing:

* Precision
* Recall
* F1-score
* Support

The recorded results show:

| Class | Precision | Recall | F1-score |
| ----- | --------: | -----: | -------: |
| 0     |      0.87 |   0.95 |     0.91 |
| 1     |      0.77 |   0.56 |     0.64 |

Overall accuracy:

**85.20%**

---

# SVC with Undersampling

Random undersampling is applied before training SVC.

```python
rus = RandomUnderSampler(
    random_state=42
)

X_res, y_res = rus.fit_resample(
    x_train_scaled,
    y_train
)
```

### Result

**SVC + Undersampling Accuracy: 79.72%**

---

# Feature Selection

The project then explores whether reducing the number of features can affect SVC performance.

`SelectKBest` with the Chi-square test is used:

```python
from sklearn.feature_selection import SelectKBest, chi2

selector = SelectKBest(
    score_func=chi2,
    k=10
)
```

The selected features are:

```text
age
fnlwgt
education-num
marital-status
occupation
relationship
sex
capital-gain
capital-loss
hours-per-week
```

The selected features are then standardized before being passed to SVC.

---

# SVC with Feature Selection

An SVC model is trained using the selected features.

```python
svc_fs = SVC()

svc_fs.fit(
    X_train_fs,
    y_train
)

y_pred_fs = svc_fs.predict(X_test_fs)
```

### Result

**SVC + Feature Selection Accuracy: 85.28%**

This provides a direct comparison between the original SVC model and the feature-selected version.

---

# Feature Analysis

The project also includes feature-level analysis beyond model training.

## Chi-Square Feature Scores

Chi-square scores are calculated for all features:

```python
selector = SelectKBest(
    score_func=chi2,
    k='all'
)

selector.fit(x, y)
```

The scores are visualized using a bar chart.

This provides a statistical view of how strongly individual features are associated with the target variable.

---

# Feature Correlation Analysis

A correlation matrix is generated using Seaborn:

```python
sns.heatmap(
    x.corr(),
    cmap='coolwarm'
)
```

The resulting heatmap provides a visual representation of relationships between the encoded numerical features.

---

# Decision Tree Feature Importance

The Decision Tree model is also used to examine feature importance.

```python
importance = pd.Series(
    dt.feature_importances_,
    index=x.columns
)
```

The importance values are visualized using a bar chart.

This provides another perspective on which features contribute to the Decision Tree's classification decisions.

---

# Model Comparison

The notebook experiments with multiple algorithms and class-balancing strategies.

| Model / Technique             | Accuracy |
| ----------------------------- | -------: |
| KNN                           |   83.53% |
| KNN + Undersampling           |   77.86% |
| KNN + Oversampling            |   77.80% |
| KNN + Tomek Links             |   83.14% |
| Decision Tree                 |   81.25% |
| Decision Tree + Undersampling |   75.93% |
| Decision Tree + Oversampling  |   81.10% |
| Decision Tree + Tomek Links   |   80.85% |
| SVC                           |   85.20% |
| SVC + Undersampling           |   79.72% |
| SVC + Feature Selection       |   85.28% |

These results are reported from the notebook's recorded runs.

---

# Key Observations

The experiments demonstrate that changing the preprocessing or sampling strategy can substantially affect model performance.

The notebook shows:

* KNN achieved 83.53% accuracy using the original scaled training data.
* Decision Tree achieved 81.25% accuracy.
* SVC achieved 85.20% accuracy.
* SVC with Chi-square feature selection achieved 85.28%.
* Random undersampling reduced accuracy for the tested KNN, Decision Tree, and SVC configurations.
* Random oversampling also did not improve the tested KNN configuration.
* Tomek Links produced results close to the original models in the tested configurations.

These observations illustrate why model selection and preprocessing should be evaluated experimentally rather than assuming that a particular technique will always improve performance.

---

# Skills Demonstrated

## Data Preprocessing

* Data loading
* Dataset inspection
* Missing-value checking
* Categorical feature identification
* Label encoding
* Train-test splitting
* Feature scaling

## Machine Learning Algorithms

* K-Nearest Neighbors
* Decision Tree
* Support Vector Classification

## Imbalanced Data Techniques

* Random Undersampling
* Random Oversampling
* Tomek Links

## Feature Engineering and Selection

* Chi-square feature selection
* SelectKBest
* Feature importance
* Correlation analysis

## Model Evaluation

* Accuracy
* Precision
* Recall
* F1-score
* Classification report
* Model comparison

## Visualization

* Feature-score bar charts
* Correlation heatmap
* Feature-importance visualization

---

# Project Structure

```text
Salary-Classification-ML-Algorithms/
│
├── salary.csv
├── salary_classification.ipynb
└── README.md
```

The dataset may be excluded from the GitHub repository if required by its source or repository-size considerations.

---

# Limitations

This project is primarily an algorithm-practice and comparative machine-learning project.

The notebook uses label encoding for categorical variables. This provides a straightforward way to convert categories into numerical values for experimentation, but categorical encoding choices can affect model behavior.

The experiments also focus heavily on accuracy. Because the target classes are imbalanced, accuracy alone does not fully describe model performance. Precision, recall, F1-score, and class-specific performance should also be considered.

The models were tested using the configurations implemented in the notebook and were not subjected to extensive hyperparameter tuning or cross-validation.

---

# Future Improvements

Possible extensions include:

* Hyperparameter tuning using GridSearchCV or RandomizedSearchCV
* Cross-validation
* One-hot encoding for categorical variables
* More systematic class-imbalance handling
* ROC-AUC analysis
* Precision-recall curves
* Confusion matrices
* Additional classification algorithms
* Ensemble methods
* Logistic Regression
* Random Forest
* Gradient Boosting
* XGBoost
* Model comparison dashboard
* More comprehensive error analysis

---

# Conclusion

This project demonstrates a practical approach to experimenting with multiple machine learning classification algorithms on the same dataset.

Instead of simply training one model, the notebook investigates how **KNN, Decision Trees, and SVC** behave under different preprocessing and class-balancing strategies.

It further explores **feature selection using Chi-square statistics, feature importance from Decision Trees, and feature correlation analysis**, creating a broader understanding of the relationship between preprocessing, feature selection, class imbalance, and model performance.

The project serves as a hands-on machine learning practice project focused on understanding and comparing classification techniques using Python and Scikit-learn.

---

# Author

**Nayanendhu CU**

