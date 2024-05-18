# Predictive_Modeling-_or_Disease_Diagnosis
sarahRefaatt/Predictive_Modeling-_or_Disease_Diagnosis using logistics and svm models

# Classification Model Evaluation

This code snippet demonstrates the evaluation of classification models using various algorithms such as Support Vector Machine (SVM), Logistic Regression, and K-Nearest Neighbors (KNN). It also includes the visualization of a confusion matrix.

## Prerequisites

Make sure you have the following libraries installed:

- `pandas`
- `sklearn`
- `numpy`
- `matplotlib`
- `seaborn`

You can install these libraries using `pip`:

```
pip install pandas scikit-learn numpy matplotlib seaborn
```

## Usage

1. Import the required libraries:

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

2. Read the training and test data from CSV files:

```python
train = pd.read_csv("path/to/train_data.csv")
test = pd.read_csv("path/to/test_data.csv")
```

Make sure to replace `"path/to/train_data.csv"` and `"path/to/test_data.csv"` with the actual paths to your data files.

3. Define a preprocessing function to prepare the data:

```python
def preprocess(data):
    x = data.drop(columns=["Disease"])
    y = data["Disease"]
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    labels = y.unique()

    label_mapping = {label: 0 if label == 'Healthy' else 1 for label in labels}

    y = y.map(label_mapping)

    return x, y
```

This function takes the data as input, separates the features (`x`) and the target variable (`y`), performs feature scaling using `StandardScaler`, and maps the target variable labels to binary values (0 and 1).

4. Preprocess the training and test data:

```python
x, y = preprocess(train)
x_test, y_test = preprocess(test)
```

5. Fit and evaluate the SVM model:

```python
svm = SVC(kernel='rbf', random_state=42)
svm.fit(x, y)
y_pred = svm.predict(x_test)
svm_accuracy = accuracy_score(y_test, y_pred)
print("SVM Accuracy:", svm_accuracy)
```

6. Fit and evaluate the Logistic Regression model:

```python
logistic_classifier = LogisticRegression(random_state=42)
logistic_classifier.fit(x, y)
y_pred = logistic_classifier.predict(x_test)
logistic_accuracy = accuracy_score(y_test, y_pred)
print("Logistic Regression Accuracy:", logistic_accuracy)
```

7. Fit and evaluate the KNN model:

```python
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x, y)
y_pred = knn.predict(x_test)
knn_accuracy = accuracy_score(y_test, y_pred)
print("KNN Accuracy:", knn_accuracy)
```

8. Plot the confusion matrix:

```python
cm = confusion_matrix(y_test, y_pred)
classes = np.unique(y_test)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=classes, yticklabels=classes)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("Actual Labels")
plt.show()
```

This code calculates the confusion matrix using `confusion_matrix` from `sklearn.metrics`, creates labels for the classes, and plots the confusion matrix as a heatmap using `seaborn` and `matplotlib.pyplot`.

## Example

You can replace the paths to the training and test data files with the actual paths on your system. Make sure the CSV files are properly formatted, and the column names match the code.

```python
train = pd.read_csv("C:/Users/Sarah/OneDrive/Desktop/Mentroness_Internship/Train_data.csv")
test = pd.read_csv("C:/Users/Sarah/OneDrive/Desktop/Mentroness_Internship/test_data.csv")
```

