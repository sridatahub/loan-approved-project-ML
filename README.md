# 🏦 **Loan Approval Prediction Using Support Vector Machine (SVM)**

This project predicts whether a loan application will be approved based on applicant details using **Machine Learning (SVM)**.  
We go through **data preprocessing**, **exploratory data analysis**, and **Support Vector Machine classification** to build the model.

---

## 📂 **Dataset Overview**
The dataset contains **614 rows** and **13 columns** with information like applicant income, education level, credit history, and more.

**Target variable**:
- `Y` → Loan Approved
- `N` → Loan Not Approved

---
# ===========================================
# 📌 Step 1: Import Required Libraries
# ===========================================
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
```
# ===========================================
# 📌 Step 2: Load Dataset
# ===========================================
```python
loan_dataset = pd.read_csv("loan_dataset.csv")
```
# Display first 5 rows
```python
print(loan_dataset.head())
```
# Shape of dataset
```python
print("Dataset Shape:", loan_dataset.shape)
```
# Statistical summary
```python
print(loan_dataset.describe())
```
# ===========================================
# 📌 Step 3: Missing Values
# ===========================================
```python
print("Missing values per column:\n", loan_dataset.isnull().sum())
```
# Drop rows with missing values
```python
loan_dataset = loan_dataset.dropna()
```
# ===========================================
# 📌 Step 4: Encode Target Column
# ===========================================
```python
loan_dataset.replace({"Loan_Status": {'N': 0, 'Y': 1}}, inplace=True)
```
# Replace '3+' in Dependents with 4
```python
loan_dataset = loan_dataset.replace(to_replace='3+', value=4)
```
# ===========================================
# 📌 Step 5: Data Visualization
# ===========================================
```python
sns.countplot(x='Education', hue='Loan_Status', data=loan_dataset)
plt.show()

sns.countplot(x='Married', hue='Loan_Status', data=loan_dataset)
plt.show()
```
# ===========================================
# 📌 Step 6: Encode Categorical Columns
# ===========================================
```python
loan_dataset.replace({
    'Married': {'No': 0, 'Yes': 1},
    'Gender': {'Male': 1, 'Female': 0},
    'Self_Employed': {'No': 0, 'Yes': 1},
    'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2},
    'Education': {'Graduate': 1, 'Not Graduate': 0}
}, inplace=True)
```
# ===========================================
# 📌 Step 7: Feature & Target Separation
# ===========================================
```python
X = loan_dataset.drop(columns=['Loan_ID', 'Loan_Status'], axis=1)
Y = loan_dataset['Loan_Status']
```
# ===========================================
# 📌 Step 8: Train-Test Split
# ===========================================
```python
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.1, stratify=Y, random_state=2
)
```
# ===========================================
# 📌 Step 9: Train SVM Model
# ===========================================
```python
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)
```
# ===========================================
# 📌 Step 10: Model Evaluation
# ===========================================
# Accuracy on training data
```python
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print("Training Accuracy:", training_data_accuracy)
```
# Accuracy on testing data
```python
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print("Testing Accuracy:", test_data_accuracy)
```
# ===========================================
# 📌 Step 11: Single Prediction Example
# ===========================================
input_data = (1, 1, 1, 1, 0, 4583, 1508.0, 128.0, 360.0, 1.0, 0)

# Convert input data to numpy array
input_array = np.asarray(input_data).reshape(1, -1)

# Make prediction
```python
prediction = classifier.predict(input_array)

if prediction[0] == 1:
    print("Person got loan")
else:
    print("Loan Not Approved")

```

🏁 Conclusion
This SVM-based model can predict loan approvals with **~83%** accuracy.

**Key influencing factors:**

Credit History

Education

Applicant Income

Marital Status

Possible improvements:

Feature scaling

Hyperparameter tuning

Trying other algorithms (Random Forest, XGBoost)

