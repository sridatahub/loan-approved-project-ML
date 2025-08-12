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

## 🔍 **Step 1: Displaying the First 5 Rows**
We start by loading the dataset and viewing the first few entries.

```python
# printing the first 5 rows of the dataframe
loan_dataset.head()
📏 Step 2: Shape of the Dataset
We check how many rows and columns the dataset contains.

python
Copy
Edit
# number of rows and columns
loan_dataset.shape
Output: (614, 13) → 614 rows, 13 columns

📊 Step 3: Statistical Summary
We get the basic statistical measures (mean, median, standard deviation, etc.) for numerical columns.

python
Copy
Edit
# statistical measures
loan_dataset.describe()
Findings:

ApplicantIncome ranges from 150 to 81,000.

LoanAmount mean ≈ 146, with some very high outliers (max 700).

Credit_History mean ≈ 0.84 → most applicants have good credit history.

🛠 Step 4: Checking Missing Values
We inspect the dataset for null values.

python
Copy
Edit
# number of missing values in each column
loan_dataset.isnull().sum()
Finding: Several columns like Gender, Dependents, LoanAmount, Credit_History have missing data.

🗑 Step 5: Removing Missing Values
We remove rows with null values for cleaner processing.

python
Copy
Edit
# dropping the missing values
loan_dataset = loan_dataset.dropna()
🏷 Step 6: Encoding Loan Status
We convert Loan_Status from categorical (Y/N) to numerical (1/0).

python
Copy
Edit
# label encoding
loan_dataset.replace({"Loan_Status": {'N': 0, 'Y': 1}}, inplace=True)
📈 Step 7: Handling 'Dependents' Column
We replace "3+" with numeric 4 for better numerical processing.

python
Copy
Edit
# replacing the value of 3+ to 4
loan_dataset = loan_dataset.replace(to_replace='3+', value=4)
📊 Step 8: Data Visualization
We explore patterns between features and loan approval.

python
Copy
Edit
# education & Loan Status
sns.countplot(x='Education', hue='Loan_Status', data=loan_dataset)

# marital status & Loan Status
sns.countplot(x='Married', hue='Loan_Status', data=loan_dataset)
Findings:

Graduates have higher loan approval rates.

Married applicants also tend to get approved more often.

🔢 Step 9: Encoding Categorical Columns
We convert all text-based categorical columns into numeric values.

python
Copy
Edit
loan_dataset.replace({
    'Married': {'No': 0, 'Yes': 1},
    'Gender': {'Male': 1, 'Female': 0},
    'Self_Employed': {'No': 0, 'Yes': 1},
    'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2},
    'Education': {'Graduate': 1, 'Not Graduate': 0}
}, inplace=True)
📤 Step 10: Feature & Target Separation
We separate the independent variables X and dependent variable Y.

python
Copy
Edit
X = loan_dataset.drop(columns=['Loan_ID', 'Loan_Status'], axis=1)
Y = loan_dataset['Loan_Status']
✂ Step 11: Train-Test Split
We split the dataset into training and testing sets (90% training, 10% testing).

python
Copy
Edit
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.1, stratify=Y, random_state=2
)
🤖 Step 12: Training the SVM Model
We train a Support Vector Machine classifier with a linear kernel.

python
Copy
Edit
from sklearn import svm

classifier = svm.SVC(kernel='linear')

# training the model
classifier.fit(X_train, Y_train)
📊 Step 13: Model Evaluation
We check accuracy on both training and testing datasets.

python
Copy
Edit
from sklearn.metrics import accuracy_score

# accuracy score on training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

# accuracy score on test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
Results:

Training Accuracy: ~79.86%

Testing Accuracy: ~83.33%

✅ The model generalizes well with minimal overfitting.

🧮 Step 14: Making a Prediction
We predict loan approval for a single applicant.

python
Copy
Edit
import numpy as np

input_data = (1, 1, 1, 1, 0, 4583, 1508.0, 128.0, 360.0, 1.0, 0)
input_array = np.asarray(input_data)

# reshaping data
reshaped_data = input_array.reshape(1, -1)
result = classifier.predict(reshaped_data)

if result[0] == 1:
    print("Person got loan")
else:
    print("Loan Not Approved")
Output:

nginx
Copy
Edit
Person got loan
🏁 Conclusion
This SVM-based model can predict loan approvals with ~83% accuracy.

Key influencing factors:

Credit History

Education

Applicant Income

Marital Status

Possible improvements:

Feature scaling

Hyperparameter tuning

Trying other algorithms (Random Forest, XGBoost)

