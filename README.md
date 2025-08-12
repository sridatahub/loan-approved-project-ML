Loan Prediction Project

Author: Sri (you can change this)

Summary: This repository contains a reproducible machine-learning project that predicts whether a loan application will be approved (Loan_Status) using a structured tabular dataset. The project includes data exploration, preprocessing, model training (Support Vector Machine), evaluation, and a simple predictive interface.
Table of Contents

Problem Statement

Dataset Overview

Exploratory Data Analysis & Findings

Data Cleaning & Preprocessing

Feature / Label Separation

Train / Test Split

Modeling (Support Vector Machine)

Evaluation & Results

Predictive System (Usage)

Reproducibility & Saving Model

Limitations, Improvements & Next Steps

How to run this project locally

Requirements

License & Contact

Problem Statement

Given applicant information (gender, marital status, income, loan amount, credit history, etc.), predict whether a loan will be approved (Y) or not (N). This is a binary classification problem.

Dataset Overview

Original dataset shape: 614 rows × 13 columns.

Columns (as in your notebook):

Loan_ID, Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area, Loan_Status.

Key summary statistics (numerical features)

ApplicantIncome: mean = 5403.459283, std = 6109.041673, min = 150, max = 81000.

CoapplicantIncome: mean = 1621.245798, std = 2926.248369.

LoanAmount: mean = 146.412162, std = 85.587325, count = 592 (22 missing originally).

Loan_Amount_Term: mean = 342.0 (median 360.0), some values like 12, 480 exist.

Credit_History: mean = 0.842199, count = 564 (50 missing originally).

Missing values before cleaning (per your run):
