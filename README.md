# Loan-Risk-Prediction-Ml-Project
End-to-end Machine Learning project for loan default prediction using Random Forest and Logistic Regression, deployed with a Flask web application.

# Project Overview
This project builds a Machine Learning model to predict the risk of loan default using historical customer financial data. The goal is to help financial institutions identify customers who may fail to repay loans.
The system performs data preprocessing, feature engineering, model training, evaluation, and deployment using a Flask web application where users can input customer details and get predictions.

# The main objective of this project is to:
Predict whether a customer will repay the loan or default
Help financial institutions reduce financial risk
Build a complete end-to-end ML pipeline
Deploy the model using Flask for real-time prediction

# Technologies Used
Python, Pandas, NumPy, Scikit-learn, Flask, HTML (Frontend),Pickle (Model saving)

# Data Preprocessing
The following preprocessing steps were applied:
1️⃣ Handling Missing Values
Median imputation for numerical columns
Missing categorical values filled with "Unknown"
2️⃣ Categorical Encoding
Applied Label Encoding for categorical variables
3️⃣ Feature Scaling
Used StandardScaler to normalize numerical features
4️⃣ Train-Test Split
Dataset split:
80% Training Data
20% Testing Data

# Machine Learning Models Used

Two classification models were trained:
1️⃣ Random Forest Classifier
Ensemble learning method
Handles non-linear relationships well
Works well with tabular data

2️⃣ Logistic Regression
Linear classification algorithm
Good baseline modelFast and interpretable

# 📈 Model Evaluation
The models were evaluated using:
Accuracy Score, Confusion Matrix

# 🌐 Model Deployment
The trained models were deployed using Flask.
The web application allows users to:
Enter customer financial details
Select the prediction model
Receive loan risk prediction
