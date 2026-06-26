# Zyphra — Loan Approval Prediction System
An intelligent, end-to-end machine learning web application that predicts loan approval outcomes using a Decision Tree classifier. Built by Team Zyphra as an academic project to demonstrate real-world AI applications in financial decision-making.

🎯 Project Overview

Zyphra leverages a Decision Tree model trained on 45,000+ real loan records to predict whether a loan application will be approved or rejected. The system analyzes 13 financial parameters (age, income, credit score, loan amount, etc.) and returns a prediction with a confidence score in under 1 second.

# URL to Website
https://loan-prediction-bhev.onrender.com

# Key Highlights


89.65% Accuracy — Decision Tree model evaluated on a held-out test set of 9,000+ records

13 Features — Age, Gender, Education, Income, Experience, Home Ownership, Loan Amount, Loan Intent, Interest Rate, Loan-to-Income %, Credit History, Credit Score, Previous Loan Default

Business Rules — Applicants under 18 or above 70 are automatically rejected (enforced in backend)

Full Stack — Data cleaning → Model training → Flask API → Multi-page web interface

Transparent & Interpretable — Feature importance analysis shows which factors matter most


Top Feature Importances


1.Previous Loan Default — 42%

2.Loan Interest Rate — 18%

3.Loan-to-Income % — 16%

4.Credit Score — 10%

5.Annual Income — 7%

6.Other Features — 7%

📈 ML Pipeline

The project follows a complete data science workflow:


1.Data Collection — 45,001 loan records with 13 features and binary target

2.Data Cleaning — Null removal, age filtering (18–70), categorical encoding

3.Model Training — 80/20 stratified train-test split, DecisionTreeClassifier

4.Evaluation — Accuracy, classification report, confusion matrix, feature importance

5.Deployment — Flask REST API serving real-time predictions to web interface


