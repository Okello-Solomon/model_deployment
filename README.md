# 02. Loan Payback Prediction App

**Live Application**

[https://your-streamlit-app-link-here](https://predictstudenttestscores-zricng54xnwma5r6qhajqt.streamlit.app/)

Use the link above to interact with the application and predict whether a loan is likely to be repaid or defaulted in real time.

<details>
<summary><strong> View Project Details</strong></summary>
  
## Project Overview

This project predicts loan payback outcomes based on borrower characteristics, financial indicators, and loan details using an XGBoost classification model.
The model is trained, evaluated, and deployed through an interactive Streamlit web application to support data-driven credit risk assessment.

The project demonstrates a complete end-to-end machine learning workflow:

data preprocessing → feature engineering → model training → model evaluation → deployment

Features

Predicts loan repayment or default in real time

Handles categorical, binary, and numerical features

Uses robust dummy variable encoding for categorical inputs

Applies consistent preprocessing during training and inference

Manages class imbalance for improved default detection

Interactive and user-friendly Streamlit interface

Machine Learning Model

Model Used: XGBoost Classifier

Objective: Predict loan repayment outcome

1 → Loan repaid

0 → Loan defaulted

Evaluation Metrics:

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

Validation Strategy:

Train-test split

Cross-validation

Hyperparameter tuning

XGBoost was selected due to its:

High predictive performance on structured tabular data

Ability to model non-linear relationships

Built-in regularization to reduce overfitting

<details>

Robust handling of class imbalance

Deployment

The trained model is deployed using Streamlit, allowing users to input loan and borrower details and receive instant predictions.
Feature alignment between training and deployment is strictly enforced to ensure reliable and consistent predictions.
