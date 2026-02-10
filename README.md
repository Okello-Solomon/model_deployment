# 02. Loan Payback Prediction App

**Live Application**

[https://your-streamlit-app-link-here](https://predictstudenttestscores-zricng54xnwma5r6qhajqt.streamlit.app/)

Use the link above to interact with the application and predict whether a loan is likely to be repaid or defaulted in real time.

<details>
<summary><strong> View Project Details</strong></summary>
  
## Project Overview

This project implements a Loan Payback Prediction Model using XGBoost, a powerful gradient boosting algorithm, to predict whether a borrower is likely to repay a loan or default.

The model is designed to support data-driven credit risk assessment by leveraging borrower demographics, financial indicators, and loan characteristics.

The solution follows an end-to-end machine learning workflow including data preprocessing, feature engineering, model training, evaluation, and deployment readiness.

## Problem Statement
Financial institutions face significant risk when issuing loans without accurately assessing repayment likelihood.
This project aims to:
- Predict loan repayment outcomes with high accuracy
- Reduce default risk
- Support consistent and explainable lending decisions

## Features

Predicts loan repayment or default in real time

Handles categorical, binary, and numerical features

Uses robust dummy variable encoding for categorical inputs

Applies consistent preprocessing during training and inference

Manages class imbalance for improved default detection

Interactive and user-friendly Streamlit interface

## Machine Learning Model

**Model Used:** XGBoost Classifier

**Objective:** Predict loan repayment outcome

- 1 -> Loan repaid

- 0 -> Loan defaulted

**Evaluation Metrics:**

- Accuracy

- Precision

- Recall

- F1-Score

**Validation Strategy:**

- Train-test split

- Cross-validation

- Hyperparameter tuning

## XGBoost was selected due to its:##

- High predictive performance on structured tabular data

- Ability to model non-linear relationships

- Built-in regularization to reduce overfitting

## Deployment

The trained model is deployed using Streamlit, allowing users to input loan and borrower details and receive instant predictions.
Feature alignment between training and deployment is strictly enforced to ensure reliable and consistent predictions.

## Dataset Access

The dataset used for this study can be accessed here:

[Insert dataset link here](kaggle competitions download -c playground-series-s5e11)

<details>

  
- Robust handling of class imbalance

