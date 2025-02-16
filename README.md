# Credit Score Model
Credit scoring is the process of assessing the creditworthiness of an individual or business. This process is used by lenders to determine whether to approve a loan application and what interest rate to charge.
This project aims to build a machine learning-based credit score prediction model to help lenders make informed decisions about loan approvals. The model analyzes various financial and behavioral features to predict a person's creditworthiness.

# Objective
The objective of this project is to develop a machine learning/deep learning model that can accurately predict credit scores based on financial history and behavioral attributes. This will enable financial institutions to:
Assess loan eligibility with greater precision.
Reduce default risks by identifying high-risk customers.
Automate the credit evaluation process.
# Technologies Used

  Python
  Flask (for web interface)
  Pandas & NumPy (for data processing)
  Scikit-learn (for machine learning model)
  Bootstrap (for frontend styling)
  Pickle (for model serialization)

# Dataset Description

The dataset contains financial and behavioral attributes of individuals, which are used as features for credit score prediction.
ID	Unique 
Customer_ID	
Month	
Name	
Age
SSN	Social
Occupation	
Annual_Income	
Monthly_Inhand_Salary
Num_Bank_Accounts	
Num_Credit_Card	
Interest_Rate
Num_of_Loan	
Type_of_Loan	
Delay_from_due_date	
Num_of_Delayed_Payment	
Changed_Credit_Limit	
Num_Credit_Inquiries	
Credit_Mix
Outstanding_Debt	
Total_EMI	Total 
Credit_Age_Years
Payment_Behaviour	
Payment_Min_Amount	
Credit_Score	
# Model Details
-Algorithm Used: 
Random Forest Classifier

- Preprocessing:
Categorical encoding of Credit Mix, Payment Behavior, and Payment Minimum Amount.
Feature Scaling using StandardScaler to normalize numerical data.

# Prediction Categories:
0 → Good

1 → Poor

2 → Standard

# How It Works
User Input: The user enters their financial details into a web form.
# Data Preprocessing:

Numerical values are standardized.
Categorical values are converted into numerical labels.
# Model Prediction:

The trained Random Forest Classifier predicts the credit score category.
# Result Display:

The predicted category (Good, Standard, Poor) is displayed on a result page.

# Run the Application
     python app.py
# Screenshots:
![Screenshot (621)](https://github.com/user-attachments/assets/f62fba93-f9de-43bb-9a08-a71da10f22d2)
![Screenshot (620)](https://github.com/user-attachments/assets/0c01347b-2fef-46f6-b866-3f66000f4ec3)
