# Credit Card Fraud Detection

## Project Overview

Credit card fraud detection presents a significant challenge due to the imbalanced nature of transaction datasets,
where fraudulent transactions are much less common than legitimate ones. This project tackles this challenge head-on,
employing a combination of data preprocessing, oversampling techniques, and a carefully architected neural network to effectively identify fraudulent transactions.

## Features
- Data Preprocessing: Standardization of features and handling of imbalanced dataset through SMOTE oversampling.
- Deep Learning Model: A Sequential model comprising dense layers, utilizing ReLU activation, and a final sigmoid activation layer for binary classification.
- Performance Metrics: Evaluation based on accuracy, precision, and recall, achieving 100% fraud detection rate.
- Visualization: Learning curves and confusion matrix for model performance analysis.

## Results
The model has achieved remarkable accuracy in identifying fraudulent transactions within the credit card dataset. It correctly classified all 492 cases of fraud and successfully recognized 284,269 transactions as legitimate. However, it misclassified 46 legitimate transactions as fraudulent.

![Confusion Matrix](Credit_card_fraud_1710357529.png)
![Classification Report](Classification_Report_1710357529.png)

Dataset from Kaggle's Credit Card Fraud Detection challenge.
