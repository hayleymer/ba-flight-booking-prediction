# Flight Booking Prediction Model

This project builds a machine learning model to predict whether a customer will complete a flight booking.

## Business Problem

Airlines need to understand which customers are likely to convert in order to optimise targeting, pricing strategy, and revenue planning.

## Business value
Allows airlines to target high-probability customers and improve conversion rates.

## Result
- Model achieved PR-AUC of 0.35 on imbalanced dataset.
- Identified booking origin, route, and baggage selection as strongest predictors.

## Approach

- Data cleaning and preprocessing
- Feature engineering
- Hyperparameter optimisation (Optuna)
- Model comparison
- Evaluation using ROC AUC and PR AUC
- Feature importance analysis

## Key Insights

- Route and booking origin were strong predictors of booking completion.
- Customers selecting add-ons (baggage, seat preference) were more likely to convert.
- Random Forest performed best overall.

## Tools Used

- Python
- Pandas
- Scikit-learn
- Optuna
- SHAP
