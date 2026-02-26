# British Airways Flight Booking Prediction

## Business Problem

Airlines invest heavily in marketing and customer acquisition, but not all search and booking interactions convert into completed bookings. Being able to predict whether a customer will complete a booking allows airlines to better target high-probability customers, optimise pricing strategies, and improve conversion rates.

This project develops a machine learning model to predict whether a customer completes a flight booking using behavioural, temporal, and route-level features.

The objective is to determine whether customer behaviour patterns, booking characteristics, and historical route-level conversion rates can reliably predict booking completion.

---

## Dataset

The dataset contains customer booking records from British Airways, with:

- 50,000 customer booking observations  
- 14 original variables including behavioural, temporal, and categorical features  
- Binary target variable indicating whether the booking was completed  

### Key predictor variables

Behavioural features:

- Purchase lead time  
- Length of stay  
- Flight duration  
- Flight hour  
- Number of passengers  

Customer preference indicators:

- Wants extra baggage  
- Wants preferred seat  
- Wants in-flight meals  

Categorical features:

- Sales channel  
- Trip type  
- Flight day  
- Booking origin  
- Route  

### Target variable

- booking_complete  
  - 0 = booking not completed  
  - 1 = booking completed  

The dataset is imbalanced, with approximately 15% completed bookings. :contentReference[oaicite:1]{index=1}

---

## Approach

### Data Preparation

- Loaded and validated dataset structure and completeness  
- Identified categorical, numeric, and binary feature groups  
- Split dataset into training and test sets using stratified sampling  

### Feature Engineering

Custom feature engineering was implemented using a scikit-learn transformer to create predictive statistical and behavioural features.

Engineered features included:

Behavioural interaction features:

- Lead time × length of stay  
- Lead time ÷ length of stay  

Statistical conversion-rate features:

- Route-level booking rate  
- Booking origin conversion rate  
- Sales channel conversion rate  
- Trip type conversion rate  
- Origin-route interaction conversion rate  

These features capture historical conversion likelihood and behavioural interaction effects.

### Preprocessing Pipeline

A full preprocessing pipeline was built using ColumnTransformer and Pipeline:

Numeric features:

- Median imputation  
- Standard scaling  

Categorical features:

- Most-frequent imputation  
- One-hot encoding  

Binary and statistical features:

- Passed through without transformation  

This ensures reproducible and leakage-safe preprocessing.

### Model Development

A Random Forest Classifier was used as the primary model due to its ability to capture nonlinear relationships and interactions.

Baseline model performance:

- ROC-AUC: 0.788 :contentReference[oaicite:2]{index=2}

Hyperparameter optimisation was performed using:

- RandomizedSearchCV  
- Bayesian optimisation using Optuna  

Best cross-validated ROC-AUC achieved:

- 0.840 (cross-validation) :contentReference[oaicite:3]{index=3}

---

## Results

Optimised Random Forest model performance on test data:

- ROC-AUC: 0.789  
- Accuracy: 70%  
- Recall (completed bookings): 74%  
- Precision (completed bookings): 30% :contentReference[oaicite:4]{index=4}

Confusion matrix results:

- True positives correctly identified at high rate  
- Model successfully detects completed bookings despite class imbalance  

The model demonstrates strong discrimination capability between completed and incomplete bookings.

---

## Key Findings

Feature importance analysis revealed that the strongest predictors of booking completion were:

- Route-level booking rate  
- Origin-route booking rate  
- Purchase lead time  
- Length of stay  
- Sales channel conversion rate  

These findings indicate that:

- Historical route conversion behaviour is highly predictive  
- Customer behavioural timing patterns strongly influence conversion  
- Route-specific and origin-specific behavioural signals are critical predictors
  
<img width="890" height="528" alt="screenshot" src="https://github.com/user-attachments/assets/9006512e-3436-4173-a938-54a7c2015439" />

The engineered statistical features significantly improved model performance beyond raw booking attributes.

---

## Tech Stack

Python  
pandas  
numpy  
scikit-learn  
matplotlib  
Optuna  

### Machine Learning Techniques

- Supervised classification  
- Feature engineering with custom transformers  
- Pipeline-based preprocessing  
- One-hot encoding  
- Hyperparameter optimisation (RandomizedSearchCV, Optuna)  
- Cross-validation  
- ROC-AUC evaluation  
- Feature importance analysis  

---

## How to Run

### Clone repository

```bash
git clone https://github.com/hayleymer/ba-flight-booking-prediction.git
```

### Navigate to project folder

```bash
cd ba-flight-booking-prediction
```

### Install dependencies

```bash
pip install -r requirements.txt
```

Run the notebook to reproduce the analysis and model results.
