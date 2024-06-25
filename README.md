# Credit Risk Analysis
In this project, credit risk analysis was performed, 
involving data cleaning, data exploration, and detailed data analysis. 
Three machine learning models (LightGBM, XGBoost, and CatBoost) were used to compare which model provided the most accurate prediction results.

summary of the code's operations:

1. Data Loading and Cleaning:
    - Load data from a CSV file.
    - Drop rows with null values and duplicates.
    - Drop rows where person_age is above 84.

2. Data Exploration and Visualization:
    - Plot distributions of person_home_ownership, loan_grade, loan_intent, loan_status, and cb_person_default_on_file.
    - Compare loan status with age, home ownership, loan grade, loan intent, and historical default.

3. Categorical Encoding:
    - Encode categorical features using LabelEncoder and OneHotEncoder.
    - Split data into training and testing sets.
    - Scale features using StandardScaler.

4. Model Training and Evaluation:
    - Train and evaluate LGBMClassifier, XGBClassifier, and CatBoostClassifier.
    - Print classification reports, accuracy, precision, recall, and confusion matrices for each model.
    - Plot feature importances from CatBoost.

5. Feature Importance Analysis:
    - Plot the feature importances for the CatBoost model.


## Conclusion
In the overview of the data, it can be seen that most people have the highest debt from education, followed by medical expenses.

From the graph comparing loan status and age, younger people tend to have more debt than older people and are more likely to default on their debts.

From the graph comparing loan status and loan intent, it was found that people who defaulted on their debts often had the highest medical expenses. This may be caused by emergencies requiring them to borrow to pay.

From the model training, 
it was found that using the CatBoost Classifier model gives the highest accuracy in predicting those who may default on their debts, 
with an accuracy of 0.933 or 93.3%. In comparison, XGBoost achieved an accuracy of 0.930, and LightGBM achieved 0.929.

The use of a forest of trees to evaluate the importance of features shows 
that the factors affecting customer default are the percentage of income represented by the loan amount, 
the type of home ownership, and the individual's income.
