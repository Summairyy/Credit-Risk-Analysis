import pandas as pd
print(pd.__version__)
pd.DataFrame.iteritems = pd.DataFrame.items
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

data = pd.read_csv(r"credit_risk_dataset.csv")
data.head()
data.columns

data.info()
# Check NULL Value
data.isnull().sum()
# Drop NULL Value
data = data.dropna()
# Check duplicates
data_dups = data.duplicated()
data_dups.value_counts() #There are 137 Duplicated rows
data = data.drop_duplicates()
data.shape #(28501, 12)

data.describe()
correlation = data.corr(numeric_only=True)
correlation.round(2);
fig, ax = plt.subplots(figsize=(10,15))
sns.heatmap(correlation, square = True,annot=True)
plt.title('Confusion Matrix',fontsize=15);
plt.show()

data["person_age"].value_counts()
base_credit = data[data["person_age"]<=84]
base_credit["person_age"].max()
# Droping age above 84
data = data.drop(data[data["person_age"] > 84].index, axis=0)

# Data Exploration
pho_count = data["person_home_ownership"].value_counts()
fig1, ax1 = plt.subplots()
ax1.pie(pho_count, labels=pho_count.index, autopct='%1.1f%%')
ax1.set_title("Type of home ownership")
plt.show()

grade_count = data["loan_grade"].value_counts()
explode = (0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01)
fig2, ax2 = plt.subplots()
ax2.pie(grade_count, labels=grade_count.index, autopct='%1.1f%%', explode=explode)
ax2.set_title("Loan Grade")
plt.show()

intent_count = data["loan_intent"].value_counts()
intent_per = [f'({v}%)' for v in np.round(intent_count.values/intent_count.values.sum()*100,2)]
fig3, ax3 = plt.subplots(figsize=[15,10])
barplot = ax3.bar(intent_count.index, intent_count.values, fc="#BCDC83")
ax3.bar_label(barplot, labels=intent_per, label_type="edge")
ax3.set_ylabel("Number of loan applications")
ax3.set_title("The intent behind the loan application.")
plt.show()

#0: Non-default - The borrower successfully repaid the loan as agreed, and there was no default.
#1: Default - The borrower failed to repay the loan according to the agreed-upon terms and defaulted on the loan.
status_count = data["loan_status"].value_counts()
fig4, ax4 = plt.subplots()
ax4.pie(status_count, labels=status_count.index, autopct='%1.1f%%')
ax4.set_title("Loan Status")
plt.show()

#Y: The individual has a history of defaults on their credit file.
#N: The individual does not have any history of defaults.
historical_default = data["cb_person_default_on_file"].value_counts()
fig5, ax5 = plt.subplots()
ax5.pie(historical_default, labels=historical_default.index, autopct='%1.1f%%')
ax5.set_title("Historical default of the individual")
plt.show()

## Analyst data 
non_dfl = data[data["loan_status"]==0]
dfl = data[data["loan_status"]==1]

# Compare between Loan Status and Age
plt.figure(figsize=[15,8])
sns.countplot(x = "person_age", hue="loan_status", data=data)
plt.show()

# Compare between Loan Status and person_home_ownership
### Default
fig1 = px.histogram(dfl,x="person_home_ownership", color="person_home_ownership")
fig1.show()
### Non default
fig1 = px.histogram(non_dfl,x="person_home_ownership", color="person_home_ownership")
fig1.show()

# Compare between Loan Status and loan_grade
### Default
fig2 = px.histogram(dfl,x="loan_grade", color="loan_grade")
fig2.show()
### Non default
fig2 = px.histogram(non_dfl,x="loan_grade", color="loan_grade")
fig2.show()

# Compare between Loan Status and loan_intent
### Default
fig3 = px.histogram(dfl,x="loan_intent", color="loan_intent")
fig3.show()
### Non default
fig3 = px.histogram(non_dfl,x="loan_intent", color="loan_intent")
fig3.show()

# Compare between Loan Status and cb_person_default_on_file
### Default
fig4 = px.histogram(dfl,x="cb_person_default_on_file", color="cb_person_default_on_file")
fig4.show()
### Non default
fig4 = px.histogram(non_dfl,x="cb_person_default_on_file", color="cb_person_default_on_file")
fig4.show()

sns.pairplot(data,hue="loan_status")
plt.show()
print("-----")

#=================================================================================================
# Categorical Encoding
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
data.info()
df_catagories = data.select_dtypes(include=[object])
df_catagories.info()

Y = data['loan_status']
X = data.drop(columns=["loan_status"])
X.columns
X = X.values

## Create a LabelEncoder object and fit it to each feature in X
label_encoder_pho = LabelEncoder()
label_encoder_loan_intent = LabelEncoder()
label_encoder_loan_grade = LabelEncoder()
label_encoder_cpdf= LabelEncoder()

X[:,2] = label_encoder_pho.fit_transform(X[:,2])
X[:,4] = label_encoder_loan_intent.fit_transform(X[:,4])
X[:,5] = label_encoder_loan_grade.fit_transform(X[:,5])
X[:,9] = label_encoder_cpdf.fit_transform(X[:,9])

## Create a OneHotEncoder object, and fit it to all of X
onehot_X = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [0,2,4,5,9,10])], remainder='passthrough')
X = onehot_X.fit_transform(X).toarray()
X.shape

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=12)
print(x_train.shape,x_test.shape)

## Apply scaling using StandardScaler class (fit_transform)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_test

# Machine Learning Model
import lightgbm as lgb
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, accuracy_score, recall_score, confusion_matrix

## LGBMClassifier : Light Gradient Boosting Machine Classifier
lgb = lgb.LGBMClassifier()
lgb.fit(x_train,y_train)
predic_lgb = lgb.predict(x_test)
predic_lgb
print(classification_report(y_test, predic_lgb))

accuracy_lgb = accuracy_score(y_test, predic_lgb)
precision_lgb = precision_score(y_test, predic_lgb)
recall_lgb = recall_score(y_test, predic_lgb)
print(f"Accuracy Lightgbm: {accuracy_lgb}")
print(f"Precision Lightgbm: {precision_lgb}")
print(f"Recall Lightgbm: {recall_lgb}")
print("Confusion Matrix Lightgbm:")
print(confusion_matrix(y_test, predic_lgb))
print("Lightgbm Classification Report:")
print(classification_report(y_test, predic_lgb))


## XGBClassifier
xgboost = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgboost.fit(x_train, y_train)
xgboost.score(x_train, y_train)
xgboost.score(x_test, y_test)
predic_xgb = xgboost.predict(x_test)
predic_xgb
y_test

accuracy_xgb = accuracy_score(y_test, predic_xgb)
precision_xgb = precision_score(y_test, predic_xgb)
recall_xgb = recall_score(y_test, predic_xgb)
print(f"Accuracy XGBoost: {accuracy_xgb}")
print(f"Precision XGBoost: {precision_xgb}")
print(f"Recall XGBoost: {recall_xgb}")
print("Confusion Matrix XGBoost:")
print(confusion_matrix(y_test, predic_xgb))
print("XGBoost Classification Report:")
print(classification_report(y_test, predic_xgb))

#==================Feature Importance======================
cat_feature_importances = catboost.feature_importances_
sorted_idx = cat_feature_importances.argsort()
feature_name = data.drop(columns=["loan_status"])
feature_name = feature_name.columns
y_ticks = np.arange(0, len(feature_name))
y_ticks
feature_name
cat_feature_importances[sorted_idx]
fig_fi, ax_fi = plt.subplots()
ax_fi.barh(y_ticks, cat_feature_importances[sorted_idx])
ax_fi.set_yticklabels(feature_name[sorted_idx])
ax_fi.set_yticks(y_ticks)
ax_fi.set_title("Feature Importance")
fig_fi.tight_layout()
plt.show()
print("=====")
#===========================================================

## CatBoostClassifier
Y_cat = data['loan_status']
X_cat = data.drop(columns=["loan_status"])

categorical_features = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']

# Label encode categorical features
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    X_cat[col] = le.fit_transform(X_cat[col])
    label_encoders[col] = le

# Identify categorical feature indices
cat_feature_indices = [X_cat.columns.get_loc(col) for col in categorical_features]

#X_cat = X_cat.values
# Split the data
x_train, x_test, y_train, y_test = train_test_split(X_cat, Y_cat, test_size=0.2, random_state=12)

# Initialize CatBoostClassifier
catboost = CatBoostClassifier(iterations=1000, depth=4, learning_rate=0.1, loss_function='Logloss', verbose=100)
catboost.fit(x_train, y_train, cat_features=cat_feature_indices, plot=True, eval_set=(x_test, y_test))
predic_cat = catboost.predict(x_test)

print(classification_report(y_test, predic_cat))
accuracy_cat = accuracy_score(y_test, predic_cat)
precision_cat = precision_score(y_test, predic_cat)
recall_cat = recall_score(y_test, predic_cat)

print(f"Accuracy CatBoost: {accuracy_cat}")
print(f"Precision CatBoost: {precision_cat}")
print(f"Recall CatBoost: {recall_cat}")
print("Confusion Matrix CatBoost:")
print(confusion_matrix(y_test, predic_cat))
print("CatBoost Classification Report:")
print(classification_report(y_test, predic_cat))


########################################################################

'''
conclusion:
In the overview of the data, it can be seen that most people have the highest debt from education, followed by medical expenses.

From the graph comparing loan status and age, younger people tend to have more debt than older people and are more likely to default on their debts.

From the graph comparing loan status and loan intent, it was found that people who defaulted on their debts often had the highest medical expenses. This may be caused by emergencies requiring them to borrow to pay.

From the model training, 
it was found that using the CatBoost Classifier model gives the highest accuracy in predicting those who may default on their debts, 
with an accuracy of 0.933 or 93.3%. In comparison, XGBoost achieved an accuracy of 0.930, and LightGBM achieved 0.929.

The use of a forest of trees to evaluate the importance of features shows 
that the factors affecting customer default are the percentage of income represented by the loan amount, 
the type of home ownership, and the individual's income.

'''