#!/usr/bin/env python
# coding: utf-8

# # Package

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import  accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import warnings
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import roc_auc_score, roc_curve
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import os
import joblib
import seaborn as sns
import math

warnings.filterwarnings("ignore")

#pip install xgboost==1.7.6


# # Data Understanding & Preparation

# In[5]:


df = pd.read_csv('train.csv')
df.head()


# In[6]:


df.info()


# In[7]:


# Datetime Conversion
df['stmt_date'] = pd.to_datetime(df['stmt_date'])
df['def_date'] = pd.to_datetime(df['def_date'])


# In[8]:


# check wehther all statement date are on 12/31
all((df['stmt_date'].dt.month == 12) & (df['stmt_date'].dt.day == 31))


# ## Target Variable

# In[10]:


# define the target variable 'default'
df['firm_year_end'] = df['stmt_date'] + pd.DateOffset(years=1)+ pd.DateOffset(months=3)

df['default'] = df.apply(lambda x: 1 if pd.notna(x['def_date']) and 
                            (x['stmt_date'] <= x['def_date'] < x['firm_year_end']) else 0, axis=1)


# In[11]:


default_counts = df['default'].value_counts()
# Number of default and nondefault
print(f"Number of rows with default=1: {default_counts.get(1, 0)}")
print(f"Number of rows with default=0: {default_counts.get(0, 0)}")


# ## Independent Variables

# In[13]:


variables = ['eqty_tot', 'asst_tot', 'liab_lt', 'liab_lt_emp', 'roa', 'rev_operating', 'COGS',
            'cf_operations', 'AP_lt', 'debt_lt', 'debt_fin_lt', 'debt_bank_lt', 'asst_current']
print(df[variables].isna().sum()) 


# In[14]:


(df[variables] == 0).sum()


# In[15]:


df[variables].describe()


# In[16]:


# Visualize the variable distributions
num_cols = len(df[variables].columns)
n_rows = math.ceil(num_cols / 6)  # 6 plot in each row

fig, axes = plt.subplots(n_rows, 6, figsize=(24, 4 * n_rows))  
axes = axes.flatten() 

for i, col in enumerate(df[variables].columns):
    ax = axes[i]  
    if df[col].dtype in ['int64', 'float64']:  # Numerical columns
        df[col].hist(bins=30, ax=ax)
        ax.set_title(col)
    else:  # Categorical columns
        sns.countplot(x=col, data=df, ax=ax)
        ax.set_title(col)

# remove blank plots
for i in range(len(df[variables].columns), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()


# ## Imputing missing values with K-mean clustering

# In[18]:


# temporarily fill missing value with column medians
imputer = SimpleImputer(strategy='median')
df_temp = pd.DataFrame(imputer.fit_transform(df[variables]), columns=variables, index=df.index)

# scale the data for clustering
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_temp)


# In[19]:


# Elbow method to find optimal number of clusters
wcss = []
for k in range(3, 16):
    kmeans = KMeans(n_clusters=k, random_state=24)
    kmeans.fit(df_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(3, 16), wcss, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.show()


# In[20]:


# clustering with 5 clusters
kmeans = KMeans(n_clusters=5, random_state=24)
clusters = kmeans.fit_predict(df_scaled)
df['Cluster'] = clusters

# calculate cluster medians
cluster_medians = df.groupby('Cluster')[variables].median()

# replacing missing value with cluster medians
for cluster in df['Cluster'].unique():
    for variable in variables:
        mask = (df['Cluster'] == cluster) & (df[variable].isna())
        df.loc[mask, variable] = cluster_medians.loc[cluster, variable]

# Save parameters
output_dir = os.path.dirname(os.path.abspath(__file__))

scaler_path = os.path.join(output_dir, "scaler.pkl")
joblib.dump(scaler, scaler_path)

kmeans_model_path = os.path.join(output_dir, "kmeans_model.pkl")
joblib.dump(kmeans, kmeans_model_path)

cluster_medians_path = os.path.join(output_dir, "cluster_medians.csv")
cluster_medians.to_csv(cluster_medians_path, index=True)

# check current missing values
print("Missing values after imputation:\n", df[variables].isna().sum())


# ## Feature Engineering

# In[22]:


# calculate total liabilities
df['total_liabilities'] = df['asst_tot'] - df['eqty_tot']


# calculate current liabilities
df['current_liabilities'] = df['total_liabilities'] - df['liab_lt']- df['liab_lt_emp'] 
- df['AP_lt'] - df['debt_lt'] - df['debt_fin_lt'] - df['debt_bank_lt']


## Derived features
# Solvency
df['equity_ratio'] = df['total_liabilities'] / df['eqty_tot']


# Profitability: ROA (Return on Assets) / gross_profit_margin
df['roa'] = df['roa']

df['gross_profit_margin'] = np.where((df['COGS'] == 0) & (df['rev_operating'] == 0), 
    0,(df['rev_operating'] - df['COGS']) / df['rev_operating']) 


# Liquidity: Cash Flow Ratio = Operating Cash Flow 
## Alternative: Current ratio
df['operating_cashflow_ratio'] = np.where((df['cf_operations'] == 0) & (df['current_liabilities'] == 0), 
    0, df['cf_operations'] / df['current_liabilities'])

#df['current_ratio'] = np.where((df['asst_current'] == 0) & (df['current_liabilities'] == 0), 
    #0, df['asst_current'] / df['current_liabilities'])



# Size: Total Assets
df['size'] = np.log(df['asst_tot'])


target = 'default'
features = ['equity_ratio', 'roa', 'size', 'operating_cashflow_ratio', 'gross_profit_margin']


# In[23]:


df[features].head()


# In[24]:


# visualize features distribution
fig, axes = plt.subplots(1, 5, figsize=(25, 5))  
axes = axes.flatten() 

for i, feature in enumerate(features):
    sns.histplot(df[feature], kde=True, ax=axes[i], bins=30)
    axes[i].set_title(feature)

# remove blank plots
if len(features) < len(axes):
    for j in range(len(features), len(axes)):
        fig.delaxes(axes[j])

plt.tight_layout() 
plt.show()


# In[25]:


print(df[features].isna().sum()) 


# ### Handling inf or -inf

# In[27]:


for feature in features:
    # 1. replace inf or -inf with null values
    df[feature].replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # 2. replace null values with medians
    median_value = df[feature].median()
    df[feature].fillna(median_value, inplace=True)


# In[28]:


print(df[features].isna().sum()) 


# In[29]:


# visualize features distribution
fig, axes = plt.subplots(1, 5, figsize=(25, 5))  
axes = axes.flatten() 

for i, feature in enumerate(features):
    sns.histplot(df[feature], kde=True, ax=axes[i], bins=30)
    axes[i].set_title(feature)

# remove blank plots
if len(features) < len(axes):
    for j in range(len(features), len(axes)):
        fig.delaxes(axes[j])

plt.tight_layout() 
plt.show()


# plot shows that distribution does not change much means the median imputation does not create outliers or bias


# ## Collinearity Check (VIF)

# In[31]:


vif_data = pd.DataFrame()
vif_data['feature'] = df[features].columns
vif_data['VIF'] = [variance_inflation_factor(df[features], i) for i in range(df[features].shape[1])]

print(vif_data)


# # Models

# In[33]:


# train test split
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


# ## 1. Baseline - Decision Tree

# In[35]:


# fit the tree model
model_dt = DecisionTreeClassifier(random_state=42, max_depth=5)
model_dt.fit(X_train, y_train)

# predict on test dataset with trained tree model
y_pred_dt = model_dt.predict(X_test)
y_pred_proba_dt = model_dt.predict_proba(X_test)[:, 1]


# In[36]:


# check model power with ROC Curve and AUC Score
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_pred_proba_dt)
auc_score_dt = roc_auc_score(y_test, y_pred_proba_dt)

roc_auc_dt = roc_auc_score(y_test, y_pred_proba_dt)
print(f"\nROC-AUC Score: {roc_auc_dt:.2f}")

plt.plot(fpr_dt, tpr_dt, label=f'ROC Curve (AUC = {auc_score_dt:.2f})')
plt.plot([0, 1], [0, 1], 'k--') 
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Decision Tree')
plt.legend()
plt.show()


# In[37]:


# check feature importance for decision tree
feature_importance_dt = pd.DataFrame({'Feature': X.columns, 'Importance': model_dt.feature_importances_})
print("\nFeature Importance:")
print(feature_importance_dt.sort_values(by='Importance', ascending=False))

plt.show()
plt.figure(figsize=(8, 6))
feature_importance_dt.sort_values(by='Importance', ascending=False, inplace=True)
plt.barh(feature_importance_dt['Feature'], feature_importance_dt['Importance'], color='skyblue')
plt.xlabel('Coefficient Value')
plt.title('Feature Importance for Decision Tree')
plt.show()



# ## 2. XGBoost

# In[ ]:


# Grid Search for tuning XGBoost model
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

model_xgb = XGBClassifier()
grid_search = GridSearchCV(estimator=model_xgb, param_grid=param_grid, cv=3, scoring='roc_auc', verbose=1)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best ROC-AUC Score:", grid_search.best_score_)


# We used grid search to tune the XGBoost model and get the best parameter set.

# In[ ]:


# fit model with best parameters and make prediction on test set

best_params = grid_search.best_params_  
model_xgb = XGBClassifier(**best_params)
model_xgb.fit(X_train, y_train)
y_pred_xgb = model_xgb.predict(X_test)
y_pred_proba_xgb = model_xgb.predict_proba(X_test)[:, 1]


# In[ ]:


fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_proba_xgb)
auc_score_xgb = roc_auc_score(y_test, y_pred_proba_xgb)

roc_auc_xgb_updated = roc_auc_score(y_test, y_pred_proba_xgb)
print(f"\nROC-AUC Score: {roc_auc_xgb_updated:.2f}")

plt.plot(fpr_xgb, tpr_xgb, label=f'ROC Curve (AUC = {auc_score_xgb:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for XGBoost')
plt.legend()
plt.show()


# The ROC curve, with an AUC of 0.85, indicates good model performance in distinguishing between positive (default) and negative (non-default) classes. An AUC of 0.85 suggests that the model has a high probability of ranking a randomly chosen positive instance higher than a randomly chosen negative instance. This performance measure helps confirm that the selected features and the model’s structure provide a reliable classification of default risk. 
# Compared with our baseline decision tree model, the AUC increased from 0.81 to 0.85.

# ### Feature Importance

# In[ ]:


feature_importance_xgb = pd.DataFrame({'Feature': X.columns, 'Importance': model_xgb.feature_importances_})
print("\nFeature Importance:")
print(feature_importance_xgb.sort_values(by='Importance', ascending=False))


plt.show()
plt.figure(figsize=(8, 6))
feature_importance_xgb.sort_values(by='Importance', ascending=False, inplace=True)
plt.barh(feature_importance_xgb['Feature'], feature_importance_xgb['Importance'], color='skyblue')
plt.xlabel('Coefficient Value')
plt.title('Feature Importance for XGBoost')
plt.show()


# ## XGBoost Calibration

# In[ ]:


# 'isotonic' regression method is a non-parametric calibration technique work for non linear data
# 'prefit' means the model is calibrated after it is trained
calibrated_xgb = CalibratedClassifierCV(estimator=model_xgb, method='isotonic', cv='prefit')
calibrated_xgb.fit(X_train, y_train)

y_pred_calibrated = calibrated_xgb.predict(X_test)
y_pred_proba_calibrated = calibrated_xgb.predict_proba(X_test)[:, 1]

prob_true, prob_pred = calibration_curve(y_test, y_pred_proba_calibrated, n_bins=10)

plt.figure(figsize=(8, 6))
plt.plot(prob_pred, prob_true, marker='o', label='Calibrated Model')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')

plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curve (Calibrated XGBoost Model)')
plt.legend()
plt.show()


# In[ ]:


model_xgb.get_booster().save_model("model_xgb.json")  # Save the trained model

