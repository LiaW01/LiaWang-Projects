import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from xgboost import XGBClassifier
import os
import joblib

def predictFunc(model, input_data):
    """
    Preprocess input data and use the trained model to predict default probabilities.

    Args:
        model: Trained XGBoost model.
        input_data: DataFrame containing new input data.

    Returns:
        Numpy array of predicted probabilities.
    """

    scaler_model = joblib.load("scaler.pkl")
    kmeans_model = joblib.load("kmeans_model.pkl")
    cluster_medians = pd.read_csv("cluster_medians.csv", index_col=0)


    # Date processing and feature engineering
    input_data['stmt_date'] = pd.to_datetime(input_data['stmt_date'])
    input_data['def_date'] = pd.to_datetime(input_data['def_date'])
    input_data['firm_year_end'] = input_data['stmt_date'] + pd.DateOffset(years=1) + pd.DateOffset(months=3)
    input_data['default'] = input_data.apply(lambda x: 1 if pd.notna(x['def_date']) and 
                                             (x['stmt_date'] <= x['def_date'] < x['firm_year_end']) else 0, axis=1)
    
    # Variables for imputation and feature engineering
    variables = ['eqty_tot', 'asst_tot', 'liab_lt', 'liab_lt_emp', 'roa', 'rev_operating', 'COGS',
                 'cf_operations', 'AP_lt', 'debt_lt', 'debt_fin_lt', 'debt_bank_lt', 'asst_current']

    # Step 1: Impute missing values temporarily with column medians
    imputer = SimpleImputer(strategy='median')
    temp_data = pd.DataFrame(imputer.fit_transform(input_data[variables]), 
                             columns=variables, index=input_data.index)

    # Step 2: Scale data using the pre-fitted scaler
    df_scaled = scaler_model.transform(temp_data)

    # Step 3: Predict clusters using the pre-fitted KMeans model
    clusters = kmeans_model.predict(df_scaled)
    input_data['Cluster'] = clusters

    # Step 4: Impute missing values with cluster medians
    for cluster in input_data['Cluster'].unique():
        for variable in variables:
            mask = (input_data['Cluster'] == cluster) & (input_data[variable].isna())
            input_data.loc[mask, variable] = cluster_medians.loc[cluster, variable]    

    # Additional feature engineering
    input_data['total_liabilities'] = input_data['asst_tot'] - input_data['eqty_tot']
    input_data['current_liabilities'] = (
        input_data['total_liabilities'] - input_data['liab_lt'] - input_data['liab_lt_emp'] 
        - input_data['AP_lt'] - input_data['debt_lt'] - input_data['debt_fin_lt'] - input_data['debt_bank_lt']
    )
    input_data['equity_ratio'] = input_data['total_liabilities'] / input_data['eqty_tot']
    input_data['gross_profit_margin'] = np.where(
        (input_data['COGS'] == 0) & (input_data['rev_operating'] == 0), 
        0, (input_data['rev_operating'] - input_data['COGS']) / input_data['rev_operating']
    )
    input_data['operating_cashflow_ratio'] = np.where(
        (input_data['cf_operations'] == 0) & (input_data['current_liabilities'] == 0), 
        0, input_data['cf_operations'] / input_data['current_liabilities']
    )
    input_data['size'] = np.log(input_data['asst_tot'])
    
    # Selected features for prediction
    features = ['equity_ratio', 'roa', 'size', 'operating_cashflow_ratio', 'gross_profit_margin']
    for feature in features:
        input_data[feature].replace([np.inf, -np.inf], np.nan, inplace=True)
        input_data[feature].fillna(input_data[feature].median(), inplace=True)

    # Generate predictions
    predictions = model.predict_proba(input_data[features])[:, 1]
    return predictions
