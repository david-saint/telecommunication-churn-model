import pandas as pd


data = pd.read_csv('Submission_xgb.csv')
data[data['Churn Status'] > 0] = -1
data[data['Churn Status'] == 0] = 1
data[data['Churn Status'] < 0] = 0

data = data[['Customer ID', 'Churn Status']]
data.to_csv('Submission_xgb2.csv', index=False)