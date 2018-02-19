import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

np.set_printoptions(suppress=True)

sns.set()
sns.set_style('whitegrid')

data = pd.read_csv('TRAIN.csv')
data = data.rename(columns=lambda x: x.strip())

data['Consistent competitor'] = (data['Most Loved Competitor network in in Month 1'
                                    ] == data['Most Loved Competitor network in in Month 2']).astype(int)
# if the competitor in month 1 and month 2 are the same, it gives a 1

data['Network Upgrade'] = 0

data.loc[(data['Network type subscription in Month 1'] == '2G') & 
              (data['Network type subscription in Month 2'] == '3G'), 'Network Upgrade'] = 1

data.loc[(data['Network type subscription in Month 1'] == '3G') & 
              (data['Network type subscription in Month 2'] == '2G'), 'Network Upgrade'] = -1
# if there is an upgrade in network, +1, downgrade: -1, else, 0

selected_data = data.iloc[:,1:]

selected_data.drop('network_age', inplace=True, axis=1)

network_month1_dummy = pd.get_dummies(selected_data['Network type subscription in Month 1'])
selected_data = selected_data.join(network_month1_dummy, rsuffix = "_1")
selected_data.drop('Network type subscription in Month 1', axis=1, inplace=True)

network_month2_dummy = pd.get_dummies(selected_data['Network type subscription in Month 2'])
selected_data = selected_data.join(network_month2_dummy, rsuffix = "_2")
selected_data.drop('Network type subscription in Month 2', axis=1, inplace=True)

competitor_month1_dummy = pd.get_dummies(selected_data['Most Loved Competitor network in in Month 1'])
selected_data = selected_data.join(competitor_month1_dummy, rsuffix = "_1")
selected_data.drop('Most Loved Competitor network in in Month 1', axis=1, inplace=True)

competitor_month2_dummy = pd.get_dummies(selected_data['Most Loved Competitor network in in Month 2'])
selected_data = selected_data.join(competitor_month2_dummy, rsuffix = "_2")
selected_data.drop('Most Loved Competitor network in in Month 2', axis=1, inplace=True)

selected_data['Churn Status'] = selected_data['Churn Status'].apply(lambda x: -1 if x==0 else 1)

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import linear_model

train_data, cv_data = train_test_split(selected_data, test_size=0.3, random_state=42)

train_x = train_data.drop('Churn Status', axis=1)
train_y = train_data['Churn Status']

cv_x = cv_data.drop('Churn Status', axis=1)
cv_y = cv_data['Churn Status']

train_x.drop(1400, inplace=True)
train_y.drop(1400, inplace=True)

from sklearn.preprocessing import MinMaxScaler
std_scaler = MinMaxScaler()
std_scaler.fit(train_x)

#\train_x_std = std_scaler.transform(train_x)
#cv_x_std = std_scaler.transform(cv_x)

train_x_std = train_x
cv_x_std = cv_x

lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
lda.fit(train_x_std, train_y)
    
train_preds_lda = lda.predict(train_x_std)
cv_preds_lda = lda.predict(cv_x_std)

train_acc_lda = accuracy_score(train_preds_lda, train_y)
cv_acc_lda = accuracy_score(cv_preds_lda, cv_y)

print("Training accuracy for linear discriminant analysis is ", train_acc_lda)
print("CV accuracy for linear discriminant analysis is ", cv_acc_lda)


qda = QuadraticDiscriminantAnalysis(store_covariance=True)
qda.fit(train_x_std, train_y)
    
train_preds_qda = qda.predict(train_x_std)
cv_preds_qda = qda.predict(cv_x_std)

train_acc_qda = accuracy_score(train_preds_qda, train_y)
cv_acc_qda = accuracy_score(cv_preds_qda, cv_y)

print("Training accuracy for Quadratic discriminant analysis is ", train_acc_qda)
print("CV accuracy for Quadratic discriminant analysis is ", cv_acc_qda)


logReg = linear_model.LogisticRegression(C=1)
logReg.fit(train_x_std, train_y)
#Value of C=1 was obtained after some tries

train_preds_lr = logReg.predict(train_x_std)
cv_preds_lr = logReg.predict(cv_x_std)

train_acc_lr = accuracy_score(train_preds_lr, train_y)
cv_acc_lr = accuracy_score(cv_preds_lr, cv_y)

print("Training accuracy for logistic regression is ", train_acc_lr)
print("CV accuracy for logistic regression is ", cv_acc_lr)


from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100, min_samples_split=3)
random_forest.fit(train_x_std, train_y)

train_preds_rf = random_forest.predict(train_x_std)
cv_preds_rf = random_forest.predict(cv_x_std)

train_acc_rf = accuracy_score(train_preds_rf, train_y)
cv_acc_rf = accuracy_score(cv_preds_rf, cv_y)

print("Training accuracy for Random Forest is ", train_acc_rf)
print("CV accuracy for Random Forest is ", cv_acc_rf)


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3, algorithm='brute')

knn.fit(train_x_std, train_y)

train_preds_knn = knn.predict(train_x_std)
cv_preds_knn = knn.predict(cv_x_std)

train_acc_knn = accuracy_score(train_preds_knn, train_y)
cv_acc_knn = accuracy_score(cv_preds_knn, cv_y)

print("Training accuracy for knn is ", train_acc_knn)
print("CV accuracy for knn is ", cv_acc_knn)


from sklearn.ensemble import AdaBoostClassifier
ada_boost = AdaBoostClassifier(n_estimators=50)

ada_boost.fit(train_x_std, train_y)

train_preds_ada = ada_boost.predict(train_x_std)
cv_preds_ada = ada_boost.predict(cv_x_std)

train_acc_ada = accuracy_score(train_preds_ada, train_y)
cv_acc_ada = accuracy_score(cv_preds_ada, cv_y)

print("Training accuracy for ada boost is ", train_acc_ada)
print("CV accuracy for ada boost is ", cv_acc_ada)


from xgboost import XGBClassifier
xgb = XGBClassifier(max_depth=6, n_estimators=600, learning_rate= 0.01)

xgb.fit(train_x_std, train_y)

train_preds_xgb = xgb.predict(train_x_std)
cv_preds_xgb = xgb.predict(cv_x_std)

train_acc_xgb = accuracy_score(train_preds_xgb, train_y)
cv_acc_xgb = accuracy_score(cv_preds_xgb, cv_y)

print("Training accuracy for xgb is ", train_acc_xgb)
print("CV accuracy for xgb is", cv_acc_xgb)


test_data = pd.read_csv('TEST.csv')

test_data_ID = test_data['Customer ID']

test_data['Consistent competitor'] = (test_data['Most Loved Competitor network in in Month 1'
                                    ] == test_data['Most Loved Competitor network in in Month 2']).astype(int)
# if the competitor in month 1 and month 2 are the same, it gives a 1

test_data['Network Upgrade'] = 0

test_data.loc[(test_data['Network type subscription in Month 1'] == '2G') & 
              (test_data['Network type subscription in Month 2'] == '3G'), 'Network Upgrade'] = 1

test_data.loc[(test_data['Network type subscription in Month 1'] == '3G') & 
              (test_data['Network type subscription in Month 2'] == '2G'), 'Network Upgrade'] = -1
# if there is an upgrade in network, +1, downgrade: -1, else, 0




selected_test = test_data.drop(['network_age', 'Customer ID'], axis=1)
# drop network_age because of the correlation results obtained in training. It is also obtained in the test data two cells below

#run the same procedure to get dummy variables
tnetwork_month1_dummy = pd.get_dummies(selected_test['Network type subscription in Month 1'])
selected_test = selected_test.join(tnetwork_month1_dummy, rsuffix = "_1")
selected_test.drop('Network type subscription in Month 1', axis=1, inplace=True)

tnetwork_month2_dummy = pd.get_dummies(selected_test['Network type subscription in Month 2'])
selected_test = selected_test.join(tnetwork_month2_dummy, rsuffix = "_2")
selected_test.drop('Network type subscription in Month 2', axis=1, inplace=True)

tcompetitor_month1_dummy = pd.get_dummies(selected_test['Most Loved Competitor network in in Month 1'])
selected_test = selected_test.join(tcompetitor_month1_dummy, rsuffix = "_1")
selected_test.drop('Most Loved Competitor network in in Month 1', axis=1, inplace=True)

tcompetitor_month2_dummy = pd.get_dummies(selected_test['Most Loved Competitor network in in Month 2'])
selected_test = selected_test.join(tcompetitor_month2_dummy, rsuffix = "_2")
selected_test.drop('Most Loved Competitor network in in Month 2', axis=1, inplace=True)

selected_test_std = selected_test

test_preds_xgb = xgb.predict(selected_test_std)
test_preds_xgb[test_preds_xgb > 0] = 0
test_preds_xgb[test_preds_xgb < 0] = 1
d = {'Customer ID': test_data_ID, 'Churn Status': test_preds_xgb}
test_df = pd.DataFrame(d)
test_df = test_df[['Customer ID', 'Churn Status']]
test_df.to_csv('Submission_xgb.csv', index=False)