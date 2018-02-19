import numpy as np
from sklearn import preprocessing, model_selection, neighbors, svm
import pandas as pd

df = pd.read_csv('TRAIN.csv')
df = df[:len(df)-1]
df.replace('?', -99999, inplace=True)
df.drop(['Customer ID', 'Customer tenure in month', 'Network type subscription in Month 1', 'Network type subscription in Month 2', 'Most Loved Competitor network in in Month 1', 'Most Loved Competitor network in in Month 2'], 1, inplace=True)

X = np.array(df.drop(['Churn Status'], 1))
y = np.array(df['Churn Status'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.1)

clf = svm.SVC()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)

tdf = pd.read_csv('TEST.csv')
tf = pd.read_csv('TEST.csv')
tdf.drop(['Customer ID', 'Customer tenure in month', 'Network type subscription in Month 1', 'Network type subscription in Month 2', 'Most Loved Competitor network in in Month 1', 'Most Loved Competitor network in in Month 2', 'Churn Status'], 1, inplace=True)

prediction = clf.predict(tdf)

submission = pd.DataFrame({ 'Customer ID': tf['Customer ID'], 'Churn Status': prediction})

submission.to_csv("submission.csv", index=True)