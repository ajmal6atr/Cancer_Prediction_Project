import pandas as pd

# load dataset

cancer = pd.read_csv("cancer.csv")
cancer.head()
cancer.index
cancer.columns

X = cancer.drop('diagnosis',axis=1)
#split dataset in features and target variable


X = cancer.drop('diagnosis',axis=1)

y = cancer.diagnosis # Target variable

# Import train_test_split function
from sklearn.model_selection import train_test_split

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3,random_state=109) # 70% training and 30% test
#Import svm model

from sklearn import svm

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

from sklearn import metrics

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix
print("Confusion matrix",cnf_matrix)

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test, y_pred))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred))

print("F1 score",metrics.f1_score(y_test, y_pred))



new1= clf.predict([[17.99,10.38,	122.8,	1001,	0.1184,	0.2776,	0.3001,	0.1471,	0.2419,	0.07871,	1.095,	0.9053	,8.589,	153.4	,0.006399,	0.04904	,0.05372999999999999	,0.01587	,0.03003	,0.006193	,25.38,	17.33,	184.6	,2019,0.1622,	0.6656,	0.7119,	0.2654,	0.4601,	0.1189]])
print(new1)
if new1==1:
    print('Cancer')
else:
    print('Non cancer')