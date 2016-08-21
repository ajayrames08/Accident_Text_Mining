# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 11:30:59 2015

@author: santhosh
"""

import pandas as pd
import sklearn
import random

from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn import metrics
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import preprocessing



def create_tfidf_training_data(docs):
    """
    Creates a document corpus list (by stripping out the
    class labels), then applies the TF-IDF transform to this
    list. 

    The function returns both the class label vector (y) and 
    the corpus token/feature matrix (X).
    """
    # Create the training data class labels
    y = [d[0] for d in docs]

    # Create the document corpus list
    corpus = [d[1] for d in docs]

    # Create the TF-IDF vectoriser and transform the corpus
    vectorizer = TfidfVectorizer(min_df=1)
    X = vectorizer.fit_transform(corpus)
    return vectorizer,X, y
    
def train_knn(X, y, n, weight):
    """
    Create and train the k-nearest neighbor.
    """
    knn = KNeighborsClassifier(n_neighbors = n, weights = weight, metric = 'cosine', algorithm = 'brute')
    knn.fit(X, y)
    return knn
    
    
def train_svm(X, y):
    """
    Create and train the Support Vector Machine.
    """
    svm = SVC(C=2500.0, gamma=0.0, kernel='rbf')
    svm.fit(X, y)
    return svm

xls = pd.ExcelFile('MsiaAccidentCases.xlsx')
data = xls.parse('MsiaAccidentCases-cleaned',index_col=None, na_values=['NA'])
data=pd.DataFrame(data);

data_test = xls.parse('Test',index_col=None, na_values=['NA']);
data_test=pd.DataFrame(data_test);
osha=pd.ExcelFile('osha.xlsx');
data_prediction= osha.parse('out_title',index_col=None, na_values=['NA'])


col_list=[0,1];
data=data[col_list];
data_test=data_test[col_list]
labelled=[];
labelled_pred=[];

for index,row in data.iterrows(): 
    labelled.append(row.tolist())
    
for index,row in data_test.iterrows():
    labelled.append(row.tolist())

for index,row in data_prediction.iterrows():
    labelled_pred.append(row.tolist())
    


####### KNN ################3
vectorizer,X,y = create_tfidf_training_data(labelled)
X_train=X[:182]
X_test=X[182:]
y_train=y[:182]
y_test=y[182:]

kn = train_knn(X_train, y_train, 5, 'distance')
predKN = kn.predict(X_test)
print(kn.score(X_test, y_test))

actual_classifications=list(y_test)
actuals = pd.Series(actual_classifications)
predicted = pd.Series(predKN)
confusion_matrix=pd.crosstab(actuals, predicted, rownames=['Actuals'], colnames=['Predicted'], margins=True)
confusion_matrix.to_csv("confusion_matrix_knn.csv")
target_names = ["Caught in/between Objects","Collapse of object","Drowning","Electrocution","Exposure to Chemical Substances","Exposure to extreme temperatures","Falls","Fires and Explosion","Other","Struck By Moving Objects","Suffocation"]

print(classification_report(actuals, predicted, target_names=target_names))


corpus=[str(d[1]) for d in labelled_pred]
pred=vectorizer.transform(corpus)
prediction=kn.predict(pred)
columns=['Cause_no', 'Predicted Cause','Fatality']
result=pd.DataFrame(columns=columns)
for index,row in data_prediction.iterrows():
    row=row.tolist();
    result=result.append({'Cause_no':row[0],'Predicted Cause':prediction[index],'Fatality':row[5]},ignore_index=True)

result.to_csv("result_knn.csv")
######### SVM  best ################0
svm = train_svm(X_train, y_train)

predSVM = svm.predict(X_test)
print(svm.score(X_test, y_test))

actual_classifications=list(y_test)
actuals = pd.Series(actual_classifications)
predicted = pd.Series(pred)
confusion_matrix=pd.crosstab(actuals, predicted, rownames=['Actuals'], colnames=['Predicted'], margins=True)
confusion_matrix.to_csv("confusion_matrix_svm.csv")
target_names = ["Caught in/between Objects","Collapse of object","Drowning","Electrocution","Exposure to Chemical Substances","Exposure to extreme temperatures","Falls","Fires and Explosion","Other","Struck By Moving Objects","Suffocation"]

print(classification_report(actuals, predicted, target_names=target_names))
pred=vectorizer.transform(corpus)
prediction=[]
prediction=svm.predict(pred)
columns=['Cause_no', 'Predicted Cause']
result_svm=pd.DataFrame(columns=columns)
for index,row in data_prediction.iterrows():
    row=row.tolist();
    result_svm=result_svm.append({'Cause_no':row[0],'Predicted Cause':prediction[index],'Fatality':row[5]},ignore_index=True)

result_svm.to_csv("result_svm.csv")
########## SVM  K best ################
#le = preprocessing.LabelEncoder()
#
#le.fit(y_train)
#y_train_transformed=le.transform(y_train) 
#
#le.fit(y_test)
#y_test_transformed = le.transform(y_test)
#
#ch22 = SelectKBest(chi2, k='all')
#Xbest_Train2 = ch22.fit_transform(X_train, y_train_transformed)
#Xbest_Test2= ch22.transform(X_test)
#
## Size of Training Set
#print Xbest_Train2.shape
#
## Size of Testing Set
#print Xbest_Test2.shape
#
#svmkbest = train_svm(Xbest_Train2, y_train_transformed)
#
#predbest = svmkbest.predict(Xbest_Test2)
#
#print(svmkbest.score(Xbest_Test2, y_test_transformed))
