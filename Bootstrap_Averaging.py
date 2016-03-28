# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 16:09:31 2016

@author: Shashank
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from datetime import datetime 
from sklearn.base import TransformerMixin
from scipy import sparse
from scipy.sparse import hstack
from sklearn import svm
from numpy import array
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from math import *


#### read the data set 
df_train_bal= pd.read_csv('F:/Projects/Breath Screening Data/Python/df_train_bal.csv')
df_test= pd.read_csv('F:/Projects/Breath Screening Data/Python/df_test.csv')


### divide the training data set into X and Y , categorical to one hot encoding
Y_train, X_train = dmatrices('Target ~ C(Reason)+C(Month)+C(WeekType)+C(TimeBand)+C(AgeBand)+C(Gender)',
                             df_train_bal, return_type="dataframe")

### new df train with encoded categorical variables
df_train = pd.concat([X_train, Y_train], join='outer', axis=1)
 

#### divide the training data set into positive and negative subsets
df_train_pos = df_train[df_train.Target==1]
df_train_neg = df_train[df_train.Target==0]


### divide the test set into X and Y sets
Y_test, X_test = dmatrices('Target ~ C(Reason)+C(Month)+C(WeekType)+C(TimeBand)+C(AgeBand)+C(Gender)',
                             df_test, return_type="dataframe")
                             
### new df test with encoded categorical variables
df_test = pd.concat([X_test, Y_test], join='outer', axis=1)



#### Bootstrap Averaging 
no_pos = 1000
no_neg = 8000
n_sim = 4000
#auc = np.empty(shape=n_sim)
coef = np.zeros(shape=34)

for i in range(0,n_sim):
  pos = df_train_pos.sample(n=no_pos,axis=0)
  neg = df_train_neg.sample(n=no_neg,axis=0)
  train = pd.concat([pos, neg], join='outer', axis=0)
  Y = pd.DataFrame(train['Target'])
  X = train.drop(['Target'],axis=1)
  Y = np.ravel(Y)
  model = LogisticRegression(class_weight='balanced')
  model.fit(X,Y)
  coef = coef + model.coef_

avg_coef = coef/n_sim


est_prob = np.empty(shape=len(X_test))

for i in range(0,len(X_test)):
  var = np.array(X_test.ix[i])
  prod = avg_coef * var
  sum = np.sum(prod)
  est_prob[i] = exp(sum)/(1+exp(sum))

fpr, tpr, _ = metrics.roc_curve(Y_test, est_prob)
aucf = np.trapz(tpr,fpr)

### ROC curve
plt.plot(fpr,tpr)
plt.xlabel('False Postitive Rate')
plt.ylabel('True Positive Rate')
plt.title('Reciever Operating Characteristic Curve')
plt.plot([0,1],[0,1],'r--')
plt.show()



#### Preparation for Kolmogorov - Smirnov Curve 
df1 = pd.DataFrame(est_prob, columns=['est. prob.'])
df2 = pd.DataFrame(Y_test,columns=['Observed val.'])
df_KS1 = pd.concat([df1, df2], join='outer', axis=1)
df_KS1 = df_KS1.sort_values(by='est. prob.', ascending=0)
df_KS1

Observed_val = np.array(df_KS1['Observed val.'])
Observed_val

x_val = np.empty(shape=100)
y_val = np.empty(shape=100)
y2_val = np.empty(shape=100)
diff = np.empty(shape=100)
tot_tar = np.sum(Observed_val)
tot_nontar = len(Observed_val) -  np.sum(Observed_val) 
for i in range(1,100):
    x_val[i]=i
    k = ceil((i/100)*len(Observed_val))
    temp_ys = np.array(Observed_val[0:k])
    y_val[i] = (np.sum(temp_ys)/tot_tar)*100
    y2_val[i] = ( (len(temp_ys)-(np.sum(temp_ys))) / tot_nontar)*100 
    diff[i] = y_val[i] - y2_val[i]

max_ind = np.argmax(diff)

#Kolmogorov - Smirnov Curve 
plt.plot(x_val,y_val,'b-')
plt.plot(x_val,y2_val,'b-')
plt.plot([x_val[max_ind],x_val[max_ind]],[y_val[max_ind],y2_val[max_ind]],'b--')
plt.xlabel('% of data')
plt.ylabel('% of target captured')
plt.title('Kolmogorov - Smirnov Curve')

diff[max_ind]






  
  
  
model = LogisticRegression()
Y1 = np.ravel(Y_train)
model.fit(X_train,Y1)
prob = model.predict_proba(X_test)
prob2 = prob[:,1]
Y_test = np.ravel(Y_test)
fpr, tpr, _ = metrics.roc_curve(Y_test, prob2)
auc = np.trapz(tpr,fpr)
model.coef_

a = np.array([0,0])
b = np.array([2,3])
c = a*b
d = exp(a*b)






        






         