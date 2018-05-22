# -*- coding: utf-8 -*-
"""
Created on Fri May 18 18:23:06 2018

@author: yvzslmbkts
"""
#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2. Veri Onisleme

#2.1. Veri Yukleme

elections = pd.read_csv('electiondaat.csv')
elections = elections.iloc[:,:-1]
ref = pd.read_csv('refdata.csv')


''' AUSTRIA (92-16)'''
electionsOfAUT92_16 = elections.iloc[:,0:1]
refOfAUT92_16 = ref.iloc[2:3,:]
refOfAUT92_16 = refOfAUT92_16.T
refOfAUT92_16 = refOfAUT92_16.iloc[4:-1,:]
electionsOfAUT92_16 = electionsOfAUT92_16.iloc[2:-2,:]


''' GERMANY (92-16)'''
electionsOfDEU92_16 = elections.iloc[:,1:2]
refOfDEU92_16 = ref.iloc[0:1,:]
refOfDEU92_16 = refOfDEU92_16.T
refOfDEU92_16 = refOfDEU92_16.iloc[4:-1,:]
electionsOfDEU92_16 = electionsOfDEU92_16.iloc[2:-2,:]


''' BELGIUM (92-16)'''
electionsOfBEL92_16 = elections.iloc[:,2:3]
refOfBEL92_16 = ref.iloc[-3:-2,:]
refOfBEL92_16 = refOfBEL92_16.T
refOfBEL92_16 = refOfBEL92_16.iloc[4:-1,:]
electionsOfBEL92_16 = electionsOfBEL92_16.iloc[2:-2,:]

''' DENMARK (92-16)'''
electionsOfDNK92_16 = elections.iloc[:,3:4]
refOfDNK92_16 = ref.iloc[-2:-1,:]
refOfDNK92_16 = refOfDNK92_16.T
refOfDNK92_16 = refOfDNK92_16.iloc[4:-1,:]
electionsOfDNK92_16 = electionsOfDNK92_16.iloc[2:-2,:]



"""
'''DECISION TREE FOR AUT'''"""
from sklearn.cross_validation import train_test_split
x_train, x_test,y_train,y_test = train_test_split(refOfAUT92_16,electionsOfAUT92_16,test_size=0.33, random_state=0) 

from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(x_train,y_train)

y_pred = r_dt.predict(x_test)

from sklearn.metrics import r2_score
print("Decion tree r^2 value")
print(r2_score(y_test,y_pred))

'''GÖRSELLEŞTİRME VE DAVRANISI TEST ETME KALDI'''



'''SVR FOR AUT(Very productive behavio'''
from sklearn.cross_validation import train_test_split
x_train, x_test,y_train,y_test = train_test_split(refOfDEU92_16,electionsOfDEU92_16,test_size=0.33, random_state=0) 
from sklearn.preprocessing import StandardScaler


sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(x_train)
sc2  = StandardScaler()
y_olcekli = sc2.fit_transform(y_train)
y_testolcekli = sc2.fit_transform(y_test)

from sklearn.svm import SVR

svr_reg = SVR(kernel = 'rbf')
svr_reg.fit(x_olcekli,y_olcekli)

svr_pred = svr_reg.predict(x_test)  
plt.scatter(x_olcekli,y_olcekli,color = 'red')
plt.plot(x_olcekli,svr_reg.predict(x_olcekli),color = 'blue')


'''LINEAR REGRESSİON'''

from sklearn.cross_validation import train_test_split
x_train, x_test,y_train,y_test = train_test_split(refOfDEU92_16,electionsOfDEU92_16,test_size=0.33, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

print(r2_score(y_test,y_pred))

