# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 20:54:45 2020

@author: Roopali
"""

#detect frauds from credit card applications


import numpy as np
import pandas as pd

dataframe=pd.read_csv('Credit_Card_Applications.csv')
X=dataframe.iloc[:,:-1].values
y=dataframe.iloc[:,-1].values #dependent variable


#normalise
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range=(0,1))
X=sc.fit_transform(X)

#importing minisom (SOM implementation) using numpy
from minisom import MiniSom
som=MiniSom(x=8,y=8,input_len=15,sigma=1.0,learning_rate=0.5)

#initialise random weights
som.random_weights_init(X)

#train som
som.train_random(data=X,num_iteration=100)

#visualise
from pylab import bone,pcolor,colorbar,plot,show
bone()
pcolor(som.distance_map().T)
colorbar()

#visualise feature map on som y
markers=['o','s']
colors=['r','g']
for i,x in enumerate(X):
    w=som.winner(x)
    plot(w[0]+0.5, w[1]+0.5,
         markers[y[i]],
         markeredgecolor=colors[y[i]],
         markerfacecolor='None',
         markersize=10,
         markeredgewidth=2)
show()

#identify outliers with feature value
mappings=som.win_map(X)
frauds=np.concatenate((mappings[(6,6)],mappings[(6,5)]),axis=0)
frauds=np.concatenate((frauds,mappings[(4,3)]),axis=0)
# but since we see 8,8 is empty
#frauds=mappings[(4,7)]

frauds=sc.inverse_transform(frauds)

#obtained possible fraudulent cases



#now go to supervised learning


#matrix of features customer_info #remove customer ID first one
customer_info=dataframe.iloc[:,1:].values

#extract all those classified as fraud into a vector
#dependant variable of fraud or not (0 or 1) binary = here will now be outcome of whether SOM classified it as fraud
#initialise a vector with all y=0 meaning no fraud 

customer_fraud=np.zeros(dataframe.shape[0])

for i in range(dataframe.shape[0]):
    if dataframe.iloc[i,0] in frauds: #if when iteration over dataset, any customer id is present in frauds (classified in SOM as fraudulent)
        customer_fraud[i]=1 #set dep var to 1

#now train ann on this output
#this model will replicate how som classified entries into frauds
#and we can give it a new entry and check whether a new entrant is fraud or not
#hence train this ann on the customer info and customer fraud

#scale independant var matrix
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customer_info = sc.fit_transform(customer_info)

from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
#1 hidden layer 2 neurons
classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu', input_dim = 15))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(customer_info, customer_fraud, batch_size = 1, epochs = 20)

#giving new input to check whether it can be fraud or not


#calculating percentage of frauds
y_pred = classifier.predict(customer_info)
#probablities of entry being a fraud

#integrating into a 2d array probabilities vs customer id , horizontal concat
y_pred = np.concatenate((dataframe.iloc[:, 0:1].values, y_pred), axis = 1)

# give most probable fraudulent customers

#sorted list
#y_pred = y_pred[y_pred[:, 1].argsort()]


most_probable_fraud=[]
for i in range(y_pred.shape[0]):
    if y_pred[i,1]>=0.5:
        most_probable_fraud.append(y_pred[i,0])


print(most_probable_fraud)




#new input check fraudulent or not
new_customer=[15776156,1,22.08,11.46,2,4,4,1.585,0,0,0,1,2,100,1213]
new_customer=np.expand_dims(new_customer,axis=0)
new_customer=sc.transform(new_customer)
#convert to 2d array for ann input
#new_customer=np.expand_dims(new_customer,axis=0)
new_customer_fraud=classifier.predict(new_customer)

if new_customer_fraud>=0.5 :
    print('Fraudulent!')
else:
    print('Safe')
        