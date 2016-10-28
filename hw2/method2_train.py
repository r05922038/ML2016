
# coding: utf-8

# In[ ]:

import sys
import csv
import numpy as np
import math

def normalize(X,length):
    
    minMax=np.array([np.append(min(X[...,i]),max(X[...,i])) for i in xrange(57)])
    for x in xrange(length):
        for y in xrange(57):
            X[x,y]=(X[x,y]-minMax[y,0])/(minMax[y,1]-minMax[y,0])
    return X

train_name=sys.argv[1]
if train_name.find('.csv')<0:
    train_name=train_name+'.csv'
model_name=sys.argv[2]
if model_name.find('.csv')<0:
    model_name=model_name+'.csv'

#process data_set
trainFile=open(train_name, 'rb')
trainSet=[row[1:] for row in csv.reader(trainFile)]    
trainFile.close()
trainSet=np.array(trainSet,dtype=np.float)

#pick feature
testSet_temp=np.array(trainSet)
testSet=[]
for i in xrange(len(testSet_temp)):
    testSet.append(list(testSet_temp[i,:58]))
    testSet[i][54]=math.log(testSet[i][54])
    testSet[i][56]=math.log(testSet[i][56])
    if testSet[i][18]!=0:
        testSet[i][18]=math.log(testSet[i][18])
    
testSet=np.array(testSet)
w_len=len(testSet[0])-1

#train:get model(weight,bias)-----------------------------------------------------------------------------------------
trainSet1=normalize(np.array(testSet[...,:w_len]),4001)
b=testSet[...,w_len]

x=trainSet1
xt=np.transpose(trainSet1)
lambdaI=0*np.identity(len(trainSet1[0]))
w=np.dot(np.dot(np.linalg.inv(np.dot(xt,x)+lambdaI),xt),b)

#write
with open(model_name,'wb') as csvfile:
    ansFile=csv.writer(csvfile)
    ansFile.writerow(w)

