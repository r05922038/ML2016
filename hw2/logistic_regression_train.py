
# coding: utf-8

# In[6]:

import csv
import numpy as np
import math
import sys

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

alpha=0.1

weight=np.array([0.0]*w_len)
bias=0.0 
gb=0.0
wb=np.array([0.0]*w_len)

iteration=15000
for i in range(iteration):
   
    b_grad=0.0
    w_grad=np.array([0.0]*w_len)
    for n in xrange(len(testSet)):
        z=np.sum(weight*testSet[n,:w_len])+bias
        f_x=1.0/(1+math.exp(-z))
        b_grad=b_grad-(testSet[n,w_len]-f_x)
        w_grad=w_grad-(testSet[n,w_len]-f_x)*testSet[n,:w_len]
    
    gb+=(b_grad**2)
    wb+=(w_grad**2)
    
    bias=bias-alpha*(1./(gb**0.5))*b_grad
    weight=weight-alpha*(1./(wb**0.5))*w_grad
    
with open(model_name,'wb') as csvfile:
    ansFile=csv.writer(csvfile)
    ansFile.writerow(weight)
    ansFile.writerow([bias])


# In[ ]:



