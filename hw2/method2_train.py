
# coding: utf-8

# In[ ]:

import csv
import numpy as np
import math
import sys

def normalize(X,length):
    
    minMax=np.array([np.append(min(X[...,i]),max(X[...,i])) for i in xrange(57)])
    for x in xrange(length):
        for y in xrange(57):
            X[x,y]=(X[x,y]-minMax[y,0])/(minMax[y,1]-minMax[y,0])
    return X

test_name=sys.argv[2]
if test_name.find('.csv')<0:
    test_name=test_name+'.csv'
model_name=sys.argv[1]
if model_name.find('.csv')<0:
    model_name=model_name+'.csv'
answer_name=sys.argv[3]   
if answer_name.find('.csv')<0:
    answer_name=answer_name+'.csv'

#model
modelf=open(model_name, 'rb')
model=[row for row in csv.reader(modelf)]
modelf.close()    
weight=np.array(model[0],dtype=np.float)

#process data_set
testFile=open(test_name, 'rb')
testSet=[row[1:] for row in csv.reader(testFile)]
testFile.close()    
testSet=np.array(testSet,dtype=np.float)

for i in xrange(len(testSet)):
    testSet[i][54]=math.log(testSet[i][54])
    testSet[i][56]=math.log(testSet[i][56])
    if testSet[i][18]!=0:
        testSet[i][18]=math.log(testSet[i][18])

testSet2=normalize(np.array(testSet),600)

with open(answer_name,'wb') as csvfile:
    ansFile=csv.writer(csvfile)
    ansFile.writerow(['id','label'])
    for i in xrange(600):
        
        f=np.sum(weight*testSet2[i])
        y_pred=1
        if (f-1)**2>(f-0)**2:
            y_pred=0
#        elif  testSet2[i,31]>0  or testSet2[i,28]>0.5  or  ValidationSet[i,27]>0.5  or ValidationSet[i,26]>0.5  or ValidationSet[i,41]>0.4  or ValidationSet[i,40]>0.1:
#            y_pred=0
        
        
        a=[str(i+1),str(y_pred)]
        ansFile.writerow(a)

