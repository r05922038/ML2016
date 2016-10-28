
# coding: utf-8

# In[ ]:

import csv
import numpy as np
import math
import sys

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
bias=float(model[1][0])

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

with open(answer_name,'wb') as csvfile:
    ansFile=csv.writer(csvfile)
    ansFile.writerow(['id','label'])
    for i in xrange(600):
        f_x=1.0/(1+math.exp(-(np.sum(weight*testSet[i])+bias)))
        answer=1
        if f_x<0.5:
            answer=0
        else:
            if testSet[i,41]>0.4  or testSet[i,40]>0.1:
                answer=0
        a=[str(i+1),str(answer)]
        ansFile.writerow(a)

