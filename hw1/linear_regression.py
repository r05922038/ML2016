
# coding: utf-8

# In[4]:

import csv
import numpy as np

#process data_set
trainFile=open('train.csv', 'rb')
next(trainFile, None)

trainSetTemp=[[]]*18
eighteenCycle=0
for row in csv.reader(trainFile):
    row=['0' if x=='NR' else x for x in row[3:27]]
    trainSetTemp[eighteenCycle]=trainSetTemp[eighteenCycle]+row            
    if eighteenCycle==17:
        eighteenCycle=0
    else:
        eighteenCycle+=1
trainFile.close()    
trainSetTemp=np.array(trainSetTemp,dtype=np.float)
trainSetTemp=np.array([trainSetTemp[...,0+i*(20*24):(20*24)+i*(20*24)] for i in xrange(12)])

#trainSet:5256*18*10
trainSet=np.array([[[0.0]*10]*18])
for i in xrange(12):
    for j in xrange(10):
        if j==0:
            trainSet=np.append(trainSet,np.array([trainSetTemp[i,...,0+x*10:10+x*10] for x in xrange(48)]),axis=0)
        else:
            trainSet=np.append(trainSet,np.array([trainSetTemp[i,...,j+x*10:(10+j)+x*10] for x in xrange(47)]),axis=0)
trainSet=trainSet[1:len(trainSet)]  

#train:get model(weight,bias)-----------------------------------------------------------------------------------------
LAMBDA=0
alpha=0.1

weight=np.array([[0.0]*9]*18)
bias=0.0 
gb=0.0
wb=np.array([[0.0]*9]*18)

iteration=30000
for i in range(iteration):
    
    b_grad=0.0
    w_grad=np.array([[0.0]*9]*18)
    for n in xrange(len(trainSet)):
        sum_WX=np.sum(weight*trainSet[n,...,:9])
        b_grad=b_grad-2.0*(trainSet[n][9][9]-bias-sum_WX)*1.0
        w_grad=w_grad-2.0*(trainSet[n][9][9]-bias-sum_WX)*trainSet[n,...,:9]
    w_grad+=2.0*LAMBDA*weight
    
    gb+=(b_grad**2)
    wb+=(w_grad**2)
    
    bias=bias-alpha*(1./(gb**0.5))*b_grad
    weight=weight-alpha*(1./(wb**0.5))*w_grad

#----------------------------------------------------------------------------------------------------------------------
#process data_set
testFile=open('test_X.csv', 'rb')

#testSet:240*18*9
testSet=[]
temp=[]
eighteenCycle=0
for row in csv.reader(testFile):
    row=np.array(['0' if x=='NR' else x for x in row[2:11]])
    temp.append(row)
    if eighteenCycle==17:
        testSet.append(temp)
        temp=[]
        eighteenCycle=0
    else:
        eighteenCycle+=1
testFile.close()    
testSet=np.array(testSet,dtype=np.float)

answer=[]
for i in xrange(240):
    answer.append(bias+np.sum(testSet[i]*weight))
answer=np.array(answer)

with open('linear_regression.csv','wb') as csvfile:
    ansFile=csv.writer(csvfile)
    ansFile.writerow(['id','value'])
    for i in xrange(240):
        a=['id_'+str(i),str(answer[i])]
        ansFile.writerow(a)


# In[ ]:



