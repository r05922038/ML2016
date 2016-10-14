
# coding: utf-8

# In[37]:

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
trainSet=np.array([[[0.0]*8]*18])
for i in xrange(12):
    for j in xrange(8):
        if j==0:
            trainSet=np.append(trainSet,np.array([trainSetTemp[i,...,0+x*8:8+x*8] for x in xrange(60)]),axis=0)
        else:
            trainSet=np.append(trainSet,np.array([trainSetTemp[i,...,j+x*8:(8+j)+x*8] for x in xrange(59)]),axis=0)
trainSet=trainSet[1:len(trainSet)]  
#--------------------------------------------------------------------------------------------------------------------
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
a=np.array(testSet,dtype=np.float)[...,...,:8]
b=np.array(testSet,dtype=np.float)[...,...,1:9]
trainSet=np.append(trainSet,a,axis=0)
trainSet=np.append(trainSet,b,axis=0)
#-----------------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------------
trainSet1=np.array([np.append(x[:14,:7].flatten(),
                                  np.append(x[8][2:7]*x[9][2:7],# 5 PM10
                                  np.append(x[2][6:7]*x[9][6:7],# 1 co
                                            x[11][6:7]*x[9][6:7]# 1 rh
                                           )))
                    for x in trainSet])


b=[x[9,7] for x in trainSet]


x=trainSet1
xt=np.transpose(trainSet1)
lambdaI=1000*np.identity(len(trainSet1[0]))
w=np.dot(np.dot(np.linalg.pinv(np.dot(xt,x)+lambdaI),xt),b)

weight=w[:len(w)-7].reshape((18-4,7))

a=list(w[len(w)-7:len(w)-2].reshape((1,5)))# 9 PM10
a.append(w[len(w)-2:len(w)-1])# 
a.append(w[len(w)-1:len(w)])#
xxx_xy_arr=np.array(a)



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
testSet=testSet[...,...,2:9]

answer=[]
for i in xrange(240):

    answer.append(np.sum(testSet[i][:14]*weight)
                  +np.sum(xxx_xy_arr[0]*testSet[i,9,2:7]*testSet[i,8,2:7])#PM10
                  +np.sum(xxx_xy_arr[1]*testSet[i,9,6:7]*testSet[i,2,6:7])#PM10
                  +np.sum(xxx_xy_arr[2]*testSet[i,9,6:7]*testSet[i,11,6:7])#PM10
                  )

answer=np.array(answer)

with open('kaggle_best.csv','wb') as csvfile:
    ansFile=csv.writer(csvfile)
    ansFile.writerow(['id','value'])
    for i in xrange(240):
        a=['id_'+str(i),str(answer[i])]
        ansFile.writerow(a)


# In[ ]:



