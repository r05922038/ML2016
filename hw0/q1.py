
# coding: utf-8

# In[76]:

import sys
from operator import itemgetter
import numpy as np
filename=sys.argv[2]
column=int(sys.argv[1])
with open(filename) as f:
#with open('test.dat') as f:    
    content = f.readlines()
    matrix=[[0.000*11]]
    for i in xrange(len(content)):
        str1=content[i].split(" ")
        str1.pop(0)
    
        matrix.append([float(i) for i in str1])
    matrix.pop(0)
    
    s=sorted(map(itemgetter(column), matrix))
#    s=sorted(map(itemgetter(1), matrix))
f.close()

s1=",".join([str(x) for x in s])
f1 = open('ans1.txt', 'w')
f1.write(s1) 
f1.close()
  

