
# coding: utf-8

# In[ ]:




# In[ ]:

import pickle
import numpy as np
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input,Dense,Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import h5py
from keras.models import load_model
import csv
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
import sys
from keras import backend as K
K.set_image_dim_ordering('tf')

path_=sys.argv[1]
TEST=path_+'test.p'
model_name=sys.argv[2]
MODEL=model_name+'.h5'
PRED=sys.argv[3]

test=pickle.load(open(TEST,'rb'))
test_id=test['ID']
test_data=test['data']
X_test=np.array(test_data)
X_test = np.array([np.array(x).reshape(3,1024).T.reshape(32,32,3) for x in X_test])
X_test = X_test.astype('float32')
X_test /= 255

model = load_model(MODEL)
batch_size = 200
pred=model.predict_classes(X_test,batch_size=batch_size)

with open(PRED,"wb") as csvfile:
    ansFile=csv.writer(csvfile)
    ansFile.writerow(['ID','class'])
    for i in xrange(10000):
        a=[str(test_id[i]),str(pred[i])]
        ansFile.writerow(a)


