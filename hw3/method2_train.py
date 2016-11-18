
# coding: utf-8

# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
import pickle
import numpy as np
import sklearn.metrics.pairwise
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input,Dense,Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import h5py
import sys
from keras import backend as K
K.set_image_dim_ordering('tf')

path_=sys.argv[1]
ALL_LABEL=path_+'all_label.p'
ALL_UNLABEL=path_+'all_unlabel.p'
TEST=path_+'test.p'
model_name=sys.argv[2]
MODEL=model_name+'.h5'

all_label = pickle.load(open(ALL_LABEL,'rb'))

x_label=np.array(all_label).reshape(5000,3072)
y_label=np.array([0]*500)
for i in xrange(1,10):
    y_label=np.append(y_label,[i]*500)

x_unlabel=pickle.load(open(ALL_UNLABEL,'rb'))

test=pickle.load(open(TEST,'rb'))
test_id=test['ID']
test_data=test['data']

#TRAIN_data-------------------------------------------------------------------------------------------------
# input image dimensions
img_rows, img_cols = 32, 32

# the data, shuffled and split between train and test sets
#(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train=np.array(x_label)
y_train=np.array(y_label)
X_candidate_train=np.array(x_unlabel)
#X_validation=np.array(x_label[4500:])
#y_validation=np.array(y_label[4500:])
X_test=np.array(test_data)

#if K.image_dim_ordering() == 'th':
X_train = np.array([np.array(x).reshape(3,1024).T.reshape(32,32,3) for x in X_train])
X_candidate_train=np.array([np.array(x).reshape(3,1024).T.reshape(32,32,3) for x in X_candidate_train])
#X_validation = np.array([np.ravel(x, order='F').reshape(1024,3).reshape(32,32,3) for x in X_validation])
X_test = np.array([np.array(x).reshape(3,1024).T.reshape(32,32,3) for x in X_test])
input_shape = X_train.shape[1:]
#else:
#    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
#    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
#    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_candidate_train=X_candidate_train.astype('float32')
#X_validation=X_validation.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_candidate_train /= 255
#X_validation /= 255
X_test /= 255

# convert class vectors to binary class matrices
nb_classes=10
Y_train = np_utils.to_categorical(y_train, nb_classes)
#Y_validation= np_utils.to_categorical(y_validation, nb_classes)

batch_size = 100
nb_classes = 10
nb_epoch = 500
pool_size = (2, 2)
#data_augmentation = True


limit=3
while limit>0:#updata X_train,y_train
    limit-=1  
    
    if len(X_train)>40000:
        batch_size = 200
        nb_epoch = 400
    elif len(X_train)>35000:
        batch_size = 200
        nb_epoch = 400
    elif len(X_train)>30000:
        batch_size = 300
        nb_epoch = 400
    elif len(X_train)>20000:
        batch_size = 200
        nb_epoch = 400
    elif len(X_train)>10000:
        batch_size = 200
        nb_epoch = 400
    else:
        batch_size = 100
        nb_epoch = 800
    
    
   
    pool_size = (2, 2)

    data_augmentation = True 
    
    model = Sequential()

    model.add(Convolution2D(32, 3, 3,border_mode='same',input_shape=input_shape))
    model.add(LeakyReLU())
    model.add(Convolution2D(32, 3, 3))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3,border_mode='same'))
    model.add(LeakyReLU())
    model.add(Convolution2D(64, 3, 3))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))

    model.add(Flatten())
    
    model.add(Dense(1024))
    model.add(LeakyReLU())
    model.add(Dropout(0.25))
    
    model.add(Dense(512))
    model.add(LeakyReLU())
    model.add(Dropout(0.5))
    
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

#    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#    model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])

    #model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,shuffle=True)
    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(X_train, Y_train,
                  batch_size=batch_size,
                  nb_epoch=nb_epoch,
                  validation_data=None,
                  shuffle=True)
    else:
        print('Using real-time data augmentation.')

        # this will do preprocessing and realtime data augmentation
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        # compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied)
        datagen.fit(X_train)

        # fit the model on the batches generated by datagen.flow()
        model.fit_generator(datagen.flow(X_train, Y_train,
                                         batch_size=batch_size),
                            samples_per_epoch=X_train.shape[0],
                            nb_epoch=nb_epoch,
                            validation_data=None)

        
#    if len(X_train)>20000:
#        pred_=model.predict_classes(X_test,batch_size=batch_size)
#        pickle.dump(pred_,open("cifar10_cnn_selfLlearning_"+str(len(X_train))+"_ytest_s_2.p","wb"))
    
    pred=model.predict(X_candidate_train,batch_size=batch_size)
    y_candidate_train_id_pred_class=zip(range(len(pred)),np.array([max(x) for x in pred]),np.array([x.argmax() for x in pred]))
    temp_id=[]
    updata_trainSet=0
    for x in y_candidate_train_id_pred_class:
        if x[1]>0.97:
            updata_trainSet+=1
            X_train=np.append(X_train,np.array([X_candidate_train[x[0]]]),axis=0)
            y_train=np.append(y_train,np.array([x[2]]),axis=0)
            temp_id.append(x[0])
            
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    X_candidate_train=np.delete(X_candidate_train,temp_id,0)
    
#    pickle.dump(X_train,open("cifar10_cnn_selfLlearning_"+str(len(X_train))+"_Xtrain_s_2.p","wb"))
#    pickle.dump(y_train,open("cifar10_cnn_selfLlearning_"+str(len(y_train))+"_ytrain_s_2.p","wb"))
#    pickle.dump(X_candidate_train,open("cifar10_cnn_selfLlearning_"+str(len( X_candidate_train))+"_XCandtrain_s_2.p","wb"))


#    model.save('my_model'+str(limit)+'_s_2.h5')  # creates a HDF5 file 'my_model.h5'
#    del model  # deletes the existing model
    
    if len(X_candidate_train)<15000 or updata_trainSet<1000:
        break

#predict------------------------------------------------------------------------------------------------------------------------
batch_size = 200
nb_classes = 10
nb_epoch = 400
data_augmentation = True

# input image dimensions
img_rows, img_cols = 32, 32
# the CIFAR10 images are RGB
img_channels = 3

model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='same',
                        input_shape=X_train.shape[1:]))
model.add(LeakyReLU())
model.add(Convolution2D(32, 3, 3))
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(LeakyReLU())
model.add(Convolution2D(64, 3, 3))
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024))
model.add(LeakyReLU())
model.add(Dropout(0.25))
model.add(Dense(512))
model.add(LeakyReLU())
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# let's train the model using SGD + momentum (how original).
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss='categorical_crossentropy',
#              optimizer=sgd,
#              metrics=['accuracy'])
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy'])

if not data_augmentation:
#    print('Not using data augmentation.')
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=None,
              shuffle=True)
else:
#    print('Using real-time data augmentation.')

    # this will do preprocessing and realtime data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(X_train)

    # fit the model on the batches generated by datagen.flow()
    model.fit_generator(datagen.flow(X_train, Y_train,
                        batch_size=batch_size),
                        samples_per_epoch=X_train.shape[0],
                        nb_epoch=nb_epoch,
                        validation_data=None)

model.save(MODEL)
    
#pred=model.predict_classes(X_test,batch_size=batch_size)
#pickle.dump(pred,open("cifar10_cnn_selfLlearning_s_2.p","wb"))
#model.save('cifar10_cnn_selfLlearning_model_s_2.h5')  # creates a HDF5 file 'my_model.h5'
#del model  # deletes the existing model



