
# coding: utf-8

# In[ ]:




# In[ ]:

import pickle
import numpy as np
#import sklearn.metrics.pairwise
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input,Dense,Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
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
test_data=test['data']

#represent code-----------------------------------------------------------------------------------------------------------

# this is the size of our encoded representations
encoding_dim = 256  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(3072,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(3072, activation='sigmoid')(encoded)
# this model maps an input to its reconstruction
autoencoder = Model(input=input_img, output=decoded)
# this model maps an input to its encoded representation
encoder = Model(input=input_img, output=encoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

#x_train = np.array(x_label)
#x_test = np.array(x_unlabel)
x_label_unlabel = np.append(np.array(x_label),np.array(x_unlabel),axis=0)
x_label_unlabel_test=np.append(x_label_unlabel,np.array(test_data),axis=0)
#x_test = np.array(x_unlabel)
x_label_unlabel_test = x_label_unlabel_test.astype('float32') / 255.
#x_test = x_test.astype('float32') / 255.
#x_label_unlabel_test = x_label_unlabel_test.reshape((len(x_label_unlabel_test), np.prod(x_label_unlabel_test.shape[1:])))
#x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

autoencoder.fit(x_label_unlabel_test, x_label_unlabel_test,
                nb_epoch=1000,
                batch_size=20,
                shuffle=True)
encoded_imgs_xtest = encoder.predict(x_label_unlabel_test[50000:])
encoded_imgs_xunlabel = encoder.predict(x_label_unlabel_test[5000:50000])
encoded_imgs_xlabel = encoder.predict(x_label_unlabel_test[:5000])
#pickle.dump(encoded_imgs_xtest,open("encoded_e1000_b20_xtest.p","wb"))
#pickle.dump(encoded_imgs_xunlabel,open("encoded_e1000_b20_xunlabel.p","wb"))
#pickle.dump(encoded_imgs_xlabel,open("encoded_e1000_b20_xlabel.p","wb"))
#encoder.save('encoder_e1000_b20.h5')
#autoencoder.save('autoencoder_e1000_b20.h5')

#part2
#encoded_imgs_xtest=pickle.load(open('encoded_e1000_b20_xtest.p','rb'))
#encoded_imgs_xunlabel=pickle.load(open('encoded_e1000_b20_xunlabel.p','rb'))
#encoded_imgs_xlabel=pickle.load(open('encoded_e1000_b20_xlabel.p','rb'))

#similarity=np.array([max(x) for x in sklearn.metrics.pairwise.cosine_similarity(encoded_imgs_xunlabel,encoded_imgs_xlabel)])
#pred_class_cosine=np.array([y_label[x.argmax()] for x in sklearn.metrics.pairwise.cosine_similarity(encoded_imgs_xunlabel,encoded_imgs_xlabel)])

all_label = pickle.load(open(ALL_LABEL,'rb'))
x_label=np.array(all_label).reshape(5000,3072)
y_label=np.array([0]*500)
for i in xrange(1,10):
    y_label=np.append(y_label,[i]*500)
x_unlabel=pickle.load(open(ALL_UNLABEL,'rb'))
test=pickle.load(open(TEST,'rb'))
test_data=test['data']
X_train=np.array(x_label)
y_train=np.array(y_label)
#X_candidation=np.array(x_unlabel)
X_test=np.array(test_data)

clf = KMeans(n_clusters=10, random_state=0).fit(encoded_imgs_xlabel)
pred_class_kmeans=clf.predict(encoded_imgs_xunlabel)

clf = RandomForestClassifier(n_estimators=10).fit(encoded_imgs_xlabel,y_train)
pred_class_rf=clf.predict(encoded_imgs_xunlabel)

clf = KNeighborsClassifier(n_neighbors=3).fit(encoded_imgs_xlabel,y_train)
pred_class_knn=clf.predict(encoded_imgs_xunlabel)
knn_prob=clf.predict_proba(encoded_imgs_xunlabel)

for i in xrange(45000):
    if pred_class_kmeans[i]==pred_class_rf[i]==pred_class_knn[i]:
        X_train=np.append(X_train,np.array([x_unlabel[i]]),axis=0)
        y_train=np.append(y_train,np.array([pred_class_kmeans[i]]),axis=0)


batch_size = 200
nb_classes = 10
nb_epoch = 1000
data_augmentation = True

if len(X_train)>40000:
    batch_size = 300
elif len(X_train)>30000:
    batch_size = 250
elif len(X_train)>20000:
    batch_size = 200
elif len(X_train)>10000:
    batch_size = 150
else: 
    batch_size = 50

img_rows, img_cols = 32, 32
img_channels = 3
X_train = np.array([np.array(x).reshape(3,1024).T.reshape(32,32,3) for x in X_train])
X_test = np.array([np.array(x).reshape(3,1024).T.reshape(32,32,3) for x in X_test])

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)

print len(X_train)

model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='same',
                        input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

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
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
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
#pickle.dump(pred,open("encode_kmeans_rf_lnn_part"+str(len(X_train))+".p","wb"))



# In[ ]:



