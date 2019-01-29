import os
import time
import json
import densenet
import numpy as np
import keras.backend as K

from keras.datasets import cifar10
from keras.optimizers import SGD
from keras.utils import np_utils


# pre-process the data
(X_train,y_train),(X_test,y_test) = cifar10.load_data()
nb_classes  =len(np.unique(y_train))
img_dim = X_train.shape[1:]

if K.image_data_format()=="channels_first":
    n_channels = X_train.shape[1]
else:
    n_channels = X_train.shape[-1]

# convert class vextors to binary class matrices
Y_train = np_utils.to_categorical(y_train,nb_classes)
Y_test = np_utils.to_categorical(y_test,nb_classes)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

model = densenet.DenseNet(nb_classes,img_dim,depth=7,nb_dense_block=1,
                          growth_rate=12,nb_filter=16,dropout_rate=0.2,weight_decay=1e-4)
model.summary()

batch_size = 64
epochs = 100
lr = 1e-3
# Define optimizer
opt=SGD(lr=lr, decay=1e-6, momentum=0.9)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy']
              )

model.fit(X_train,y_train,batch_size=batch_size,epochs=epochs,validation_data=(X_test,y_test),shuffle=True)

scores = model.evaluate(X_test,y_test,verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
