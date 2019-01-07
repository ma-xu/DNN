"""
An implement of ResNet on cifar10 dataset.
"""

from keras.datasets import  cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, CSVLogger,EarlyStopping

import numpy as np
import resnet

# Reduce learning rate.
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),cooldown=0,patience=5,min_lr=0.5e-6)
early_stopper = EarlyStopping(min_delta=0.001,patience=10)
csv_logger = CSVLogger('resnet18_cifar10.csv')

batch_size = 32
nb_classes = 10
nb_epoch = 100
data_augmentation = True

#input image dimensions
image_rows, image_cols = 32,32
image_channels = 3

# Load data
(X_train,Y_train),(X_test,Y_test)=cifar10.load_data()

# Convert label to one-hot
Y_train = np_utils.to_categorical(Y_train,nb_classes)
Y_test=np_utils.to_categorical(Y_test,nb_classes)

X_train=X_train.astype('float32')
X_test=X_test.astype('float32')

# Centerilize and Normalize
mean_image = np.mean(X_train,axis=0)
X_train = X_train-mean_image
X_test =X_test - mean_image
X_train/=128.
X_test/=128.

model = resnet.ResNetBuilder.build_resnet_18((image_channels,image_rows,image_cols),nb_classes)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
