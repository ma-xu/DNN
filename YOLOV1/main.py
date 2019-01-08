from keras.models import Sequential, Model
from keras.layers import Reshape,Activation,Conv2D,Input,MaxPooling2D,BatchNormalization,Flatten,Dense,Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint,TensorBoard
from keras.optimizers import SGD, Adam,RMSprop
from keras.layers.merge import concatenate
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf
import numpy as np
from tqdm import tqdm # for progress bar
import os
import cv2

labels = open('labels.txt').read().split('\n')
IMAGE_H,IMAGE_W = 448,448
S = 7
B = 2
num_classes = len(labels)
class_weights=np.ones(num_classes,dtype='float32')
ANCHORS=[]