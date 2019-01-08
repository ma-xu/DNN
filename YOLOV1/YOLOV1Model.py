"""
Define my yolov1 model class.

"""
from keras.models import Sequential,Model
from keras.layers import Reshape,Activation,Conv2D,Input,MaxPooling2D,Flatten,Dense,Lambda
from keras.layers.advanced_activations import LeakyReLU

def get_model():
     model = Sequential()
     model.add(Conv2D(kernel_size=(7,7),filters=64,input_shape=(448,448,3),padding='same',strides=2,name='Conv1_1'))
