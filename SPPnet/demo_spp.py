import keras.backend as K
import numpy as np
import keras

from keras.models import Sequential
from keras.datasets import cifar10

# Must import like this. First SPP is a module ,next SPP is the class we want.
# Otherwise will retrun module object is not callable.
from SPP import SPP

# As default of SPP
pooling_regions = [1, 2, 4]
number_channels = 3
batch_size = 16
input_shape = (None, None, number_channels)


model = Sequential()
model.add()
# Output shape of SPP layer is sample_numbers*(21*number*channels) 21 is the default of SPP[1, 2,4]
model.add(SPP(pooling_regions, input_shape=input_shape, name="SPPooling"))
model.summary()

