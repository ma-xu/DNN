"""
Training VGG16 using cifar10 datasets.

Some functions:
    1. Different optimizers.
    2. Two validation schemes.
    3. Save checkpoint, the same as snapshot in Caffe.
    4. Use TensorBoard.
"""

import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Conv2D,MaxPooling2D
# import TensorBoard
from keras.callbacks import TensorBoard,ModelCheckpoint
import os

batch_size = 32
num_class = 10
epochs = 40
save_dir = os.path.join(os.getcwd(),'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

# import data
(x_train,y_train),(x_test,y_test)=cifar10.load_data()
print('x_train shape: ',x_train.shape)
print("training pictures number: ",x_train.shape[0])
print("testing pictures number: ",x_test.shape[0])

y_test=keras.utils.to_categorical(y_test,num_class)
y_train=keras.utils.to_categorical(y_train,num_class)
print("Training label shape: ",y_train.shape)

model = Sequential()

# Block 1
# padding same means output has the same shape as input. padding valid means no padding.
model.add(Conv2D(64, (3, 3),
                 padding='same',
                 name='block1_conv1',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(64,(3,3),
                 padding='same',
                 name='block1_conv2'))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2),
                       strides=(2,2),
                       name='block1_pool'))

# Block 2
model.add(Conv2D(128,(3,3),
                 padding='same',
                 name='block2_conv1'))
model.add(Activation('relu'))
model.add(Conv2D(128,(3,3),
                 padding='same',
                 name='block2_conv2'))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2),
                       strides=(2,2),
                       name='block2_pool'))

# Block 3
model.add(Conv2D(256,(3,3),
                 padding='same',
                 name='block3_conv1'))
model.add(Activation('relu'))
model.add(Conv2D(256,(3,3),
                 padding='same',
                 name='block3_conv2'))
model.add(Activation('relu'))
model.add(Conv2D(256,(3,3),
                 padding='same',
                 name='block3_conv3'))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2),
                       strides=(2,2),
                       name='block3_pool'))

# Block 4
model.add(Conv2D(512,(3,3),
                 padding='same',
                 name='block4_conv1'))
model.add(Activation('relu'))
model.add(Conv2D(512,(3,3),
                 padding='same',
                 name='block4_conv2'))
model.add(Activation('relu'))
model.add(Conv2D(512,(3,3),
                 padding='same',
                 name='block4_conv3'))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2),
                       strides=(2,2),
                       name='block4_pool'))

# Block 5
model.add(Conv2D(512,(3,3),
                 padding='same',
                 name='block5_conv1'))
model.add(Activation('relu'))
model.add(Conv2D(512,(3,3),
                 padding='same',
                 name='block5_conv2'))
model.add(Activation('relu'))
model.add(Conv2D(512,(3,3),
                 padding='same',
                 name='block5_conv3'))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2),
                       strides=(2,2),
                       name='block5_pool'))

# Block Classification
model.add(Flatten(name='flatten'))
model.add(Dense(4096,name='fc1'))
model.add(Activation('relu'))
model.add(Dense(4096,name="fc2"))
model.add(Activation('relu'))
model.add(Dense(num_class,name='predictions'))
model.add(Activation('softmax'))

# Define optimizer
opt=keras.optimizers.sgd(lr=0.01, decay=1e-6, momentum=0.9)
opt=keras.optimizers.adadelta(lr=0.01, decay=1e-5)

model.summary()

# Print model layers length
print(len(model.layers))

# Get a layer by name
layer4_3 = model.get_layer("block4_conv3")
print(layer4_3.output)

# Remove metrics will improve training speed because it will ignore the validation step.
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy']
              )

# Use tensorboard
tbCallBack=TensorBoard(log_dir='./logs',write_grads=True,write_graph=True,write_images=True)

#Save model each 5 epoches
cpCallBack=ModelCheckpoint(filepath="checkpoint-{epoch:02d}e-val-loss_{val_loss:.2f}.h5",
                           period=5)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Two validation schemes.
model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              #validation_split=0.1,
              callbacks=[tbCallBack,cpCallBack],
              shuffle=True)


model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Test model
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])




