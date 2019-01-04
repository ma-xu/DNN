"""
Load a saved model for prediction.

"""

import keras
from tensorflow import expand_dims
import os
from keras.datasets import cifar10
from keras import backend as K
from keras.models import load_model
from keras.callbacks import TensorBoard,ModelCheckpoint
from keras import Input

num_class = 10
batch_size = 32
epochs = 20
save_dir = os.path.join(os.getcwd(),'saved_models')
model_name = 'keras_cifar10_train_on_pretrained__model.h5'

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

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

model = load_model('./saved_models/keras_cifar10_train_on_pretrained__model.h5')
model.summary()

# Get a layer by name
layer4_3 = model.get_layer("block4_conv3")
print(layer4_3.output)

# Get the output of certain layer.
data= expand_dims(x_test[0],0)
input_t= Input(shape=x_train.shape[1:])
layer_model= K.function([model.layers[0].input],[model.get_layer('block4_conv3').output])
layer_output=layer_model([data])[0]
# print(model.input.shape)
# layer_model = keras.Model(inputs=model.input, outputs=model.get_layer('block4_conv3').output)
# layer_outputs=layer_model.predict(data,steps=1)
# print(layer_outputs.shape)
















# Test model
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

# Define optimizer
opt=keras.optimizers.sgd(lr=0.01, decay=1e-6)

# Remove metrics will improve training speed because it will ignore the validation step.
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy']
              )


# Use tensorboard
tbCallBack=TensorBoard(log_dir='./logs',write_grads=True,write_graph=True,write_images=True)

#Save model each 5 epoches,
#Didn't added to callbacks
cpCallBack=ModelCheckpoint(filepath="checkpoint-{epoch:02d}e-val-loss_{val_loss:.2f}.h5",
                           period=5)


# Two validation schemes.
model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              #validation_split=0.1,
              callbacks=[tbCallBack],
              shuffle=True)


model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Test model Again.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

