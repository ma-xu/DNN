from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np
import cv2
import random
from keras import backend as K

selected_layer = 'block5_conv3'

base_model = VGG16(weights='imagenet',include_top=False)
# Choose which layer to output
#base_model = Model(inputs=base_model.input,outputs =base_model.get_layer('block5_conv3').output)
base_model.summary()

image_path='003758.png';
img = image.load_img(image_path);
x = image.img_to_array(img)
print(x.shape)
x = np.expand_dims(x,axis=0)
x = preprocess_input(x)

# Output of the base model
base_output = base_model.predict(x)


inter_tensor_function = K.function([base_model.layers[0].input],[base_model.get_layer(selected_layer).output])
inter_tensor = inter_tensor_function([x])[0]
print(inter_tensor.shape)
channels = inter_tensor.shape[3]
print(channels)
for i in range(20):

    rand_channel = random.randint(0,channels-1)

    cv2.imshow(selected_layer+"_channel_"+str(rand_channel),inter_tensor[0,:,:,rand_channel])
    cv2.imwrite('aaa.png',inter_tensor[0,:,:,124])
    cv2.waitKey(0)
    cv2.destroyAllWindows();