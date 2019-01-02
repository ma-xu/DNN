"""
Predict an image using pre-trained built-in model.

"""


from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.preprocessing import image
import numpy as np

# include_top means if include the decision block (3 Dense layers and softmax activation layer)
model=VGG16(weights='imagenet',include_top=True)
print(len(model.layers))
model.summary()

img=image.load_img('demo1.jpeg',target_size=(224,224))
x=image.img_to_array(img)
print(x.shape)
x=np.expand_dims(x,axis=0)
x=preprocess_input(x)
predicts=model.predict(x)
print('Predicted:', decode_predictions(predicts,top=3)[0])

