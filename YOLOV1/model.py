# Refer to: https://github.com/Dunimon/yolo-v1-keras


from keras.models import Sequential,Model
from keras.layers import Reshape,Activation,Conv2D,Input,MaxPooling2D,Flatten,Dense,Lambda
from keras.layers.advanced_activations import LeakyReLU

# Default as https://arxiv.org/pdf/1506.02640.pdf
IMAGE_H, IMAGE_W = 448, 448
S = 7
B = 2
classNum = 20

input_image=Input(shape=(IMAGE_H,IMAGE_W,3))

# Block 1
x = Conv2D(kernel_size=(7,7),filters=64,padding='same',strides=2,name='Conv1_1')(input_image)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2,2),strides=2,name='maxpool1_1')(x)

# Block 2
x = Conv2D(kernel_size=(3,3),filters=192,padding='same',name='Conv2_1')(x)
x=LeakyReLU(alpha=0.1)(x)
x=MaxPooling2D(pool_size=(2,2),strides=2,name="Maxpool2-1")(x)

# Block 3
x = Conv2D(kernel_size=(1,1),filters=128,padding='same',name='Conv3_1')(x)
x=LeakyReLU(alpha=0.1)(x)
x = Conv2D(kernel_size=(3,3),filters=256,padding='same',name='Conv3_2')(x)
x=LeakyReLU(alpha=0.1)(x)
x = Conv2D(kernel_size=(1,1),filters=256,padding='same',name='Conv3_3')(x)
x=LeakyReLU(alpha=0.1)(x)
x = Conv2D(kernel_size=(3,3),filters=512,padding='same',name='Conv3_1')(x)
x=LeakyReLU(alpha=0.1)(x)
x=MaxPooling2D(pool_size=(2,2),strides=2,name="Maxpool3_1")(x)

# Block 4
x = Conv2D(kernel_size=(1,1),filters=256,padding='same',name='Conv4_1')(x)
x = LeakyReLU(alpha=0.1)(x)
x = Conv2D(kernel_size=(3,3),filters=512,padding='same',name='Conv4_2')(x)
x = LeakyReLU(alpha=0.1)(x)
x = Conv2D(kernel_size=(1,1),filters=256,padding='same',name='Conv4_3')(x)
x = LeakyReLU(alpha=0.1)(x)
x = Conv2D(kernel_size=(3,3),filters=512,padding='same',name='Conv4_4')(x)
x = LeakyReLU(alpha=0.1)(x)
x = Conv2D(kernel_size=(1,1),filters=256,padding='same',name='Conv4_5')(x)
x = LeakyReLU(alpha=0.1)(x)
x = Conv2D(kernel_size=(3,3),filters=512,padding='same',name='Conv4_6')(x)
x = LeakyReLU(alpha=0.1)(x)
x = Conv2D(kernel_size=(1,1),filters=256,padding='same',name='Conv4_7')(x)
x = LeakyReLU(alpha=0.1)(x)
x = Conv2D(kernel_size=(3,3),filters=512,padding='same',name='Conv4_8')(x)
x = LeakyReLU(alpha=0.1)(x)

x = Conv2D(kernel_size=(1,1),filters=512,padding='same',name='Conv4_9')(x)
x = LeakyReLU(alpha=0.1)(x)
x = Conv2D(kernel_size=(3,3),filters=1024,padding='same',name='Conv4_10')(x)
x = LeakyReLU(alpha=0.1)(x)
x=MaxPooling2D(pool_size=(2,2),strides=2,name="Maxpool4_1")(x)

# Block 5
x = Conv2D(kernel_size=(1,1),filters=512,padding='same',name='Conv5_1')(x)
x = LeakyReLU(alpha=0.1)(x)
x = Conv2D(kernel_size=(3,3),filters=1024,padding='same',name='Conv5_2')(x)
x = LeakyReLU(alpha=0.1)(x)
x = Conv2D(kernel_size=(1,1),filters=512,padding='same',name='Conv5_3')(x)
x = LeakyReLU(alpha=0.1)(x)
x = Conv2D(kernel_size=(3,3),filters=1024,padding='same',name='Conv5_4')(x)
x = LeakyReLU(alpha=0.1)(x)
x = Conv2D(kernel_size=(3,3),filters=1024,padding='same',name='Conv5_5')(x)
x = LeakyReLU(alpha=0.1)(x)
x = Conv2D(kernel_size=(3,3),filters=1024,padding='same', strides=2,name='Conv5_6')(x)
x = LeakyReLU(alpha=0.1)(x)

# Block 6
x = Conv2D(kernel_size=(3,3),filters=1024,padding='same',name='Conv6_1')(x)
x = LeakyReLU(alpha=0.1)(x)
x = Conv2D(kernel_size=(3,3),filters=1024,padding='same',name='Conv6_2')(x)
x = LeakyReLU(alpha=0.1)(x)

# Block 7
x = Flatten(name='Flatten')(x)
x = Dense(4096,activation='softmax',name="FC1")(x)

x = Dense(S*S*(B*5+classNum),activation='softmax',name="FC2")(x)
output = Reshape((S,S,B*5+classNum))(x)
