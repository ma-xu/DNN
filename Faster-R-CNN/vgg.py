import warnings
from keras.models import Model
from keras.layers import Flatten,Dense,Input,Conv2D,MaxPooling2D,Dropout
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D,TimeDistributed

from keras.utils import layer_utils
from keras.engine.topology import get_source_inputs
from keras.utils.data_utils import get_file
from keras import  backend as K
from RoiPoolingConv import RoiPoolingConv


def get_weight_path():
    return 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'


# Get the last feature map size
# This is specifically for VGG16
def get_img_output_length(width,height):
    def get_output_length(input_length):
        #  // for integer
        return input_length // 16
    return get_output_length(width), get_output_length(height)


# Neural Network Base, define the VGG 16
def nn_base(input_tensor=None,trainable=False):
    if K.image_dim_ordering() == 'th':
        input_shape = (3, None, None)
    else:
        input_shape = (None, None, 3)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor,shape=input_shape)
        else:
            img_input = input_tensor

    if K.image_dim_ordering() == 'tf':
        bn_axis = 3 # channels
    else:
        bn_axis = 1

    # Start VGG16

    #Block 1
    x = Conv2D(64,(3,3),activation='relu',padding='same',name='conv1_1')(img_input)
    x = Conv2D(64,(3,3),activation='relu',padding='same',name='conv1_2')(x)
    x = MaxPooling2D((2,2),strides=(2,2),name="pool1_1")(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="pool2_1")(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="pool3_1")(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="pool4_1")(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="pool4_1")(x)

    # Without classification block
    return x


# Region Proposal Network
# Output is the predict class and regression.
def rpn(base_layers, number_anchors):
    x = Conv2D(512, (3, 3), padding='same', activation='relu',
               kernel_initializer='normal', name='rpn_conv1')(base_layers)
    x_class= Conv2D(number_anchors, (1, 1), activation='sigmoid',
                    kernel_initializer='uniform',name='rpn_out_class')(x)
    x_regr = Conv2D(number_anchors*4, (1, 1), activation='linear',
                    kernel_initializer='zero', name='rpn_out_regress')(x)
    return [x_class,x_regr,base_layers]


def classifier(base_layers, input_rois, num_rois, nb_classes =21, trainable=False):
    pooling_regions = 7
    input_shape = (num_rois,7,7,512)
    out_roi_pool = RoiPoolingConv(pooling_regions,num_rois)([base_layers,input_rois])

    out = Flatten(name='flatten')(out_roi_pool)
    out= Dense(4096,activation='relu',name='fc1')(out)
    out=Dropout(0.5)(out)
    out= Dense(4096,activation='relu',name='fc2')(out)
    out=Dropout(0.5)(out)

    out_class = Dense(nb_classes,activation='softmax',
                      kernel_initializer='zero',name='dense_class_{}'.format(nb_classes))(out)
    out_regr = Dense(4*(nb_classes-1),activation='linear',
                     kernel_initializer='zero', name='dense_regress_{}'.format(nb_classes))(out)
    return [out_class, out_regr]













