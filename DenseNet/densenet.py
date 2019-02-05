"""
refer to https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/DenseNet
This is design for cifar10 dataset.
"""
from keras.models import Model
from keras.layers.core import Dense,Dropout,Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import AveragePooling2D,GlobalAveragePooling2D
from keras.layers import Input,Concatenate
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import keras.backend as K


def conv_factory(x,concat_axis,nb_filter,dropout_rate=None,weight_decay=1e-4):
    """
    Apply BtachNorm, Relu, 3X3 Conv2D, optional dropout
    Note that each “conv” layer shown in the table corresponds the sequence BN-ReLU-Conv.
    refer to paper Table1 caption
    :param x:
    :param concat_axis:
    :param nb_filter:
    :param dropout_rate:
    :param weight_decay:
    :return:
    """

    x = BatchNormalization(axis=concat_axis,gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x=Conv2D(nb_filter,(3,3),kernel_initializer='he_uniform',padding='same',
             use_bias=False,kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x=Dropout(dropout_rate)(x)
    return x


def transition(x,concat_axis,nb_filter,dropout_rate=None,weight_decay=1e-4):
    """
    Apply BatchNorm, Relu,1x1Conv2D,optional dropout and Maxpooling2D
    :param x:
    :param concat_axis:
    :param nb_filter:
    :param dropout_rate:
    :param weight_decay:
    :return:
    """
    x = BatchNormalization(axis=concat_axis,gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x =Conv2D(nb_filter,(1,1),kernel_initializer='he_uniform',padding='same',
              use_bias=False,kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    x = AveragePooling2D((2,2),strides=(2,2))(x)
    return x


def dense_block(x, concat_axis, nb_layers, nb_filter,growth_rate,dropout_rate=None, weight_decay=1e-4):
    """
    Build a denseblock
    :param x:
    :param concat_axis:
    :param nb_layers:
    :param nb_filter:
    :param growth_rate:
    :param dropout_rate:
    :param weight_decay:
    :return:
    """
    list_feat=[x]
    for i in range(nb_layers):
        x = conv_factory(x,concat_axis, growth_rate,dropout_rate,weight_decay)
        list_feat.append(x)
        x=Concatenate(axis=concat_axis)(list_feat)
        nb_filter +=growth_rate
    return x, nb_filter


def DenseNet(nb_classes,img_dim,depth, nb_dense_block,growth_rate,nb_filter, dropout_rate=None, weight_decay=1e-4):
    """
    This is  different from the architecture in the paper, because it's designed for cifar10 dataset (32 by 32)
    The architecture in paper is designed for ImageNet (224 by 224)
    :param bn_classes:
    :param img_dim:
    :param depth:
    :param nb_dense_block:
    :param growth_rate:
    :param nb_filter:
    :param dropout_rate:
    :param weight_decay:
    :return:
    """
    if K.image_dim_ordering()=='th':
        concat_axis = 1
    elif K.image_dim_ordering() =='tf':
        concat_axis = -1

    model_input = Input(shape=img_dim)

    # Depth must be 3 N + 4 ??? should be K*N+K0
    assert (depth -4) %3  ==0

    nb_layers = int((depth-4)/3)

    #Inital convolution
    x = Conv2D(nb_filter,(3,3),kernel_initializer='he_uniform',padding='same',
               name='initial_conv2D',use_bias=False,kernel_regularizer=l2(weight_decay))(model_input)

    #Add dense blocks
    for block_idx in range(nb_dense_block-1):
        x, nb_filter = dense_block(x,concat_axis,nb_layers,nb_filter,growth_rate,
                                  dropout_rate=dropout_rate,weight_decay=weight_decay)
        # add transition
        x = transition(x,nb_filter,dropout_rate=dropout_rate,weight_decay=weight_decay)

    # The last denseblock does not have a transition
    x, nb_filter = dense_block(x,concat_axis,nb_layers,nb_filter,growth_rate,
                               dropout_rate=dropout_rate,weight_decay=weight_decay)

    x = BatchNormalization(axis=concat_axis,gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)

    x = Activation('relu')(x)
    x = GlobalAveragePooling2D(data_format=K.image_data_format())(x)
    x = Dense(nb_classes,activation='softmax',kernel_regularizer=l2(weight_decay),bias_regularizer=(weight_decay))(x)

    densenet = Model(inputs=[model_input],outputs=[x],name='DenseNet')

    return densenet
