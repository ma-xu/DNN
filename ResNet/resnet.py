"""
This is an implement of ResNet.
Author: Xu Ma.
Date: Jan/5/2019
Paper:  "Deep Residual Learning for Image Recognition", Kaiming He.
Remark: Solving the problem of 'degradation', which means more layers incurs more errors.
Refer to : https://github.com/raghakot/keras-resnet
"""
import six
from keras.models import Model
from keras.layers import Input,Activation,Dense,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D,AveragePooling2D
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K


def _handle_image_ordering():
    """
    Handle different backend image channel order.
    :return:
    """
    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS
    if K.image_dim_ordering()=='tf':
        ROW_AXIS = 1
        COL_AXIS = 2
        CHANNEL_AXIS = 3
    else:
        CHANNEL_AXIS = 1
        ROW_AXIS = 2
        COL_AXIS = 3


def _bn_relu(data):
    """
    Adapt batch normalization after each Conv and before Activation.
    Combine BN and Activation layer for convenience.
    :param data:
    :return:
    """
    BN_out=BatchNormalization(axis=CHANNEL_AXIS)(data)
    return Activation('relu')(BN_out)


def _conv_bn_relu(**conv_params):
    """
    Build a conv -> BN -> relu block
    :param conv_params:
    :return:
    """
    filters = conv_params["filters"]
    kernel_size=conv_params["kernel_size"]
    strides = conv_params.setdefault("strides",[1,1])
    # Different initializers, including he_normal,zeros, ones,RandomNormal ...
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    # keep same shape as previous
    padding =conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(data):
        conv = Conv2D(filters=filters,kernel_size=kernel_size,
                      strides=strides,padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(data)
        return  _bn_relu(conv)

    return f


def _bn_relu_conv(**conv_params):
    """
    Build a BN_relu_conv block.
    Reference: http://arxiv.org/pdf/1603.05027v2.pdf
    Section 4.1  Experiments on Activation
    :param conv_params:
    :return:
    """

    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", [1, 1])
    # Different initializers, including he_normal,zeros, ones,RandomNormal ...
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    # keep same shape as previous
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(data):
        activation = _bn_relu(data)
        return Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(activation)

    return f


def _shortcut(input,residual):
    """
    Add a shortcut between input and residual block and merge them with "sum"
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)o
    # Should be int if network architecture is crrectly configured.
    :param data:
    :param residual:
    :return:
    """
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
    equal_channels = (input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS])

    #input channels changes to residual channels
    shortcut = input
    # Using [1,1] kernel size Conv layer to make the shapes same.
    # Actually, we don't have to validate the stride since the feature maps size will not change in each block.--xuma.
    # The above line is WRONG. Validating size is for connected short cut between different sizes feature maps.
    # Which is indicated by dotted line in Fig 3 in the paper.
    if stride_width >1 or stride_height>1 or not equal_channels:
        shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],kernel_size=(1,1),
                          strides=(stride_width,stride_height),padding='valid',
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001))(input)

    return add([shortcut, residual])


def _residual_block(block_function,filters,repetitions,is_first_layer=False):
   """
   Build a residual block with repeating bottleneck blocks.
   :param block_function: The block function to use. This is either `basic_block` or `bottleneck`.
   for layers<50, we use basic lock, which means [3X3,64],[3X3,64]
   for layers>=50, we use bottleneck, which means [1X1,64],[3X3,64],[1X1,256]
   Here the channels is not fixed to 64, 256 et., refer to the original paper.
   :param filters:
   :param repetitions: repeat times
   :param is_first_layer:
   :return:
   """
   def f(data):
        for i in range(repetitions):
            init_strides=(1,1)
            # Described in Table1, Conv3_1,Conv4_1,Conv5_1, stride=2
            if(i == 0) and not is_first_layer:
                init_strides=(2,2)

            data = block_function(filters=filters,init_strides=init_strides,
                                   is_first_block_of_first_layer=(is_first_layer and i==0))(data)

        return data

   return f


def basic_block(filters,init_strides=(1,1),is_first_block_of_first_layer=False):
    """
    Basic 3X3 conv Blocks for use on resnets with layers <=34.
    Follows improved proposed scheme in : http://arxiv.org/pdf/1603.05027v2.pdf
    """
    def f(input):
        if is_first_block_of_first_layer:
            # Don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = Conv2D(filters=filters,kernel_size=(3,3),
                           strides=init_strides,padding="same",
                           kernel_initializer="he_normal",
                           kernel_regularizer=l2(1e-4))(input)
        else:
            conv1= _bn_relu_conv(filters=filters,kernel_size=(3,3),
                                 init_strides=init_strides)(input)

        residual = _bn_relu_conv(filters=filters,kernel_size=(3,3))(conv1)
        return _shortcut(input,residual)

    return f

def bottleneck(filters,init_strides=(1,1),is_first_block_of_first_layer=False):
    """Bottleneck architecture for > 34 layer resnet.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    Returns:
        A final conv layer of filters * 4
    """
    def f(data):
        if is_first_block_of_first_layer:
            # Don't repeat bn->relu since we just did bn->relu->maxpool
            conv_1_1 = Conv2D(filters=filters,kernel_size=(1,1),
                              strides=init_strides,padding="same",
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2(1e-4))(data)

        else:
            conv_1_1 = _bn_relu_conv(ilters=filters,kernel_size=(1,1),
                              strides=init_strides)(data)

        conv_3_3 = _bn_relu_conv(filters=filters,kernel_size=(3,3))(conv_1_1)
        resdual = _bn_relu_conv(filters=filters*4,kernel_size=(1,1))(conv_3_3)
        return _shortcut(data,resdual)
    return f


def _get_block(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier


class ResNetBuilder(object):
    @staticmethod
    def build(input_shape,num_outputs,block_fn,repetations):
        """Builds a custom ResNet like architecture.
        Args:
            input_shape: The input shape in the form (nb_channels, nb_rows, nb_cols)
            num_outputs: The number of outputs at final softmax layer
            block_fn: The block function to use. This is either `basic_block` or `bottleneck`.
                The original paper used basic_block for layers < 50
            repetitions: Number of repetitions of various block units.
                At each block unit, the number of filters are doubled and the input size is halved
        Returns:
            The keras `Model`.
        """
        _handle_image_ordering()
        if len(input_shape)!=3:
            raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

        # Permute dimension order if necessary
        if K.image_dim_ordering() == 'tf':
            input_shape = (input_shape[1], input_shape[2], input_shape[0])

        # Load function from str if needed.
        block_fn = _get_block(block_fn)

        input = Input(shape=input_shape)
        conv1 = _conv_bn_relu(filters=64, kernel_size=(7, 7), strides=(2, 2))(input)
        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1)

        block = pool1
        filters = 64
        for i, r in enumerate(repetations):
            block = _residual_block(block_fn, filters=filters, repetitions=r, is_first_layer=(i == 0))(block)
            filters *= 2

        # Last activation
        block = _bn_relu(block)

        # Classifier block
        block_shape = K.int_shape(block)
        pool2 = AveragePooling2D(pool_size=(block_shape[ROW_AXIS], block_shape[COL_AXIS]),
                                 strides=(1, 1))(block)
        flatten1 = Flatten()(pool2)
        dense = Dense(units=num_outputs, kernel_initializer="he_normal",
                      activation="softmax")(flatten1)

        model = Model(inputs=input, outputs=dense)
        return model

    @staticmethod
    def build_resnet_18(input_shape, num_outputs):
        return ResNetBuilder.build(input_shape, num_outputs, basic_block, [2, 2, 2, 2])

    @staticmethod
    def build_resnet_34(input_shape, num_outputs):
        return ResNetBuilder.build(input_shape, num_outputs, basic_block, [3, 4, 6, 3])

    @staticmethod
    def build_resnet_50(input_shape, num_outputs):
        return ResNetBuilder.build(input_shape, num_outputs, bottleneck, [3, 4, 6, 3])

    @staticmethod
    def build_resnet_101(input_shape, num_outputs):
        return ResNetBuilder.build(input_shape, num_outputs, bottleneck, [3, 4, 23, 3])

    @staticmethod
    def build_resnet_152(input_shape, num_outputs):
        return ResNetBuilder.build(input_shape, num_outputs, bottleneck, [3, 8, 36, 3])










