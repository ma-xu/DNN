from keras.engine.topology import Layer
import keras.backend as K


# define a layer in keras fashion.
class spp(Layer):
    """
    Spatial Pyramid Pooling Layer, referred to as SPP.

    Reference:  Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition

    # Arguments:
        pool_list:  list of int
                    List of pooling regions to use. The length of the list is the number of pooling regions.
                    Each int in the list is the number of regions in that pool.
                    e.g. [1,2,4] would be 3 regions with 1, 2*2,4*4 max pooling, so 21 outputs per feature map.

    # Input_shape
        4D tensor with shape:
            (samples.rows,cols,channels) for tensorflow backend.

    #output shape
        2D tensor with shape:
            (samples, channels *sum(i^2 for i in pool_list))
            details can be found in papper.
            (1^2+2^2+4^2)*256=5376


    """

    def __init__(self,pool_list,**kwargs):
        super(spp, self).__init__(**kwargs)
        self.pool_list=pool_list
        self.num_outputs_per_channel=sum([i*i for i in pool_list])
        super(spp, self).__init__(**kwargs)

    # Build is used to define weights, trainable weights should be added here.
    def build(self, input_shape):
        self.nb_channels = input_shape[3]

    # Define the changes of shape here.
    def compute_output_shape(self,input_shape):
        return (input_shape[0], self.nb_channels * self.num_outputs_per_channel)

    # Define the function of this layer.
    def call(self, inputs, mask=None):
        input_shape=K.shape(inputs)
        num_rows=input_shape[1]
        num_cols=input_shape[2]

        # short write for for loop
        # K.cast changes dtype
        row_length=[K.cast(num_rows,'float32')/i for i in self.pool_list]
        col_length = [K.cast(num_cols, 'float32') / i for i in self.pool_list]

        outputs = []

        #pool num is the index, num_pool_regions is the value in pool_list.
        for pool_num, num_pool_regions in enumerate(self.pool_list):
            for jy in range (num_pool_regions):
                for ix in range(num_pool_regions):
                    # 1 for floor, 2 for ceil
                    x1 = ix * col_length[pool_num]
                    x2 = ix * col_length[pool_num] + col_length[pool_num]

                    y1 = jy * row_length[pool_num]
                    y2 = jy * row_length[pool_num] + row_length[pool_num]

                    # Element-wise rounding to the closest integer
                    x1 = K.cast(K.round(x1), 'int32')
                    x2 = K.cast(K.round(x2), 'int32')
                    y1 = K.cast(K.round(y1), 'int32')
                    y2 = K.cast(K.round(y2), 'int32')

                    new_shape = [input_shape[0],y2-y1,x2-x1,input_shape[3]]

                    inputs_crop=inputs[:,y1:y2,x1:x2,:]
                    xm = K.reshape(inputs_crop,new_shape)
                    pooled_val = K.max(xm,axis=(1,2))
                    outputs.append(pooled_val)
        outputs=K.concatenate(outputs)
        return outputs



    # Process parameters.
    def get_config(self):
        config = {"pool_list":self.pool_list}
        base_config = super(spp, self).get_config()
        return dict(list(base_config.items())+list(config.items()))