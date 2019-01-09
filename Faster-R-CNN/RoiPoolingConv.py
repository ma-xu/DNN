from keras.engine.topology import Layer
import keras.backend as K
import tensorflow as tf


class RoiPoolingConv(Layer):
    """ROI pooling layer for 2D inputs
    # Parameters:
        pool_size: int
            Size of pooling region to use. pool_size = 7 will result in a 7X7 region.
        num_rois: number of regions of interest to be used.
    # Input shape
        list of two 4D tensors [x_img,X_roi] with shape:
        X_img:
            (1,channels,rows,cols) if dim_ordering = "th"
            or
            (1,rows,cols,channels) if dim_ordering = "tf"
        X_roi:
            (1,num_rois,4) list of rois,with ordering (x,y,w,h)
    # Output shape
        3D tensor with shape:
        (1, num_rois,channels,pool_size,pool_size)
    """

    def __init__(self,pool_size,num_rois,**kwargs):
        self.dim_ordering = K.image_dim_ordering()
        assert  self.dim_ordering in {"tf","th"},'dim_ordering must be in {tf, th}'
        self.pool_size = pool_size
        self.num_rois = num_rois
        super(RoiPoolingConv, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.dim_ordering == 'th':
            self.nb_channels = input_shape[0][1]
        elif self.dim_ordering == 'tf':
            #input_shape should be :[[num_pics,rows,cols,channels]]
            self.nb_channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        if self.dim_ordering == 'th':
            return None,self.num_rois,self.nb_channels,self.pool_size,self.pool_size
        else:
            return None,self.num_rois,self.pool_size,self.pool_size,self.nb_channels

    def call(self, x, mask=None):
        """x is inputs, a list of two 4D tensors [x_img,X_roi]
        """
        assert (len(x) == 2)
        img = x[0]
        rois = x[1]
        input_shape = K.shape(img)
        outputs = []
        for roi_idx in range(self.num_rois):
            #rois: (1,num_rois,4) list of rois,with ordering (x,y,w,h)
            x = rois[0, roi_idx, 0]
            y = rois[0, roi_idx, 1]
            w = rois[0, roi_idx, 2]
            h = rois[0, roi_idx, 3]

            # This is just suitable for TF
            x = K.cast(x, 'int32')
            y = K.cast(x, 'int32')
            w = K.cast(x, 'int32')
            h = K.cast(x, 'int32')
            # TF  can resize image to certain size.
            rs = tf.image.resize_images(img[:, y:y+h, x:x+w, :],
                                        (self.pool_size, self.pool_size))
            outputs.append(rs)

        final_outputs = K.concatenate(outputs,axis=0)
        final_output = K.reshape(final_outputs,
                                  (1, self.num_rois, self.pool_size,
                                   self.pool_size, self.nb_channels))

        # For different dim in th and tf.
        if self.dim_ordering == 'th':
            final_output = K.permute_dimensions(final_output, (0, 1, 4, 2, 3))
        else:
            final_output = K.permute_dimensions(final_output, (0, 1, 2, 3, 4))

        return final_output

    def get_config(self):
        config = {'pool_size': self.pool_size, 'num_rois': self.num_rois}
        base_confg= super(RoiPoolingConv, self).get_config()
        return dict(list(base_confg.items()) + list(config.items()))




















        