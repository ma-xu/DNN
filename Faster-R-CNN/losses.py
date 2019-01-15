from keras import backend as K
from keras.objectives import categorical_crossentropy
import tensorflow as tf

#RPN
lambda_rpn_regr = 1.0
lambda_rpn_class = 1.0

# Classification
lambda_cls_regr = 1.0
lambda_cls_class = 1.0

epsilon = 1e-4

# ref to Fast formulation (2)(3) and Faster (1)
def rpn_loss_regr(num_anchors):
    def rpn_loss_regr_fixed_num(y_true,y_pred):
        x = y_true[:,:,:,4*num_anchors:]-y_pred
        x_abs = K.abs(x)
        x_bool = K.cast(K.less_equal(x_abs,1.0), tf.float32)

        return lambda_rpn_regr * K.sum(
            y_true[:,:,:,:4*num_anchors]*(
                    x_bool*(0.5*x*x)+(1-x_bool)*(x_abs-0.5)
            ))/K.sum(epsilon+y_true[:,:,:,:4*num_anchors])
    return rpn_loss_regr_fixed_num

def rpn_loss_cls(num_anchors):
    def rpn_loss_cls_fixed_num(y_true,y_pred):
        return lambda_rpn_class 