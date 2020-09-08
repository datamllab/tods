from .mapping import *

step_function = {
    'Dense': fetch_dense_step,
    'Conv1D': fetch_conv1D_step,
    'Conv2D': fetch_conv2D_step,
    'Conv3D': fetch_conv3D_step,
    'BatchNormalization': fetch_batch_norm_step,
    'MaxPooling2D': fetch_maxpool2d_step,
    'Dropout': fetch_dropout_step,
    'AvgPooling2D': fetch_avgpool2d_step,
    # 'GlobalMaxPooling2d': JPL does not have such primitives,
    'GlobalAveragePooling2D': fetch_global_avgpooling_step,
    'Flatten': fetch_flatten_step,
    'Add': fetch_add_step,
    'Concatenate': fetch_concatenate_step,
    'Null': fetch_null_step,
    # 'Substract': we do not implement
}

ACTIVATIONS = {'ReLU'}
OMIT_LAYERS = {'InputLayer', 'Normalization', 'ReLU', 'ZeroPadding2D', 'Softmax', 'Activation'}
FORWARD_LAYERS = {'Dense', 'Conv1d', 'Conv2d', 'Conv3d'}
