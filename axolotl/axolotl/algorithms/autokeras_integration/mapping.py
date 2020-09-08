from .block import *


def fetch_conv1D_step(block_id, layer):
    return Conv1D(
        block_id,
        layer.filters,
        layer.kernel_size,
        layer.strides,
        layer.padding,
        layer.previous_layer_ids[0]
    ).get_step()


def fetch_conv2D_step(block_id, layer):
    return Conv2D(
        block_id,
        layer.filters,
        layer.kernel_size,
        layer.strides,
        layer.padding,
        layer.previous_layer_ids[0]
    ).get_step()


def fetch_conv3D_step(block_id, layer):
    return Conv3D(
        block_id,
        layer.filters,
        layer.kernel_size,
        layer.strides,
        layer.padding,
        layer.previous_layer_ids[0]
    ).get_step()


def fetch_dense_step(block_id, layer):
    return Dense(
        block_id,
        layer.units,
        layer.activation,
        layer.previous_layer_ids[0]
    ).get_step()


def fetch_batch_norm_step(block_id, layer):
    return BatchNorm2D(
        block_id,
        layer.previous_layer_ids[0]
    ).get_step()


def fetch_maxpool2d_step(block_id, layer):
    return MaxPooling2D(
        block_id,
        layer.pool_size,
        layer.strides,
        layer.padding,
        layer.previous_layer_ids[0]
    ).get_step()


def fetch_avgpool2d_step(block_id, layer):
    return AvgPooling2D(
        block_id,
        layer.pool_size,
        layer.strides,
        layer.padding,
        layer.previous_layer_ids[0]
    ).get_step()


def fetch_dropout_step(block_id, layer):
    return Dropout(
        block_id,
        layer.rate,
        layer.previous_layer_ids[0]
    ).get_step()


# JPL does not have such primitives,
# def fetch_global_maxpooling_step(block_id, layer):
#     return GlobalMaxPooling2d(
#         block_id,
#         layer.input.shape,
#         layer.previous_layer_ids[0]
#     ).get_step()


def fetch_global_avgpooling_step(block_id, layer):
    return GlobalAvgPooling2d(
        block_id,
        layer.data_format,
        layer.previous_layer_ids[0]
    ).get_step()


def fetch_flatten_step(block_id, layer):
    return Flatten(
        block_id,
        layer.previous_layer_ids[0]
    ).get_step()


def fetch_add_step(block_id, layer):
    return Add(
        block_id,
        layer.previous_layer_ids
    ).get_step()


def fetch_concatenate_step(block_id, layer):
    return Concatenate(
        block_id,
        layer.previous_layer_ids
    ).get_step()


def fetch_null_step(block_id):
    return Null(
        block_id,
    ).get_step()
