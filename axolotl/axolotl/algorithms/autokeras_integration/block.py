from d3m import index
from d3m.metadata.pipeline import PrimitiveStep
from d3m.metadata.base import ArgumentType


class Block:
    def __init__(self, block_id, primitive, previous_layer_id):
        self.block_id = block_id
        self.primitive = primitive
        self.previous_layer_id = previous_layer_id

    def get_step(self):
        step = PrimitiveStep(primitive=index.get_primitive(self.primitive))
        if self.previous_layer_id is not None:
            step.add_hyperparameter(name='previous_layer', argument_type=ArgumentType.PRIMITIVE,
                                    data=self.previous_layer_id)
        return step


class Conv(Block):
    def __init__(self, filters, kernel_size, strides, padding, block_id, primitive, previous_layer_id):
        super(Conv, self).__init__(block_id, primitive, previous_layer_id)
        self.filters = filters
        self.kernel_size = kernel_size[0]
        self.strides = strides[0]
        self.padding = 'same' if padding else 'valid'
    
    def get_step(self):
        step = super().get_step()
        step.add_hyperparameter(name='filters', argument_type=ArgumentType.VALUE, data=self.filters)
        step.add_hyperparameter(name='kernel_size', argument_type=ArgumentType.VALUE, data=self.kernel_size)
        step.add_hyperparameter(name='strides', argument_type=ArgumentType.VALUE, data=self.strides)
        step.add_hyperparameter(name='padding', argument_type=ArgumentType.VALUE, data=self.padding)
        return step


class Conv1D(Conv):
    def __init__(self, block_id, filters=10, kernel_size=2, strides=1, padding='valid', previous_layer_id=None):
        super(Conv1D, self).__init__(filters, kernel_size, strides, padding, block_id,
                                     "d3m.primitives.layer.convolution_1d.KerasWrap", previous_layer_id)


class Conv2D(Conv):
    def __init__(self, block_id, filters=10, kernel_size=2, strides=1, padding='valid', previous_layer_id=None):
        super(Conv2D, self).__init__(filters, kernel_size, strides, padding, block_id,
                                     "d3m.primitives.layer.convolution_2d.KerasWrap", previous_layer_id)


class Conv3D(Conv):
    def __init__(self, block_id, filters=10, kernel_size=2, strides=1, padding='valid', previous_layer_id=None):
        super(Conv3D, self).__init__(filters, kernel_size, strides, padding, block_id,
                                     "d3m.primitives.layer.convolution_3d.KerasWrap", previous_layer_id)


class Dense(Block):
    def __init__(self, block_id, units=120, activation='linear', previous_layer_id=None):
        super(Dense, self).__init__(block_id, "d3m.primitives.layer.dense.KerasWrap", previous_layer_id)
        self.units = units
        self.activation = activation.__name__.lower()

    def get_step(self):
        step = super().get_step()
        step.add_hyperparameter(name='units', argument_type=ArgumentType.VALUE, data=self.units)
        step.add_hyperparameter(name='activation', argument_type=ArgumentType.VALUE, data=self.activation)
        return step


class BatchNorm2D(Block):
    def __init__(self, block_id, previous_layer_id):
        super(BatchNorm2D, self).__init__(block_id, "d3m.primitives.layer.batch_normalization.KerasWrap",
                                          previous_layer_id)

    def get_step(self):
        step = super().get_step()
        return step


class MaxPooling(Block):
    def __init__(self, pool_size, strides, padding, block_id, primitive, previous_layer_id):
        super(MaxPooling, self).__init__(block_id, primitive, previous_layer_id)
        self.pool_size = pool_size
        self.strides = strides[0]
        self.padding = 'same' if padding else 'valid'

    def get_step(self):
        step = super().get_step()
        step.add_hyperparameter(name='pool_size', argument_type=ArgumentType.VALUE, data=self.pool_size)
        step.add_hyperparameter(name='strides', argument_type=ArgumentType.VALUE, data=self.strides)
        step.add_hyperparameter(name='padding', argument_type=ArgumentType.VALUE, data=self.padding)
        return step


class MaxPooling1D(MaxPooling):
    def __init__(self, block_id, pool_size=(2, 2), strides=(1, 1), padding='valid', previous_layer_id=None):
        super(MaxPooling1D, self).__init__(pool_size, strides, padding, block_id,
                                           "d3m.primitives.layer.max_pooling_1d.KerasWrap", previous_layer_id)


class MaxPooling2D(MaxPooling):
    def __init__(self, block_id, pool_size=(2, 2), strides=(1, 1), padding='valid', previous_layer_id=None):
        super(MaxPooling2D, self).__init__(pool_size, strides, padding, block_id,
                                           "d3m.primitives.layer.max_pooling_2d.KerasWrap", previous_layer_id)


class MaxPooling3D(MaxPooling):
    def __init__(self, block_id, pool_size=(2, 2), strides=(1, 1), padding='valid', previous_layer_id=None):
        super(MaxPooling3D, self).__init__(pool_size, strides, padding, block_id,
                                           "d3m.primitives.layer.max_pooling_3d.KerasWrap", previous_layer_id)


class AvgPooling(Block):
    def __init__(self, pool_size, strides, padding, block_id, primitive, previous_layer_id):
        super(AvgPooling, self).__init__(block_id, primitive, previous_layer_id)
        self.pool_size = pool_size[0]
        self.strides = strides[0]
        self.padding = 'same' if padding else 'valid'

    def get_step(self):
        step = super().get_step()
        step.add_hyperparameter(name='pool_size', argument_type=ArgumentType.VALUE, data=self.pool_size)
        step.add_hyperparameter(name='strides', argument_type=ArgumentType.VALUE, data=self.strides)
        step.add_hyperparameter(name='padding', argument_type=ArgumentType.VALUE, data=self.padding)
        return step


class AvgPooling1D(AvgPooling):
    def __init__(self, block_id, pool_size=(2, 2), strides=(1, 1), padding='valid', previous_layer_id=None):
        super(AvgPooling1D, self).__init__(pool_size, strides, padding, block_id,
                                           "d3m.primitives.layer.average_pooling_1d.KerasWrap", previous_layer_id)


class AvgPooling2D(AvgPooling):
    def __init__(self, block_id, pool_size=(2, 2), strides=(1, 1), padding='valid', previous_layer_id=None):
        super(AvgPooling2D, self).__init__(pool_size, strides, padding, block_id,
                                           "d3m.primitives.layer.average_pooling_2d.KerasWrap", previous_layer_id)


class AvgPooling3D(AvgPooling):
    def __init__(self, block_id, pool_size=(2, 2), strides=(1, 1), padding='valid', previous_layer_id=None):
        super(AvgPooling3D, self).__init__(pool_size, strides, padding, block_id,
                                           "d3m.primitives.layer.average_pooling_3d.KerasWrap", previous_layer_id)


class GlobalAvgPooling2d(Block):
    def __init__(self, block_id, data_format='channels_last', previous_layer_id=None):
        super(GlobalAvgPooling2d, self).__init__(block_id, "d3m.primitives.layer.global_average_pooling_2d.KerasWrap",
                                                 previous_layer_id=previous_layer_id)
        self.data_format = data_format

    def get_step(self):
        step = super().get_step()
        step.add_hyperparameter(name='data_format', argument_type=ArgumentType.VALUE, data=self.data_format)
        return step


# JPL does not have such primitives,
# class GlobalMaxPooling2d(MaxPooling2D):
#     def __init__(self, block_id, input_shape, previous_layer_id):
#         kernel_size = input_shape[0]
#         super(GlobalMaxPooling2d, self).__init__(block_id, kernel_size, previous_layer_id=previous_layer_id)


class Dropout(Block):
    def __init__(self, block_id, rate=0.2, previous_layer_id=None):
        super(Dropout, self).__init__(block_id, "d3m.primitives.layer.dropout.KerasWrap", previous_layer_id)
        self.rate = rate

    def get_step(self):
        step = super().get_step()
        step.add_hyperparameter(name='rate', argument_type=ArgumentType.VALUE, data=self.rate)
        return step


class Flatten(Block):
    def __init__(self, block_id, previous_layer_id):
        super(Flatten, self).__init__(block_id, "d3m.primitives.layer.flatten.KerasWrap", previous_layer_id)


class Add(Block):
    def __init__(self, block_id, previous_layer_ids):
        super(Add, self).__init__(block_id, "d3m.primitives.layer.add.KerasWrap", None)
        self.previous_layer_ids = previous_layer_ids

    def get_step(self):
        step = PrimitiveStep(primitive=index.get_primitive(self.primitive))
        step.add_hyperparameter(name='previous_layers', argument_type=ArgumentType.PRIMITIVE,
                                data=self.previous_layer_ids)
        return step


class Concatenate(Block):
    def __init__(self, block_id, previous_layer_ids):
        super(Concatenate, self).__init__(block_id, "d3m.primitives.layer.concat.KerasWrap", None)
        self.previous_layer_ids = previous_layer_ids

    def get_step(self):
        step = PrimitiveStep(primitive=index.get_primitive(self.primitive))
        step.add_hyperparameter(name='previous_layers', argument_type=ArgumentType.PRIMITIVE,
                                data=self.previous_layer_ids)
        return step


class Null(Block):
    def __init__(self, block_id):
        super(Null, self).__init__(block_id, "d3m.primitives.layer.null.KerasWrap", None)
