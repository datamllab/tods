from autokeras.engine import block as block_module
from autokeras.utils import utils
from autokeras.blocks import reduction 

from keras_tuner.engine import hyperparameters
from tensorflow.python.util import nest
from tensorflow.keras import layers

from typing import Optional
from typing import Union

from autokeras import adapters
from autokeras import analysers
from autokeras import hyper_preprocessors as hpps_module
from autokeras.utils import io_utils
from autokeras import preprocessors
# from autokeras.blocks import reduction
from autokeras.engine import head as head_module
from autokeras.utils import types
# from autokeras.utils import utils
        
class AEBlock(block_module.Block): #AEBlock

    """Block for Dense layers.
    # Arguments
        num_layers: Int or keras_tuner.engine.hyperparameters.Choice.
            The number of Dense layers in the block.
            If left unspecified, it will be tuned automatically.
        num_units: Int or keras_tuner.engine.hyperparameters.Choice.
            The number of units in each dense layer.
            If left unspecified, it will be tuned automatically.
        use_bn: Boolean. Whether to use BatchNormalization layers.
            If left unspecified, it will be tuned automatically.
        dropout: Float or keras_tuner.engine.hyperparameters.Choice.
            The dropout rate for the layers.
            If left unspecified, it will be tuned automatically.
    """

    def __init__(
        self,
        num_layers: Optional[Union[int, hyperparameters.Choice]] = None, 
        middle_unit: Optional[Union[int, hyperparameters.Choice]] = None,
        multiplier: Optional[Union[int, hyperparameters.Choice]] = None, 
        use_batchnorm: Optional[bool] = None,
        dropout: Optional[Union[float, hyperparameters.Choice]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_layers = utils.get_hyperparameter(
            num_layers,
            hyperparameters.Choice("num_layers", [3, 5, 7], default=5),
            int,
        )
        self.middle_unit = utils.get_hyperparameter(
            middle_unit,
            hyperparameters.Choice("middle_unit", [4, 8, 16], default=4),
            int,
        )
        self.multiplier = utils.get_hyperparameter(
            multiplier,
            hyperparameters.Choice("multiplier", [2, 3, 4], default=2),
            int,
        )
        self.use_batchnorm = use_batchnorm
        self.dropout = utils.get_hyperparameter(
            dropout,
            hyperparameters.Choice("dropout", [0.0, 0.25, 0.5], default=0.0),
            float,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_layers": hyperparameters.serialize(self.num_layers),
                "middle_unit": hyperparameters.serialize(self.middle_unit),
                "multiplier": hyperparameters.serialize(self.multiplier),
                "use_batchnorm": self.use_batchnorm,
                "dropout": hyperparameters.serialize(self.dropout),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config["num_layers"] = hyperparameters.deserialize(config["num_layers"])
        config["middle_unit"] = hyperparameters.deserialize(config["middle_unit"])
        config["multiplier"] = hyperparameters.deserialize(config["multiplier"])
        config["dropout"] = hyperparameters.deserialize(config["dropout"])
        return cls(**config)

    def architecture(self, layer_range, multiplier, middle_unit):
        # middle_unit = self.middle_unit.random_sample()
        # multiplier = self.multiplier.random_sample()
        nueral_arch = [middle_unit]
        # print('initial arch:', nueral_arch)

        for i in range(int((layer_range - 1) / 2)):
            num_u = multiplier**(i+1) * middle_unit
            nueral_arch.append(num_u)
            nueral_arch.insert(0, num_u)
        # print('final arch:', nueral_arch)

        return nueral_arch

    def build(self, hp, inputs=None):
        inputs = nest.flatten(inputs)
        utils.validate_num_inputs(inputs, 1)
        input_node = inputs[0]
        output_node = input_node
        output_node = reduction.Flatten().build(hp, output_node)

        use_batchnorm = self.use_batchnorm
        if use_batchnorm is None:
            use_batchnorm = hp.Boolean("use_batchnorm", default=False)

        num_layers = utils.add_to_hp(self.num_layers, hp)
        multiplier = utils.add_to_hp(self.multiplier, hp)
        middle_unit = utils.add_to_hp(self.middle_unit, hp)


        arch = self.architecture(num_layers, multiplier, middle_unit)
    
        for i in range(num_layers):
            units = utils.add_to_hp(arch[i], hp)
            # print('units:',units)
            output_node = layers.Dense(units)(output_node)
            if use_batchnorm:
                output_node = layers.BatchNormalization()(output_node)
            # else:

            output_node = layers.ReLU()(output_node)
            if utils.add_to_hp(self.dropout, hp) > 0:
                output_node = layers.Dropout(utils.add_to_hp(self.dropout, hp))(
                    output_node
                )
        return output_node

class RNNBlock(block_module.Block):
    """An RNN Block.
    # Arguments
        return_sequences: Boolean. Whether to return the last output in the
            output sequence, or the full sequence. Defaults to False.
        bidirectional: Boolean or keras_tuner.engine.hyperparameters.Boolean.
            Bidirectional RNN. If left unspecified, it will be
            tuned automatically.
        num_layers: Int or keras_tuner.engine.hyperparameters.Choice.
            The number of layers in RNN. If left unspecified, it will
            be tuned automatically.
        layer_type: String or or keras_tuner.engine.hyperparameters.Choice.
            'gru' or 'lstm'. If left unspecified, it will be tuned
            automatically.
    """

    def __init__(
        self,
        return_sequences: bool = False,
        bidirectional: Optional[Union[bool, hyperparameters.Boolean]] = None,
        num_layers: Optional[Union[int, hyperparameters.Choice]] = None,
        layer_type: Optional[Union[str, hyperparameters.Choice]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.return_sequences = return_sequences
        self.bidirectional = utils.get_hyperparameter(
            bidirectional,
            hyperparameters.Boolean("bidirectional", default=True),
            bool,
        )
        self.num_layers = utils.get_hyperparameter(
            num_layers,
            hyperparameters.Choice("num_layers", [1, 2, 3], default=2),
            int,
        )
        self.layer_type = utils.get_hyperparameter(
            layer_type,
            hyperparameters.Choice("layer_type", ["gru", "lstm"], default="lstm"),
            str,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "return_sequences": self.return_sequences,
                "bidirectional": io_utils.serialize_block_arg(self.bidirectional),
                "num_layers": io_utils.serialize_block_arg(self.num_layers),
                "layer_type": io_utils.serialize_block_arg(self.layer_type),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config["bidirectional"] = io_utils.deserialize_block_arg(
            config["bidirectional"]
        )
        config["num_layers"] = io_utils.deserialize_block_arg(config["num_layers"])
        config["layer_type"] = io_utils.deserialize_block_arg(config["layer_type"])
        return cls(**config)

    def build(self, hp, inputs=None):
        inputs = nest.flatten(inputs)
        utils.validate_num_inputs(inputs, 1)
        input_node = inputs[0]
        shape = input_node.shape.as_list()
        if len(shape) != 3:
            raise ValueError(
                "Expect the input tensor of RNNBlock to have dimensions of "
                "[batch_size, time_steps, vec_len], "
                "but got {shape}".format(shape=input_node.shape)
            )

        feature_size = shape[-1]
        output_node = input_node

        bidirectional = utils.add_to_hp(self.bidirectional, hp)
        layer_type = utils.add_to_hp(self.layer_type, hp)
        num_layers = utils.add_to_hp(self.num_layers, hp)
        rnn_layers = {"gru": layers.GRU, "lstm": layers.LSTM}
        in_layer = rnn_layers[layer_type]
        for i in range(num_layers):
            return_sequences = True
            if i == num_layers - 1:
                return_sequences = self.return_sequences
            if bidirectional:
                output_node = layers.Bidirectional(
                    in_layer(feature_size, return_sequences=return_sequences)
                )(output_node)
            else:
                output_node = in_layer(
                    feature_size, return_sequences=return_sequences
                )(output_node)
                output_node = layers.Dropout(utils.add_to_hp(self.dropout, hp))(
                    output_node)
        return output_node

# class RNNBlock(block_module.Block):
#     """An RNN Block.
#     # Arguments
#         return_sequences: Boolean. Whether to return the last output in the
#             output sequence, or the full sequence. Defaults to False.
#         bidirectional: Boolean or keras_tuner.engine.hyperparameters.Boolean.
#             Bidirectional RNN. If left unspecified, it will be
#             tuned automatically.
#         num_layers: Int or keras_tuner.engine.hyperparameters.Choice.
#             The number of layers in RNN. If left unspecified, it will
#             be tuned automatically.
#         layer_type: String or or keras_tuner.engine.hyperparameters.Choice.
#             'gru' or 'lstm'. If left unspecified, it will be tuned
#             automatically.
#     """

#     def __init__(
#         self,
#         return_sequences: bool = False,
#         bidirectional: Optional[Union[bool, hyperparameters.Boolean]] = None,
#         num_layers: Optional[Union[int, hyperparameters.Choice]] = None,
#         middle_unit: Optional[Union[int, hyperparameters.Choice]] = None,
#         multiplier: Optional[Union[int, hyperparameters.Choice]] = None, 
#         layer_type: Optional[Union[str, hyperparameters.Choice]] = None,
#         **kwargs,
#     ):
#         super().__init__(**kwargs)
#         self.return_sequences = return_sequences
#         self.bidirectional = utils.get_hyperparameter(
#             bidirectional,
#             hyperparameters.Boolean("bidirectional", default=True),
#             bool,
#         )
#         self.num_layers = utils.get_hyperparameter(
#             num_layers,
#             hyperparameters.Choice("num_layers", [3, 5, 7], default=3),
#             int,
#         )
#         self.middle_unit = utils.get_hyperparameter(
#             middle_unit,
#             hyperparameters.Choice("middle_unit", [4, 8, 16], default=4),
#             int,
#         )
#         self.multiplier = utils.get_hyperparameter(
#             multiplier,
#             hyperparameters.Choice("multiplier", [2, 3, 4], default=2),
#             int,
#         )
#         self.layer_type = utils.get_hyperparameter(
#             layer_type,
#             hyperparameters.Choice("layer_type", ["gru", "lstm"], default="lstm"),
#             str,
#         )

#     def get_config(self):
#         config = super().get_config()
#         config.update(
#             {
#                 "return_sequences": self.return_sequences,
#                 "bidirectional": io_utils.serialize_block_arg(self.bidirectional),
#                 "num_layers": io_utils.serialize_block_arg(self.num_layers),
#                 "middle_unit": hyperparameters.serialize(self.middle_unit),
#                 "multiplier": hyperparameters.serialize(self.multiplier),
#                 "layer_type": io_utils.serialize_block_arg(self.layer_type),
#             }
#         )
#         return config

#     @classmethod
#     def from_config(cls, config):
#         config["bidirectional"] = io_utils.deserialize_block_arg(
#             config["bidirectional"]
#         )
#         config["num_layers"] = io_utils.deserialize_block_arg(config["num_layers"])
#         config["layer_type"] = io_utils.deserialize_block_arg(config["layer_type"])
#         return cls(**config)

#     def architecture(self, layer_range, multiplier, middle_unit):
#         # middle_unit = self.middle_unit.random_sample()
#         # multiplier = self.multiplier.random_sample()
#         nueral_arch = [middle_unit]
#         # print('initial arch:', nueral_arch)

#         for i in range(int((layer_range - 1) / 2)):
#             num_u = multiplier**(i+1) * middle_unit
#             nueral_arch.append(num_u)
#             nueral_arch.insert(0, num_u)
#         print('final arch:', nueral_arch)

#         return nueral_arch

#     def build(self, hp, inputs=None):
#         inputs = nest.flatten(inputs)
#         utils.validate_num_inputs(inputs, 1)
#         input_node = inputs[0]
#         shape = input_node.shape.as_list()
#         if len(shape) != 3:
#             raise ValueError(
#                 "Expect the input tensor of RNNBlock to have dimensions of "
#                 "[batch_size, time_steps, vec_len], "
#                 "but got {shape}".format(shape=input_node.shape)
#             )

#         inputs = nest.flatten(inputs)
#         utils.validate_num_inputs(inputs, 1)
#         input_node = inputs[0]
#         output_node = input_node
#         output_node = reduction.Flatten().build(hp, output_node)


#         feature_size = shape[-1]
#         print('featuresize-------------------:', feature_size)
#         output_node = input_node

#         bidirectional = utils.add_to_hp(self.bidirectional, hp)
#         layer_type = utils.add_to_hp(self.layer_type, hp)
#         num_layers = utils.add_to_hp(self.num_layers, hp)
#         multiplier = utils.add_to_hp(self.multiplier, hp)
#         middle_unit = utils.add_to_hp(self.middle_unit, hp)
#         arch = self.architecture(num_layers, multiplier, middle_unit)
#         print('arch-----------------:', arch)
#         rnn_layers = {"gru": layers.GRU, "lstm": layers.LSTM}
#         in_layer = rnn_layers[layer_type]
#         for i in range(num_layers):
#             units = utils.add_to_hp(arch[i], hp)
#             return_sequences = True
#             if i == num_layers - 1:
#                 return_sequences = self.return_sequences
#             if bidirectional:
#                 print('current number of units in this bidirectional layer', units)
#                 output_node = layers.Bidirectional(
#                     in_layer(units, return_sequences=return_sequences)
#                 )(output_node)
#             else:
#                 # output_node = layers.Dense(units)(output_node) #???
#                 print('current number of units in this layer', units)
#                 output_node = in_layer(
#                     units, return_sequences=return_sequences
#                 )(output_node)
#         return output_node