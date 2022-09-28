import autokeras as ak
# from autokeras.engine.block import Block
import tensorflow as tf
import numpy as np
from numpy import percentile
# import pandas as pd
import pdb
from sklearn import metrics

from keras_tuner.engine import hyperparameters
from autokeras.engine import block as block_module
from typing import Optional
from typing import Union
from autokeras.utils import layer_utils
from autokeras.utils import utils
from autokeras.blocks import reduction

from tensorflow.python.util import nest
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from tensorflow.keras import layers
from tods.sk_interface.timeseries_processing.SubsequenceSegmentation_skinterface import SubsequenceSegmentationSKI


from tods.detection_algorithm.core.ak.blocks import AEBlock
from tods.detection_algorithm.core.ak.blocks import RNNBlock
from tods.detection_algorithm.core.ak.heads import ReconstructionHead
# load dataset yahoo
# dataset = pd.read_csv("./yahoo_sub_5.csv")
# data = dataset.to_numpy()
# label = dataset.iloc[:,6]

#another dataset
# data = np.loadtxt("./machine-1-2-test.txt", delimiter=',')
# label = np.loadtxt("./machine-1-2-label.txt")
# data = np.loadtxt("./omni/test/machine-1-1.txt", delimiter=',')
# label = np.loadtxt("./omni/label/machine-1-1.txt")

# data = np.loadtxt("./omni/test/machine-2-1.txt", delimiter=',')
# label = np.loadtxt("./omni/label/machine-2-1.txt")

# data = np.loadtxt("./omni/test/machine-3-1.txt", delimiter=',')
# label = np.loadtxt("./omni/label/machine-3-1.txt")

# data = np.loadtxt("./ucr/005_UCR_Anomaly_DISTORTEDCIMIS44AirTemperature1_4000_5391_5392.txt")
data = np.loadtxt("./ucr/012_UCR_Anomaly_DISTORTEDECG2_15000_16000_16100.txt")
# data = np.loadtxt("./ucr/019_UCR_Anomaly_DISTORTEDGP711MarkerLFM5z1_5000_6168_6212.txt")

data = np.expand_dims(data[:15000], axis=1)
X_test = np.expand_dims(data[15000:30000], axis=1)

# X_train = np.expand_dims(   #this is for convblock and rnn
#     X_train, axis=2
# )
# labels = np.expand_dims(data[5000:10000], axis=1)
label = np.zeros(15000,)
label[1000:1100] = 1
print(label)
# data = np.loadtxt("./omni/test/machine-1-1.txt", delimiter=',')
# label = np.loadtxt("./omni/label/machine-1-1.txt")


# data = np.loadtxt("./ucr/005_UCR_Anomaly_DISTORTEDCIMIS44AirTemperature1_4000_5391_5392.txt")
# # X_train = data[:4000]
# # X_test = data[4000:]

# X_train = np.expand_dims(data[:4000], axis=1)
# X_test = np.expand_dims(data[4000:8000], axis=1)

# print(data.shape)
# pdb.set_trace()
data_copy = data
# transformer = SubsequenceSegmentationSKI()
# tods_output = transformer.produce(data)
# print('result from SubsequenceSegmentation primitive:\n', tods_output)
# print('tods output shape:\n', tods_output.shape)

data = np.expand_dims(   #this is for convblock and rnn
    data, axis=2
) 
# print(data.shape)

# print()
class DenseBlock(block_module.Block): #AEBlock

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
            hyperparameters.Choice(
                "middle_unit", [4, 8, 16], default=4 
            ),
            int,
        )
        self.multiplier = utils.get_hyperparameter(
            multiplier,
            hyperparameters.Choice(
                "multiplier", [2, 3, 4], default=2 
            ),
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
        print('initial arch:', nueral_arch)

        for i in range(int((layer_range - 1) / 2)):
            num_u = multiplier**(i+1) * middle_unit
            nueral_arch.append(num_u)
            nueral_arch.insert(0, num_u)
        print('final arch:', nueral_arch)

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
            print('units:',units)
            output_node = layers.Dense(units)(output_node)
            if use_batchnorm:
                output_node = layers.BatchNormalization()(output_node)
            output_node = layers.ReLU()(output_node)
            if utils.add_to_hp(self.dropout, hp) > 0:
                output_node = layers.Dropout(utils.add_to_hp(self.dropout, hp))(
                    output_node
                )
        return output_node
class ConvBlock(block_module.Block):
    """Block for vanilla ConvNets.
    # Arguments
        kernel_size: Int or keras_tuner.engine.hyperparameters.Choice.
            The size of the kernel.
            If left unspecified, it will be tuned automatically.
        num_blocks: Int or keras_tuner.engine.hyperparameters.Choice.
            The number of conv blocks, each of which may contain
            convolutional, max pooling, dropout, and activation. If left unspecified,
            it will be tuned automatically.
        num_layers: Int or hyperparameters.Choice.
            The number of convolutional layers in each block. If left
            unspecified, it will be tuned automatically.
        filters: Int or keras_tuner.engine.hyperparameters.Choice. The number of
            filters in the convolutional layers. If left unspecified, it will
            be tuned automatically.
        max_pooling: Boolean. Whether to use max pooling layer in each block. If left
            unspecified, it will be tuned automatically.
        separable: Boolean. Whether to use separable conv layers.
            If left unspecified, it will be tuned automatically.
        dropout: Float. Between 0 and 1. The dropout rate for after the
            convolutional layers. If left unspecified, it will be tuned
            automatically.
    """

    def __init__(
        self,
        kernel_size: Optional[Union[int, hyperparameters.Choice]] = None,
        num_blocks: Optional[Union[int, hyperparameters.Choice]] = None,
        num_layers: Optional[Union[int, hyperparameters.Choice]] = None,
        filters: Optional[Union[int, hyperparameters.Choice]] = None,
        max_pooling: Optional[bool] = None,
        separable: Optional[bool] = None,
        dropout: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.kernel_size = utils.get_hyperparameter(
            kernel_size,
            hyperparameters.Choice("kernel_size", [3, 5, 7], default=3),
            int,
        )
        self.num_blocks = utils.get_hyperparameter(
            num_blocks,
            hyperparameters.Choice("num_blocks", [1, 2, 3], default=2),
            int,
        )
        self.num_layers = utils.get_hyperparameter(
            num_layers,
            hyperparameters.Choice("num_layers", [1, 2], default=2),
            int,
        )
        self.filters = utils.get_hyperparameter(
            filters,
            hyperparameters.Choice(
                "filters", [16, 32, 64, 128, 256, 512], default=32
            ),
            int,
        )
        self.max_pooling = max_pooling
        self.separable = separable
        self.dropout = dropout

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "kernel_size": hyperparameters.serialize(self.kernel_size),
                "num_blocks": hyperparameters.serialize(self.num_blocks),
                "num_layers": hyperparameters.serialize(self.num_layers),
                "filters": hyperparameters.serialize(self.filters),
                "max_pooling": self.max_pooling,
                "separable": self.separable,
                "dropout": self.dropout,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config["kernel_size"] = hyperparameters.deserialize(config["kernel_size"])
        config["num_blocks"] = hyperparameters.deserialize(config["num_blocks"])
        config["num_layers"] = hyperparameters.deserialize(config["num_layers"])
        config["filters"] = hyperparameters.deserialize(config["filters"])
        return cls(**config)

    def build(self, hp, inputs=None):
        inputs = nest.flatten(inputs)
        utils.validate_num_inputs(inputs, 1)
        input_node = inputs[0]
        output_node = input_node

        kernel_size = utils.add_to_hp(self.kernel_size, hp)

        separable = self.separable
        if separable is None:
            separable = hp.Boolean("separable", default=False)

        if separable:
            conv = layer_utils.get_sep_conv(input_node.shape)
        else:
            conv = layer_utils.get_conv(input_node.shape)

        max_pooling = self.max_pooling
        if max_pooling is None:
            max_pooling = hp.Boolean("max_pooling", default=True)
        pool = layer_utils.get_max_pooling(input_node.shape)

        if self.dropout is not None:
            dropout = self.dropout
        else:
            dropout = hp.Choice("dropout", [0.0, 0.25, 0.5], default=0)

        for i in range(utils.add_to_hp(self.num_blocks, hp)):
            for j in range(utils.add_to_hp(self.num_layers, hp)):
                output_node = conv(
                    utils.add_to_hp(
                        self.filters, hp, "filters_{i}_{j}".format(i=i, j=j)
                    ),
                    kernel_size,
                    padding=self._get_padding(kernel_size, output_node),
                    activation="relu",
                )(output_node)
            if max_pooling:
                output_node = pool(
                    kernel_size - 1,
                    padding=self._get_padding(kernel_size - 1, output_node),
                )(output_node)
            if dropout > 0:
                output_node = layers.Dropout(dropout)(output_node)
        return output_node

    @staticmethod
    def _get_padding(kernel_size, output_node):
        if all(kernel_size * 2 <= length for length in output_node.shape[1:-1]):
            return "valid"
        return "same"

class Embedding(block_module.Block):
    """Word embedding block for sequences.
    The input should be tokenized sequences with the same length, where each element
    of a sequence should be the index of the word.
    # Arguments
        max_features: Int. Size of the vocabulary. Must be set if not using
            TextToIntSequence before this block. Defaults to 20001.
        pretraining: String or keras_tuner.engine.hyperparameters.Choice.
            'random' (use random weights instead any pretrained
            model), 'glove', 'fasttext' or 'word2vec'. Use pretrained word embedding.
            If left unspecified, it will be tuned automatically.
        embedding_dim: Int or keras_tuner.engine.hyperparameters.Choice.
            Output dimension of the Attention block.
            If left unspecified, it will be tuned automatically.
        dropout: Float or keras_tuner.engine.hyperparameters.Choice.
            The dropout rate for the layers.
            If left unspecified, it will be tuned automatically.
    """

    def __init__(
        self,
        max_features: int = 20001,
        pretraining: Optional[Union[str, hyperparameters.Choice]] = None,
        embedding_dim: Optional[Union[int, hyperparameters.Choice]] = None,
        dropout: Optional[Union[float, hyperparameters.Choice]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_features = max_features
        self.pretraining = utils.get_hyperparameter(
            pretraining,
            hyperparameters.Choice(
                "pretraining",
                ["random", "glove", "fasttext", "word2vec", "none"],
                default="none",
            ),
            str,
        )
        self.embedding_dim = utils.get_hyperparameter(
            embedding_dim,
            hyperparameters.Choice(
                "embedding_dim", [32, 64, 128, 256, 512], default=128
            ),
            int,
        )
        self.dropout = utils.get_hyperparameter(
            dropout,
            hyperparameters.Choice("dropout", [0.0, 0.25, 0.5], default=0.25),
            float,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "max_features": self.max_features,
                "pretraining": hyperparameters.serialize(self.pretraining),
                "embedding_dim": hyperparameters.serialize(self.embedding_dim),
                "dropout": hyperparameters.serialize(self.dropout),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config["pretraining"] = hyperparameters.deserialize(config["pretraining"])
        config["dropout"] = hyperparameters.deserialize(config["dropout"])
        config["embedding_dim"] = hyperparameters.deserialize(
            config["embedding_dim"]
        )
        return cls(**config)

    def build(self, hp, inputs=None):
        input_node = nest.flatten(inputs)[0]
        # TODO: support more pretrained embedding layers.
        # glove, fasttext, and word2vec
        pretraining = utils.add_to_hp(self.pretraining, hp)
        embedding_dim = utils.add_to_hp(self.embedding_dim, hp)
        if pretraining != "none":
            # TODO: load from pretrained weights
            layer = layers.Embedding(
                input_dim=self.max_features,
                output_dim=embedding_dim,
                input_length=input_node.shape[1],
            )
            # trainable=False,
            # weights=[embedding_matrix])
        else:
            layer = layers.Embedding(
                input_dim=self.max_features, output_dim=embedding_dim
            )
            # input_length=input_node.shape[1],
            # trainable=True)
        output_node = layer(input_node)
        dropout = utils.add_to_hp(self.dropout, hp)
        if dropout > 0:
            output_node = layers.Dropout(dropout)(output_node)
        return output_node

class MultiHeadSelfAttention(block_module.Block):
    """Block for Multi-Head Self-Attention.
    # Arguments
        head_size: Int. Dimensionality of the `query`, `key` and `value` tensors
            after the linear transformation. If left unspecified, it will be
            tuned automatically.
        num_heads: Int. The number of attention heads. Defaults to 8.
    """

    def __init__(
        self, head_size: Optional[int] = None, num_heads: int = 8, **kwargs
    ):
        super().__init__(**kwargs)
        self.head_size = head_size
        self.num_heads = num_heads

    def get_config(self):
        config = super().get_config()
        config.update({"head_size": self.head_size, "num_heads": self.num_heads})
        return config

    def build(self, hp, inputs=None):
        """
        # Arguments
             hp: HyperParameters. The hyperparameters for building the model.
             inputs: Tensor of Shape [batch_size, seq_len, embedding_dim]
        # Returns
            Self-Attention outputs of shape `[batch_size, seq_len, embedding_dim]`.
        """
        inputs = nest.flatten(inputs)
        utils.validate_num_inputs(inputs, 1)
        input_node = inputs[0]
        num_heads = self.num_heads
        head_size = (
            self.head_size
            or hp.Choice("head_size_factor", [4, 8, 16, 32, 64], default=16)
            * num_heads
        )

        projection_dim = head_size // num_heads
        query_dense = layers.Dense(head_size)
        key_dense = layers.Dense(head_size)
        value_dense = layers.Dense(head_size)
        combine_heads = layers.Dense(head_size)
        batch_size = tf.shape(input_node)[0]
        query = query_dense(input_node)  # (batch_size, seq_len, head_size)
        key = key_dense(input_node)  # (batch_size, seq_len, head_size)
        value = value_dense(input_node)  # (batch_size, seq_len, head_size)
        query, key, value = [
            self.separate_heads(var, batch_size, num_heads, projection_dim)
            for var in [query, key, value]
        ]
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, tf.shape(attention)[1], self.head_size)
        )  # (batch_size, seq_len, head_size)
        return combine_heads(concat_attention)  # (batch_size, seq_len, head_size)

    @staticmethod
    def attention(query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    @staticmethod
    def separate_heads(x, batch_size, num_heads, projection_dim):
        x = tf.reshape(x, (batch_size, -1, num_heads, projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

class Transformer(block_module.Block):
    """Block for Transformer.
    The input should be tokenized sequences with the same length, where each element
    of a sequence should be the index of the word. The implementation is derived from
    the this
    [example](https://keras.io/examples/nlp/text_classification_with_transformer/).
    # Example
    ```python
        # Using the Transformer Block with AutoModel.
        import autokeras as ak
        from tensorflow.keras import losses
        text_input = ak.TextInput()
        output_node = ak.TextToIntSequence(output_sequence_length=200)(text_input)
        output_node = ak.Transformer(embedding_dim=32,
                             pretraining='none',
                             num_heads=2,
                             dense_dim=32,
                             dropout = 0.25)(output_node)
        output_node = ak.SpatialReduction(reduction_type='global_avg')(output_node)
        output_node = ak.DenseBlock(num_layers=1, use_batchnorm = False)(output_node)
        output_node = ak.ClassificationHead(
            loss=losses.SparseCategoricalCrossentropy(),
            dropout = 0.25)(output_node)
        clf = ak.AutoModel(inputs=text_input, outputs=output_node, max_trials=2)
    ```
    # Arguments
        max_features: Int. Size of the vocabulary. Must be set if not using
            TextToIntSequence before this block. Defaults to 20001.
        pretraining: String or keras_tuner.engine.hyperparameters.Choice.
            'random' (use random weights instead any pretrained
            model), 'glove', 'fasttext' or 'word2vec'. Use pretrained word embedding.
            If left unspecified, it will be tuned automatically.
        embedding_dim: Int or keras_tuner.engine.hyperparameters.Choice.
            Output dimension of the Attention block.
            If left unspecified, it will be tuned automatically.
        num_heads: Int or keras_tuner.engine.hyperparameters.Choice.
            The number of attention heads. If left unspecified,
            it will be tuned automatically.
        dense_dim: Int or keras_tuner.engine.hyperparameters.Choice.
            The output dimension of the Feed-Forward Network. If left
            unspecified, it will be tuned automatically.
        dropout: Float or keras_tuner.engine.hyperparameters.Choice.
            Between 0 and 1. If left unspecified, it will be
            tuned automatically.
    """

    def __init__(
        self,
        max_features: int = 20001,
        pretraining: Optional[Union[str, hyperparameters.Choice]] = None,
        embedding_dim: Optional[Union[int, hyperparameters.Choice]] = None,
        num_heads: Optional[Union[int, hyperparameters.Choice]] = None,
        dense_dim: Optional[Union[int, hyperparameters.Choice]] = None,
        dropout: Optional[Union[float, hyperparameters.Choice]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_features = max_features
        self.pretraining = utils.get_hyperparameter(
            pretraining,
            hyperparameters.Choice(
                "pretraining",
                ["random", "glove", "fasttext", "word2vec", "none"],
                default="none",
            ),
            str,
        )
        self.embedding_dim = utils.get_hyperparameter(
            embedding_dim,
            hyperparameters.Choice(
                "embedding_dim", [32, 64, 128, 256, 512], default=128
            ),
            int,
        )
        self.num_heads = utils.get_hyperparameter(
            num_heads,
            hyperparameters.Choice("num_heads", [8, 16, 32], default=8),
            int,
        )
        self.dense_dim = utils.get_hyperparameter(
            dense_dim,
            hyperparameters.Choice(
                "dense_dim", [128, 256, 512, 1024, 2048], default=2048
            ),
            int,
        )
        self.dropout = utils.get_hyperparameter(
            dropout,
            hyperparameters.Choice("dropout", [0.0, 0.25, 0.5], default=0.0),
            float,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "max_features": self.max_features,
                "pretraining": hyperparameters.serialize(self.pretraining),
                "embedding_dim": hyperparameters.serialize(self.embedding_dim),
                "num_heads": hyperparameters.serialize(self.num_heads),
                "dense_dim": hyperparameters.serialize(self.dense_dim),
                "dropout": hyperparameters.serialize(self.dropout),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config["pretraining"] = hyperparameters.deserialize(config["pretraining"])
        config["embedding_dim"] = hyperparameters.deserialize(
            config["embedding_dim"]
        )
        config["num_heads"] = hyperparameters.deserialize(config["num_heads"])
        config["dense_dim"] = hyperparameters.deserialize(config["dense_dim"])
        config["dropout"] = hyperparameters.deserialize(config["dropout"])
        return cls(**config)

    def build(self, hp, inputs=None):
        """
        # Arguments
             hp: HyperParameters. The hyperparameters for building the model.
             inputs: Tensor of Shape [batch_size, seq_len]
        # Returns
            Output Tensor of shape `[batch_size, seq_len, embedding_dim]`.
        """
        inputs = nest.flatten(inputs)
        utils.validate_num_inputs(inputs, 1)
        pretraining = utils.add_to_hp(self.pretraining, hp)
        embedding_dim = utils.add_to_hp(self.embedding_dim, hp)
        num_heads = utils.add_to_hp(self.num_heads, hp)

        dense_dim = utils.add_to_hp(self.dense_dim, hp)
        dropout = utils.add_to_hp(self.dropout, hp)

        ffn = tf.keras.Sequential(
            [
                layers.Dense(dense_dim, activation="relu"),
                layers.Dense(embedding_dim),
            ]
        )

        layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        dropout1 = layers.Dropout(dropout)
        dropout2 = layers.Dropout(dropout)
        # Token and Position Embeddings
        input_node = nest.flatten(inputs)[0]
        token_embedding = Embedding(
            max_features=self.max_features,
            pretraining=pretraining,
            embedding_dim=embedding_dim,
            dropout=dropout,
        ).build(hp, input_node)
        maxlen = input_node.shape[-1]
        batch_size = tf.shape(input_node)[0]
        positions = self.pos_array_funct(maxlen, batch_size)
        position_embedding = Embedding(
            max_features=maxlen,
            pretraining=pretraining,
            embedding_dim=embedding_dim,
            dropout=dropout,
        ).build(hp, positions)
        output_node = tf.keras.layers.Add()([token_embedding, position_embedding])
        attn_output = MultiHeadSelfAttention(embedding_dim, num_heads).build(
            hp, output_node
        )
        attn_output = dropout1(attn_output)
        add_inputs_1 = tf.keras.layers.Add()([output_node, attn_output])
        out1 = layernorm1(add_inputs_1)
        ffn_output = ffn(out1)
        ffn_output = dropout2(ffn_output)
        add_inputs_2 = tf.keras.layers.Add()([out1, ffn_output])
        return layernorm2(add_inputs_2)

    @staticmethod
    def pos_array_funct(maxlen, batch_size):
        pos_ones = tf.ones((batch_size, 1), dtype=tf.int32)
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = tf.expand_dims(positions, 0)
        positions = tf.matmul(pos_ones, positions)
        return positions


class RNNBlock_ak(block_module.Block):
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
                "bidirectional": hyperparameters.serialize(self.bidirectional),
                "num_layers": hyperparameters.serialize(self.num_layers),
                "layer_type": hyperparameters.serialize(self.layer_type),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config["bidirectional"] = hyperparameters.deserialize(
            config["bidirectional"]
        )
        config["num_layers"] = hyperparameters.deserialize(config["num_layers"])
        config["layer_type"] = hyperparameters.deserialize(config["layer_type"])
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
        return output_node
# RNN uses 4, best .48 ish
# conv use 3 or 5
#1. RNN and Conv quits, is this bc my computer? or need a new dataset? or expand dim?
#2. should i try to use server, or my desktop? is it RAM or what, if RAM, how big?
#3. for conv, I think current is Conv 3D, is that what we need, if not, which class would it be from ak basic.py
#4. might need to schedule more individual meetings with henry
#5. how to work on the new demo, what do i need to do?


# inputs = ak.Input(shape=[38,]) #important!!! depends on data shape above
# inputs = ak.Input(shape=[38,], batch_size = 32, )
inputs = ak.Input(shape=[38,])

#below is testing for wrapping into tods
# mlp_output = AEBlock()([inputs])

# mlp_output = DenseBlock()([inputs])
mlp_output = RNNBlock()([inputs]) #RNN datalab4

# mlp_output = ConvBlock()([inputs]) #CNN datalab5

# mlp_output = DenseBlock()([mlp_input])

# Step 2.3: Setup optimizer to handle the target task
#below is testing for wrapping into tods
output = ReconstructionHead()(mlp_output) 
# output = ak.RegressionHead()(mlp_output) 
# print('output:', output[0].__dict__)

# Step 3: Build the searcher, which provides search algorithm
auto_model = ak.AutoModel(inputs=inputs,
                          outputs=output, #final mlp out
                          objective='val_mean_squared_error',
                          max_trials=1 #10
                          )

# Step 4: Use the searcher to search the recommender
auto_model.fit(x=[data], #0 - n-1 tods output
               y=data,  # 1 - n tods output, or data
               batch_size=32,#128
            #    time_steps = 10,
               epochs=1)#20

pred = auto_model.predict(x=[data])
# label = auto_model.predict(x=[X_test])
# pred_prob = auto_model.export_model()
# print('probbbbbbb',pred_prob)

# prob = pred_prob.predict(data)
# print('probability', prob)

print(pred.shape)
print('pred:', pred)
# data = np.squeeze(data, axis=1)

y_true = label
y_pred = pred

# Using 'auto'/'sum_over_batch_size' reduction type.
mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
decision_scores_ = mse(y_true, y_pred).numpy()


threshold_ = percentile(decision_scores_, 100 * (1 - 0.1)) #0.01 = contamination
y_pred = (decision_scores_ > threshold_).astype('int').ravel()
# y_true = (decision_scores_ > threshold_).astype('int').ravel()

print('confusion matrix:\n', confusion_matrix(y_true, y_pred))
print('classification report:\n', classification_report(y_true, y_pred))

fpr, tpr, threshold = metrics.roc_curve(y_true, y_pred)
roc_auc = metrics.auc(fpr, tpr)

print('AUC_score: \n', roc_auc)