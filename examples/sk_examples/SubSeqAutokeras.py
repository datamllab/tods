import autokeras as ak
from autokeras.engine.block import Block
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.python.util import nest
from tods.sk_interface.timeseries_processing.SubsequenceSegmentation_skinterface import SubsequenceSegmentationSKI
# load dataset
dataset = pd.read_csv("./yahoo_sub_5.csv")
data = dataset.to_numpy()
labels = dataset.iloc[:,6]
print(labels)
transformer = SubsequenceSegmentationSKI()
tods_output = transformer.produce(data)
print('result from SubsequenceSegmentation primitive:', tods_output)
print(tods_output.shape)

#autoregression 

class MLPInteraction(Block):
    """Module for MLP operation. This block can be configured with different layer, unit, and other settings.
    # Attributes:
        units (int). The units of all layer in the MLP block.
        num_layers (int). The number of the layers in the MLP block.
        use_batchnorm (Boolean). Use batch normalization or not.
        dropout_rate(float). The value of drop out in the last layer of MLP.
    """

    def __init__(self,
                 units=None,
                 num_layers=None,
                 use_batchnorm=None,
                 dropout_rate=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.num_layers = num_layers
        self.use_batchnorm = use_batchnorm
        self.dropout_rate = dropout_rate

    def get_state(self):
        state = super().get_state()
        state.update({
            'units': self.units,
            'num_layers': self.num_layers,
            'use_batchnorm': self.use_batchnorm,
            'dropout_rate': self.dropout_rate})
        return state

    def set_state(self, state):
        super().set_state(state)
        self.units = state['units']
        self.num_layers = state['num_layers']
        self.use_batchnorm = state['use_batchnorm']
        self.dropout_rate = state['dropout_rate']

    def build(self, hp, inputs=None):
        input_node = [tf.keras.layers.Flatten()(node) if len(node.shape) > 2 else node for node in nest.flatten(inputs)]
        output_node = tf.concat(input_node, axis=1)
        num_layers = self.num_layers or hp.Choice('num_layers', [1, 2, 3], default=2)
        use_batchnorm = self.use_batchnorm
        if use_batchnorm is None:
            use_batchnorm = hp.Choice('use_batchnorm', [True, False], default=False)
        dropout_rate = self.dropout_rate or hp.Choice('dropout_rate',
                                                      [0.0, 0.25, 0.5],
                                                      default=0)

        for i in range(num_layers):
            units = self.units or hp.Choice(
                'units_{i}'.format(i=i),
                [16, 32, 64, 128, 256, 512, 1024],
                default=32)
            output_node = tf.keras.layers.Dense(units)(output_node)
            if use_batchnorm:
                output_node = tf.keras.layers.BatchNormalization()(output_node)
            output_node = tf.keras.layers.ReLU()(output_node)
            output_node = tf.keras.layers.Dropout(dropout_rate)(output_node)
        return output_node
inputs = ak.Input(shape=[7,]) #important!!! depends on shape above
print(inputs.shape)
print(inputs.dtype)
mlp_input = MLPInteraction()([inputs])
mlp_output = MLPInteraction()([mlp_input])

# Step 2.3: Setup optimizer to handle the target task
output = ak.RegressionHead()(mlp_output)

# Step 3: Build the searcher, which provides search algorithm
auto_model = ak.AutoModel(inputs=inputs,#produce
                          outputs=output, #final mlp out
                          objective='val_mean_squared_error',
                          max_trials=5
                          )
# Step 4: Use the searcher to search the recommender
auto_model.fit(x=[tods_output],
               y=tods_output, #make new colume of labels of yahoo dataset # first element of next part
               batch_size=32,
               epochs=5)

accuracy = auto_model.evaluate(x=[tods_output],
               y=labels)

print(accuracy)
# logger.info('Validation Accuracy (mse): {}'.format(auto_model.evaluate(x=[val_X_categorical],
#                                                                        y=val_y)))
# # Step 5: Evaluate the searched model
# logger.info('Test Accuracy (mse): {}'.format(auto_model.evaluate(x=[tods_output],
#                                                                  y=labels)))