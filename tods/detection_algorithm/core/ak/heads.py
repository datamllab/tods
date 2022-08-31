# Copyright 2020 The AutoKeras Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional




import tensorflow as tf
from tensorflow.keras import activations
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.python.util import nest

from autokeras import adapters
from autokeras import analysers
from autokeras import hyper_preprocessors as hpps_module
from autokeras import preprocessors
from autokeras.blocks import reduction
from autokeras.engine import head as head_module
from autokeras.utils import types
from autokeras.utils import utils

class ReconstructionHead(head_module.Head):
    """Regression Dense layers.
    The targets passing to the head would have to be tf.data.Dataset, np.ndarray,
    pd.DataFrame or pd.Series. It can be single-column or multi-column. The
    values should all be numerical.
    # Arguments
        output_dim: Int. The number of output dimensions. Defaults to None.
            If None, it will be inferred from the data.
        multi_label: Boolean. Defaults to False.
        loss: A Keras loss function. Defaults to use `mean_squared_error`.
        metrics: A list of Keras metrics. Defaults to use `mean_squared_error`.
        dropout: Float. The dropout rate for the layers.
            If left unspecified, it will be tuned automatically.
    """

    def __init__(
        self,
        output_dim: Optional[int] = None,
        loss: types.LossType = "mean_squared_error",
        metrics: Optional[types.MetricsType] = None,
        dropout: Optional[float] = None,
        **kwargs
    ):
        # print('hi..')
        # input()
        if metrics is None:
            metrics = ["mean_squared_error"]
        super().__init__(loss=loss, metrics=metrics, **kwargs)
        self.output_dim = output_dim
        self.dropout = dropout

    def get_config(self):
        config = super().get_config()
        config.update({"output_dim": self.output_dim, "dropout": self.dropout})
        return config

    def build(self, hp, inputs=None):
        inputs = nest.flatten(inputs)
        utils.validate_num_inputs(inputs, 1)
        # print('input:',inputs)
        # input()
        input_node = inputs[0]
        output_node = input_node

        # if self.dropout is not None:
        #     dropout = self.dropout
        # else:
        #     dropout = hp.Choice("dropout", [0.0, 0.25, 0.5], default=0)

        # if dropout > 0:
        #     output_node = layers.Dropout(dropout)(output_node)
        output_node = reduction.Flatten().build(hp, output_node)
        output_node = layers.Dense(self.shape[-1], name=self.name)(output_node)
        return output_node

    def config_from_analyser(self, analyser):
        super().config_from_analyser(analyser)
        self._add_one_dimension = len(analyser.shape) == 1

    def get_adapter(self):
        return adapters.RegressionAdapter(name=self.name) 

    def get_analyser(self):
        return analysers.RegressionAnalyser(
            name=self.name, output_dim=self.output_dim
        )

    def get_hyper_preprocessors(self):
        hyper_preprocessors = []
        if self._add_one_dimension:
            hyper_preprocessors.append(
                hpps_module.DefaultHyperPreprocessor(preprocessors.AddOneDimension())
            )
        return hyper_preprocessors
