import unittest

from d3m import container, utils
from d3m.metadata import base as metadata_base

from tods.timeseries_processing import SKQuantileTransformer
import numpy as np
import pandas as pd
from d3m.container import DataFrame as d3m_dataframe
from scipy.stats import kstest, shapiro
import os

this_path = os.path.dirname(os.path.realpath(__file__))

class SKQuantileTransformerTestCase(unittest.TestCase):
    def test_basic(self):
        self.maxDiff=None
        #dataset = pd.DataFrame([[0,2],[1,4],[2,6],[3,8],[4,10],[5,12],[6,14]])
        dataset_fname = os.path.join(this_path, '../../../datasets/anomaly/kpi/TRAIN/dataset_TRAIN/tables/learningData.csv')
        dataset = pd.read_csv(dataset_fname)
        # dataset = np.random.rand(1000)
        main = d3m_dataframe(dataset, generate_metadata=True)
        # print(main)

        hyperparams_class = SKQuantileTransformer.SKQuantileTransformerPrimitive.metadata.get_hyperparams()
        primitive = SKQuantileTransformer.SKQuantileTransformerPrimitive(hyperparams=hyperparams_class.defaults())
        primitive.set_training_data(inputs=main)
        primitive.fit()
        new_main = primitive.produce(inputs=main).value

        test_data = new_main.values[:, 1]
        # hist_data = new_main.values
        std_normal_samples = np.random.randn(test_data.__len__())

        # Plot the distribution
        # import matplotlib.pyplot as plt
        # plt.hist(test_data, bins=100, alpha=0.6)
        # plt.hist(std_normal_samples, bins=100, alpha=0.6)
        # plt.legend(labels=['QuantileTransformer', 'Standard Gaussian'], loc='best')
        # plt.savefig('./fig/test_SKQuantileTransformer.png')
        # plt.close()
        # plt.show()

        # centerization check
        new_mean, new_std = test_data.mean(), test_data.std()
        mean_mse = new_mean ** 2
        std_mse = (new_std-1) ** 2
        # print(mean_mse, std_mse)
        self.assertAlmostEqual(mean_mse.__float__(), 0., delta=1e-5)
        self.assertAlmostEqual(std_mse.__float__(), 0., delta=1e-5)

        self.assertEqual(utils.to_json_structure(new_main.metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                # 'top_level': 'main',
                'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
                'structural_type': 'd3m.container.pandas.DataFrame',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Table'],
                'dimension': {
                    'name': 'rows',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                    'length': 7027,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__'],
            'metadata': {
                'dimension': {
                    'name': 'columns',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                    'length': 4,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 0],
            'metadata': {
                'name': 'd3mIndex',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
                'structural_type': 'numpy.float64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {
                'name': 'timestamp',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
                'structural_type': 'numpy.float64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 2],
            'metadata': {
                'name': 'value',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
                'structural_type': 'numpy.float64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 3],
            'metadata': {
                'name': 'ground_truth',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
                'structural_type': 'numpy.float64',
            },
        }])

        params = primitive.get_params()
        primitive.set_params(params=params)


if __name__ == '__main__':
    unittest.main()
