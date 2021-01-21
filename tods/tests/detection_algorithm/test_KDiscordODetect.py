import unittest

from d3m import container, utils
from d3m.metadata import base as metadata_base
from d3m.container import DataFrame as d3m_dataframe

from tods.detection_algorithm.KDiscordODetect import KDiscordODetectorPrimitive
from pyod.utils.data import generate_data

from tods.detection_algorithm.core.CollectiveCommonTest import CollectiveCommonTest

import numpy as np

class KDiscordODetectTestCase(unittest.TestCase):
    def setUp(self):

        self.maxDiff = None
        self.contamination = 0.1
        self.window_size = 2
        self.roc_floor = 0.  # 0.8
        # self.n_train = 200
        # self.n_test = 100
        # self.X_train, self.y_train, self.X_test, self.y_test = generate_data(
        #     n_train=self.n_train, n_test=self.n_test,
        #     contamination=self.contamination, random_state=42)
        # self.X_train = d3m_dataframe(self.X_train, generate_metadata=True)
        # self.X_test = d3m_dataframe(self.X_test, generate_metadata=True)

        self.X_train = d3m_dataframe({'data': [3., 4., 8., 16, 18, 13., 22., 36., 59., 128, 62, 67, 78, 100]},
                                      columns=['data'], generate_metadata=True)
        self.y_train = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.X_test = d3m_dataframe({'data': [3., 4., 8.6, 13.4, 22.5, 17, 19.2, 36.1, 127, -23, 59.2]},
                                     columns=['data'], generate_metadata=True)
        self.y_test = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])



        hyperparams_default = KDiscordODetectorPrimitive.metadata.get_hyperparams().defaults()
        hyperparams = hyperparams_default.replace({'contamination': self.contamination, })
        hyperparams = hyperparams.replace({'window_size': self.window_size, })
        hyperparams = hyperparams.replace({'return_subseq_inds': True, })
        self.primitive = KDiscordODetectorPrimitive(hyperparams=hyperparams)

        self.primitive.set_training_data(inputs=self.X_train)
        self.primitive.fit()
        self.prediction_labels = self.primitive.produce(inputs=self.X_test).value
        self.prediction_score = self.primitive.produce_score(inputs=self.X_test).value

        self.collective_common_test = CollectiveCommonTest(model=self.primitive._clf,
                                                            X_train=self.X_train,
                                                            y_train=self.y_train,
                                                            X_test=self.X_test,
                                                            y_test=self.y_test,
                                                            roc_floor=self.roc_floor,
                                                            )

    def test_detector(self):
        self.collective_common_test.test_detector()
        pass

    def test_metadata(self):
        # print(self.prediction_labels.metadata.to_internal_simple_structure())
        self.assertEqual(utils.to_json_structure(self.prediction_labels.metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                # 'top_level': 'main',
                'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
                'structural_type': 'd3m.container.pandas.DataFrame',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Table'],
                'dimension': {
                    'name': 'rows',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                    'length': 10,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__'],
            'metadata': {
                'dimension': {
                    'name': 'columns',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                    'length': 3,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 0],
            'metadata': {
                'name': 'KDiscordODetector0_0',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {
                'name': 'KDiscordODetector0_1',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
                'structural_type': 'numpy.int64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 2],
            'metadata': {
                'name': 'KDiscordODetector0_2',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
                'structural_type': 'numpy.int64',
            },
        }])

    def test_params(self):
        params = self.primitive.get_params()
        self.primitive.set_params(params=params)


if __name__ == '__main__':
    unittest.main()
