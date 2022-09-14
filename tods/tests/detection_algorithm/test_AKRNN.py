import unittest

from d3m import container, utils
from d3m.metadata import base as metadata_base
from d3m.container import DataFrame as d3m_dataframe
from d3m.container import ndarray as d3m_numpy
from tods.detection_algorithm.AKRNN import AKRNNPrimitive
# from tods.detection_algorithm.PyodAE import AutoEncoderPrimitive
from pyod.utils.data import generate_data

from tods.detection_algorithm.core.UODCommonTest import UODCommonTest

import numpy as np

class AKRNNCase(unittest.TestCase):
    def setUp(self):

        self.maxDiff = None
        self.n_train = 200
        self.n_test = 100
        # self.contamination = 0.1
        self.roc_floor = 0.8
        self.X_train, self.y_train, self.X_test, self.y_test = generate_data(
            n_train=self.n_train, n_test=self.n_test)
            #FIXME shape of x train and x test shape not equal, for example x train (200,2) while x test will be (200,)

        print("shapes:", self.X_train.shape, self.X_test.shape)

        self.X_train = d3m_dataframe(self.X_train, generate_metadata=True)
        self.X_test = d3m_dataframe(self.X_test, generate_metadata=True)

        hyperparams_default = AKRNNPrimitive.metadata.get_hyperparams().defaults()
        hyperparams = hyperparams_default.defaults()
        hyperparams = hyperparams.replace({'epochs': 4})

        self.primitive = AKRNNPrimitive(hyperparams=hyperparams)
        print('to numpy after result:',self.X_train)
        print('type afterwards:',type(self.X_train))
        # print('xtrain shape:',self.X_train.shape)
        print('xtrain:',type(self.X_train))
        self.primitive.set_training_data(inputs=self.X_train)
        self.primitive.fit()
        self.prediction_labels = self.primitive.produce(inputs=self.X_test).value
        self.prediction_score = self.primitive.produce_score(inputs=self.X_test).value

        self.uodbase_test = UODCommonTest(model=self.primitive._clf,
                                          X_train=self.X_train,
                                          y_train=self.y_train,
                                          X_test=self.X_test,
                                          y_test=self.y_test,
                                          roc_floor=self.roc_floor,
                                          )

    def test_detector(self):
        self.uodbase_test.test_detector()

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
                    'length': 100,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__'],
            'metadata': {
                'dimension': {
                    'name': 'columns',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                    'length': 1,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 0],
            'metadata': {
                'name': 'AutoKeras Auto Encoder Primitive0_0',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
                'structural_type': 'numpy.int64',
            },
        }])

    def test_params(self):
        params = self.primitive.get_params()
        self.primitive.set_params(params=params)


if __name__ == '__main__':
    unittest.main()
    # suite = unittest.TestSuite()
    # for test_case in (
    #     'test_metadata',
    # ):
    #     suite.addTest(AKRNNCase(test_case))
    # unittest.TextTestRunner(verbosity=2).run(suite)
