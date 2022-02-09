import unittest

from d3m import container, utils
from d3m.metadata import base as metadata_base
from tods.detection_algorithm.DeepLog import DeepLogPrimitive



class DeepLogTest(unittest.TestCase):
    def test_basic(self):
        self.maxDiff = None
        self.main = container.DataFrame({'a': [1., 2., 3., 4.], 'b': [2., 3., 4., 5.], 'c': [3., 4., 5., 6.]},
                                    columns=['a', 'b', 'c'],
                                    generate_metadata=True)

        print(self.main)


        self.assertEqual(utils.to_json_structure(self.main.metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                # 'top_level': 'main',
                'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
                'structural_type': 'd3m.container.pandas.DataFrame',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Table'],
                'dimension': {
                    'name': 'rows',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                    'length': 4,
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
            'metadata': {'structural_type': 'numpy.float64', 'name': 'a'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {'structural_type': 'numpy.float64', 'name': 'b'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 2],
            'metadata': {'structural_type': 'numpy.float64', 'name': 'c'}
        }])


        self.assertIsInstance(self.main, container.DataFrame)


        hyperparams_class = DeepLogPrimitive.metadata.get_hyperparams()
        hyperparams = hyperparams_class.defaults()
        hyperparams = hyperparams.replace({'batch_size': 4})
        print(hyperparams)

        self.primitive = DeepLogPrimitive(hyperparams=hyperparams)
        self.primitive.set_training_data(inputs=self.main)
        #print("*****************",self.primitive.get_params())

        self.primitive.fit()
        self.new_main = self.primitive.produce(inputs=self.main).value
        self.new_main_score = self.primitive.produce_score(inputs=self.main).value
        print(self.new_main)       
        print(self.new_main_score) 

        params = self.primitive.get_params()
        self.primitive.set_params(params=params)

        self.assertEqual(utils.to_json_structure(self.main.metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                # 'top_level': 'main',
                'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
                'structural_type': 'd3m.container.pandas.DataFrame',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Table'],
                'dimension': {
                    'name': 'rows',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                    'length': 4,
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
            'metadata': {'structural_type': 'numpy.float64', 'name': 'a'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {'structural_type': 'numpy.float64', 'name': 'b'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 2],
            'metadata': {'structural_type': 'numpy.float64', 'name': 'c'}
        }])

    # def test_params(self):
    #     params = self.primitive.get_params()
    #     self.primitive.set_params(params=params)



if __name__ == '__main__':
    unittest.main()
