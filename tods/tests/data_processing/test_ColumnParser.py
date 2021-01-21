import numpy
import os.path
import unittest
import math


from d3m import container, utils
from d3m.metadata import base as metadata_base

from tods.data_processing import DatasetToDataframe, ColumnParser

import utils as test_utils


class ColumnParserPrimitiveTestCase(unittest.TestCase):
    def test_basic(self):
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '..', '..', 'datasets', 'anomaly','yahoo_sub_5','TRAIN','dataset_TRAIN', 'datasetDoc.json'))


        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        hyperparams_class = DatasetToDataframe.DatasetToDataFramePrimitive.metadata.get_hyperparams()

        primitive = DatasetToDataframe.DatasetToDataFramePrimitive(hyperparams=hyperparams_class.defaults())

        call_metadata = primitive.produce(inputs=dataset)

        dataframe = call_metadata.value

        hyperparams_class = ColumnParser.ColumnParserPrimitive.metadata.get_hyperparams()

        primitive = ColumnParser.ColumnParserPrimitive(hyperparams=hyperparams_class.defaults())

        call_metadata = primitive.produce(inputs=dataframe)

        dataframe = call_metadata.value

        first_row = list(dataframe.itertuples(index=False, name=None))[0]

        self.assertEqual(first_row, (0, 1, 12183.0, 0.0, 3.7166666666667, 5.0, 2109.0, 0))

        self.assertEqual([type(o) for o in first_row], [int,int, float,float, float, float, float, int])

        self._test_basic_metadata(dataframe.metadata)

    def _test_basic_metadata(self, metadata):
        self.maxDiff = None

        self.assertEqual(test_utils.convert_through_json(metadata.query(())), {
            'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
            'structural_type': 'd3m.container.pandas.DataFrame',
            'semantic_types': [
                'https://metadata.datadrivendiscovery.org/types/Table',
            ],
            'dimension': {
                'name': 'rows',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                'length': 1260,
            }
        })

        self.assertEqual(test_utils.convert_through_json(metadata.query((metadata_base.ALL_ELEMENTS,))), {
            'dimension': {
                'name': 'columns',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                'length': 8,
            }
        })

        self.assertEqual(test_utils.convert_through_json(metadata.query((metadata_base.ALL_ELEMENTS, 0))), {
            'name': 'd3mIndex',
            'structural_type': 'int',
            'semantic_types': [
                'http://schema.org/Integer',
                'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
            ],
        })



        self.assertEqual(test_utils.convert_through_json(metadata.query((metadata_base.ALL_ELEMENTS, 0))), {'name': 'd3mIndex', 'structural_type': 'int', 'semantic_types': ['http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/PrimaryKey']})
        self.assertEqual(test_utils.convert_through_json(metadata.query((metadata_base.ALL_ELEMENTS, 1))), {'name': 'timestamp', 'structural_type': 'int', 'semantic_types': ['http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/Attribute']})
        self.assertEqual(test_utils.convert_through_json(metadata.query((metadata_base.ALL_ELEMENTS, 2))), {'name': 'value_0', 'structural_type': 'float', 'semantic_types': ['http://schema.org/Float', 'https://metadata.datadrivendiscovery.org/types/Attribute']})

        self.assertEqual(test_utils.convert_through_json(metadata.query((metadata_base.ALL_ELEMENTS, 3))), {'name': 'value_1', 'structural_type': 'float', 'semantic_types': ['http://schema.org/Float', 'https://metadata.datadrivendiscovery.org/types/Attribute']})

        self.assertEqual(test_utils.convert_through_json(metadata.query((metadata_base.ALL_ELEMENTS, 4))), {'name': 'value_2', 'structural_type': 'float', 'semantic_types': ['http://schema.org/Float', 'https://metadata.datadrivendiscovery.org/types/Attribute']})

        self.assertEqual(test_utils.convert_through_json(metadata.query((metadata_base.ALL_ELEMENTS, 5))), {'name': 'value_3', 'structural_type': 'float', 'semantic_types': ['http://schema.org/Float', 'https://metadata.datadrivendiscovery.org/types/Attribute']})

        self.assertEqual(test_utils.convert_through_json(metadata.query((metadata_base.ALL_ELEMENTS, 6))), {'name': 'value_4', 'structural_type': 'float', 'semantic_types': ['http://schema.org/Float', 'https://metadata.datadrivendiscovery.org/types/Attribute']})

        self.assertEqual(test_utils.convert_through_json(metadata.query((metadata_base.ALL_ELEMENTS, 7))), {'name': 'ground_truth', 'structural_type': 'int', 'semantic_types': ['http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/SuggestedTarget', 'https://metadata.datadrivendiscovery.org/types/Attribute']})

    def test_new(self):
        dataset_doc_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..', '..', 'datasets', 'anomaly', 'yahoo_sub_5', 'TRAIN',
                         'dataset_TRAIN', 'datasetDoc.json'))

        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        hyperparams_class = DatasetToDataframe.DatasetToDataFramePrimitive.metadata.get_hyperparams()

        primitive = DatasetToDataframe.DatasetToDataFramePrimitive(hyperparams=hyperparams_class.defaults())

        call_metadata = primitive.produce(inputs=dataset)

        dataframe = call_metadata.value

        hyperparams_class = ColumnParser.ColumnParserPrimitive.metadata.get_hyperparams()

        primitive = ColumnParser.ColumnParserPrimitive(
            hyperparams=hyperparams_class.defaults().replace({'return_result': 'new', 'use_columns': [2]}))

        call_metadata = primitive.produce(inputs=dataframe)

        dataframe = call_metadata.value

        first_row = list(dataframe.itertuples(index=False, name=None))[0]

        self.assertEqual(first_row, ('0', 12183.0))

        self.assertEqual([type(o) for o in first_row], [str, float])

        self._test_new_metadata(dataframe.metadata)

    def _test_new_metadata(self, metadata):
        self.maxDiff = None

        self.assertEqual(test_utils.convert_through_json(metadata.query(())), {
            'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
            'structural_type': 'd3m.container.pandas.DataFrame',
            'semantic_types': [
                'https://metadata.datadrivendiscovery.org/types/Table',
            ],
            'dimension': {
                'name': 'rows',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                'length': 1260,
            }
        })

        self.assertEqual(test_utils.convert_through_json(metadata.query((metadata_base.ALL_ELEMENTS,))), {
            'dimension': {
                'name': 'columns',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                'length': 2,
            }
        })

        self.assertEqual(test_utils.convert_through_json(metadata.query((metadata_base.ALL_ELEMENTS, 0))), {
            'name': 'd3mIndex',
            'structural_type': 'str',
            'semantic_types': [
                'http://schema.org/Integer',
                'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
            ],
        })

        self.assertEqual(test_utils.convert_through_json(metadata.query((metadata_base.ALL_ELEMENTS, 1))), {
            'name': 'value_0',
            'structural_type': 'float',
            'semantic_types': [
                'http://schema.org/Float',
                'https://metadata.datadrivendiscovery.org/types/Attribute',
            ],
        })

    def test_append(self):
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '..', '..', 'datasets', 'anomaly','yahoo_sub_5','TRAIN','dataset_TRAIN', 'datasetDoc.json'))

        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        hyperparams_class = DatasetToDataframe.DatasetToDataFramePrimitive.metadata.get_hyperparams()

        primitive = DatasetToDataframe.DatasetToDataFramePrimitive(hyperparams=hyperparams_class.defaults())

        call_metadata = primitive.produce(inputs=dataset)

        dataframe = call_metadata.value

        hyperparams_class = ColumnParser.ColumnParserPrimitive.metadata.get_hyperparams()

        primitive = ColumnParser.ColumnParserPrimitive(hyperparams=hyperparams_class.defaults().replace(
            {'return_result': 'append', 'replace_index_columns': False,
             'parse_semantic_types': ['https://metadata.datadrivendiscovery.org/types/CategoricalData',
                                      'http://schema.org/Integer']}))

        call_metadata = primitive.produce(inputs=dataframe)

        dataframe = call_metadata.value

        first_row = list(dataframe.itertuples(index=False, name=None))[0]

        self.assertEqual(first_row, ('0', '1', '12183', '0.0', '3.7166666666667', '5', '2109', '0', 0, 1, 0))

        self.assertEqual([type(o) for o in first_row], [str, str, str, str, str, str, str, str,int , int , int])

        self._test_append_metadata(dataframe.metadata, False)

    def test_append_replace_index_columns(self):
        dataset_doc_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..', '..', 'datasets', 'anomaly', 'yahoo_sub_5', 'TRAIN',
                         'dataset_TRAIN', 'datasetDoc.json'))

        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        hyperparams_class = DatasetToDataframe.DatasetToDataFramePrimitive.metadata.get_hyperparams()

        primitive = DatasetToDataframe.DatasetToDataFramePrimitive(hyperparams=hyperparams_class.defaults())

        call_metadata = primitive.produce(inputs=dataset)

        dataframe = call_metadata.value

        hyperparams_class = ColumnParser.ColumnParserPrimitive.metadata.get_hyperparams()

        primitive = ColumnParser.ColumnParserPrimitive(hyperparams=hyperparams_class.defaults().replace(
            {'return_result': 'append',
             'parse_semantic_types': ['https://metadata.datadrivendiscovery.org/types/CategoricalData',
                                      'http://schema.org/Integer']}))

        call_metadata = primitive.produce(inputs=dataframe)

        dataframe = call_metadata.value

        first_row = list(dataframe.itertuples(index=False, name=None))[0]

        self.assertEqual(first_row, (0, '1', '12183', '0.0', '3.7166666666667', '5', '2109', '0', 1, 0))

        self.assertEqual([type(o) for o in first_row], [int, str, str, str, str, str,str,str, int , int])

        self._test_append_replace_metadata(dataframe.metadata, True)

    def _test_append_replace_metadata(self, metadata, replace_index_columns):
        self.maxDiff = None

        self.assertEqual(test_utils.convert_through_json(metadata.query(())), {
            'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
            'structural_type': 'd3m.container.pandas.DataFrame',
            'semantic_types': [
                'https://metadata.datadrivendiscovery.org/types/Table',
            ],
            'dimension': {
                'name': 'rows',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                'length': 1260,
            }
        })

        self.assertEqual(test_utils.convert_through_json(metadata.query((metadata_base.ALL_ELEMENTS,))), {
            'dimension': {
                'name': 'columns',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                'length': 10,
            }
        })


        self.assertEqual(test_utils.convert_through_json(metadata.query((metadata_base.ALL_ELEMENTS, 0))),
                         {'name': 'd3mIndex', 'structural_type': 'int', 'semantic_types': ['http://schema.org/Integer',
                                                                                           'https://metadata.datadrivendiscovery.org/types/PrimaryKey']})
        self.assertEqual(test_utils.convert_through_json(metadata.query((metadata_base.ALL_ELEMENTS, 1))),
                         {'name': 'timestamp', 'structural_type': 'str', 'semantic_types': ['http://schema.org/Integer',
                                                                                            'https://metadata.datadrivendiscovery.org/types/Attribute']})
        self.assertEqual(test_utils.convert_through_json(metadata.query((metadata_base.ALL_ELEMENTS, 2))),
                         {'name': 'value_0', 'structural_type': 'str', 'semantic_types': ['http://schema.org/Float',
                                                                                            'https://metadata.datadrivendiscovery.org/types/Attribute']})

        self.assertEqual(test_utils.convert_through_json(metadata.query((metadata_base.ALL_ELEMENTS, 3))),
                         {'name': 'value_1', 'structural_type': 'str', 'semantic_types': ['http://schema.org/Float',
                                                                                            'https://metadata.datadrivendiscovery.org/types/Attribute']})

        self.assertEqual(test_utils.convert_through_json(metadata.query((metadata_base.ALL_ELEMENTS, 4))),
                         {'name': 'value_2', 'structural_type': 'str', 'semantic_types': ['http://schema.org/Float',
                                                                                            'https://metadata.datadrivendiscovery.org/types/Attribute']})

        self.assertEqual(test_utils.convert_through_json(metadata.query((metadata_base.ALL_ELEMENTS, 5))),
                         {'name': 'value_3', 'structural_type': 'str', 'semantic_types': ['http://schema.org/Float',
                                                                                            'https://metadata.datadrivendiscovery.org/types/Attribute']})

        self.assertEqual(test_utils.convert_through_json(metadata.query((metadata_base.ALL_ELEMENTS, 6))),
                         {'name': 'value_4', 'structural_type': 'str', 'semantic_types': ['http://schema.org/Float',
                                                                                            'https://metadata.datadrivendiscovery.org/types/Attribute']})

        self.assertEqual(test_utils.convert_through_json(metadata.query((metadata_base.ALL_ELEMENTS, 7))),
                         {'name': 'ground_truth', 'structural_type': 'str',
                          'semantic_types': ['http://schema.org/Integer',
                                             'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',
                                             'https://metadata.datadrivendiscovery.org/types/Attribute']})

        self.assertEqual(test_utils.convert_through_json(metadata.query((metadata_base.ALL_ELEMENTS, 8))),{'name': 'timestamp', 'structural_type': 'int', 'semantic_types': ['http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/Attribute']})
        self.assertEqual(test_utils.convert_through_json(metadata.query((metadata_base.ALL_ELEMENTS, 9))),{'name': 'ground_truth',
 'semantic_types': ['http://schema.org/Integer',
                    'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',
                    'https://metadata.datadrivendiscovery.org/types/Attribute'],
 'structural_type': 'int'})


    def _test_append_metadata(self, metadata, replace_index_columns):
        self.maxDiff = None

        self.assertEqual(test_utils.convert_through_json(metadata.query(())), {
            'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
            'structural_type': 'd3m.container.pandas.DataFrame',
            'semantic_types': [
                'https://metadata.datadrivendiscovery.org/types/Table',
            ],
            'dimension': {
                'name': 'rows',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                'length': 1260,
            }
        })

        self.assertEqual(test_utils.convert_through_json(metadata.query((metadata_base.ALL_ELEMENTS,))), {
            'dimension': {
                'name': 'columns',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                'length': 11,
            }
        })


        self.assertEqual(test_utils.convert_through_json(metadata.query((metadata_base.ALL_ELEMENTS, 0))),
                         {'name': 'd3mIndex', 'structural_type': 'str', 'semantic_types': ['http://schema.org/Integer',
                                                                                           'https://metadata.datadrivendiscovery.org/types/PrimaryKey']})
        self.assertEqual(test_utils.convert_through_json(metadata.query((metadata_base.ALL_ELEMENTS, 1))),
                         {'name': 'timestamp', 'structural_type': 'str', 'semantic_types': ['http://schema.org/Integer',
                                                                                            'https://metadata.datadrivendiscovery.org/types/Attribute']})
        self.assertEqual(test_utils.convert_through_json(metadata.query((metadata_base.ALL_ELEMENTS, 2))),
                         {'name': 'value_0', 'structural_type': 'str', 'semantic_types': ['http://schema.org/Float',
                                                                                            'https://metadata.datadrivendiscovery.org/types/Attribute']})

        self.assertEqual(test_utils.convert_through_json(metadata.query((metadata_base.ALL_ELEMENTS, 3))),
                         {'name': 'value_1', 'structural_type': 'str', 'semantic_types': ['http://schema.org/Float',
                                                                                            'https://metadata.datadrivendiscovery.org/types/Attribute']})

        self.assertEqual(test_utils.convert_through_json(metadata.query((metadata_base.ALL_ELEMENTS, 4))),
                         {'name': 'value_2', 'structural_type': 'str', 'semantic_types': ['http://schema.org/Float',
                                                                                            'https://metadata.datadrivendiscovery.org/types/Attribute']})

        self.assertEqual(test_utils.convert_through_json(metadata.query((metadata_base.ALL_ELEMENTS, 5))),
                         {'name': 'value_3', 'structural_type': 'str', 'semantic_types': ['http://schema.org/Float',
                                                                                            'https://metadata.datadrivendiscovery.org/types/Attribute']})

        self.assertEqual(test_utils.convert_through_json(metadata.query((metadata_base.ALL_ELEMENTS, 6))),
                         {'name': 'value_4', 'structural_type': 'str', 'semantic_types': ['http://schema.org/Float',
                                                                                            'https://metadata.datadrivendiscovery.org/types/Attribute']})

        self.assertEqual(test_utils.convert_through_json(metadata.query((metadata_base.ALL_ELEMENTS, 7))),
                         {'name': 'ground_truth', 'structural_type': 'str',
                          'semantic_types': ['http://schema.org/Integer',
                                             'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',
                                             'https://metadata.datadrivendiscovery.org/types/Attribute']})

        self.assertEqual(test_utils.convert_through_json(metadata.query((metadata_base.ALL_ELEMENTS, 8))),{'name': 'd3mIndex', 'structural_type': 'int', 'semantic_types': ['http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/PrimaryKey']})
        self.assertEqual(test_utils.convert_through_json(metadata.query((metadata_base.ALL_ELEMENTS, 9))),{'name': 'timestamp', 'structural_type': 'int', 'semantic_types': ['http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/Attribute']})
        self.assertEqual(test_utils.convert_through_json(metadata.query((metadata_base.ALL_ELEMENTS, 10))),{'name': 'ground_truth', 'structural_type': 'int', 'semantic_types': ['http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/SuggestedTarget', 'https://metadata.datadrivendiscovery.org/types/Attribute']})


    def test_integer(self):
        hyperparams_class = ColumnParser.ColumnParserPrimitive.metadata.get_hyperparams()

        primitive = ColumnParser.ColumnParserPrimitive(hyperparams=hyperparams_class.defaults())

        dataframe = container.DataFrame({'a': ['1.0', '2.0', '3.0']}, generate_metadata=True)

        dataframe.metadata = dataframe.metadata.update((metadata_base.ALL_ELEMENTS, 0), {
            'name': 'test',
            'semantic_types': [
                'http://schema.org/Integer',
                'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
            ],
        })

        call_metadata = primitive.produce(inputs=dataframe)

        parsed_dataframe = call_metadata.value

        self.assertEqual(
            test_utils.convert_through_json(parsed_dataframe.metadata.query((metadata_base.ALL_ELEMENTS, 0))), {
                'name': 'test',
                'structural_type': 'int',
                'semantic_types': [
                    'http://schema.org/Integer',
                    'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                ],
            })

        self.assertEqual(list(parsed_dataframe.iloc[:, 0]), [1, 2, 3])

        dataframe.iloc[2, 0] = '3.1'

        call_metadata = primitive.produce(inputs=dataframe)

        parsed_dataframe = call_metadata.value

        self.assertEqual(
            test_utils.convert_through_json(parsed_dataframe.metadata.query((metadata_base.ALL_ELEMENTS, 0))), {
                'name': 'test',
                'structural_type': 'int',
                'semantic_types': [
                    'http://schema.org/Integer',
                    'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                ],
            })

        self.assertEqual(list(parsed_dataframe.iloc[:, 0]), [1, 2, 3])

        dataframe.iloc[2, 0] = 'aaa'

        with self.assertRaisesRegex(ValueError,
                                    'Not all values in a column can be parsed into integers, but only integers were expected'):
            primitive.produce(inputs=dataframe)

        dataframe.metadata = dataframe.metadata.update((metadata_base.ALL_ELEMENTS, 0), {
            'name': 'test',
            'structural_type': str,
            'semantic_types': [
                'http://schema.org/Integer',
            ],
        })

        call_metadata = primitive.produce(inputs=dataframe)

        parsed_dataframe = call_metadata.value

        self.assertEqual(
            test_utils.convert_through_json(parsed_dataframe.metadata.query((metadata_base.ALL_ELEMENTS, 0))), {
                'name': 'test',
                'structural_type': 'float',
                'semantic_types': [
                    'http://schema.org/Integer',
                ],
            })

        self.assertEqual(list(parsed_dataframe.iloc[0:2, 0]), [1.0, 2.0])
        self.assertTrue(math.isnan(parsed_dataframe.iloc[2, 0]))







if __name__ == '__main__':
    unittest.main()
