import os
import unittest

import pandas as pd

from d3m import container
from d3m.metadata import base as metadata_base

from common_primitives import dataset_to_dataframe, column_parser, rename_duplicate_columns


class RenameDuplicateColumnsPrimitiveTestCase(unittest.TestCase):
    def _get_iris(self):
        dataset_doc_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'iris_dataset_1', 'datasetDoc.json'))

        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        hyperparams_class = \
            dataset_to_dataframe.DatasetToDataFramePrimitive.metadata.query()['primitive_code']['class_type_arguments'][
                'Hyperparams']
        primitive = dataset_to_dataframe.DatasetToDataFramePrimitive(hyperparams=hyperparams_class.defaults())

        dataframe = primitive.produce(inputs=dataset).value

        return dataframe

    def _get_iris_columns(self):
        dataframe = self._get_iris()
        # We set semantic types like runtime would.
        dataframe.metadata = dataframe.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, 5),
                                                                  'https://metadata.datadrivendiscovery.org/types/Target')
        dataframe.metadata = dataframe.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, 5),
                                                                  'https://metadata.datadrivendiscovery.org/types/TrueTarget')
        dataframe.metadata = dataframe.metadata.remove_semantic_type((metadata_base.ALL_ELEMENTS, 5),
                                                                     'https://metadata.datadrivendiscovery.org/types/Attribute')

        # Parsing.
        hyperparams_class = \
            column_parser.ColumnParserPrimitive.metadata.query()['primitive_code']['class_type_arguments'][
                'Hyperparams']
        primitive = column_parser.ColumnParserPrimitive(hyperparams=hyperparams_class.defaults())
        dataframe = primitive.produce(inputs=dataframe).value

        return dataframe

    def test_basic(self):
        test_data_inputs = {'col1': [1.0, 2.0, 3.0],
                            'col2': [4.0, 5.0, 6.0],
                            'col3': [100, 200, 300]}
        dataframe_inputs = container.DataFrame.from_dict(data=test_data_inputs)
        test_data_inputs_dup = {'col1': [1.0, 2.0, 3.0],
                                'col2': [4.0, 5.0, 6.0]}
        dataframe_inputs_dup = container.DataFrame.from_dict(data=test_data_inputs_dup)
        test_data_inputs_dup_2 = {'col1': [1.0, 2.0, 3.0],
                                  'col2': [4.0, 5.0, 6.0],
                                  'col3': [100, 200, 300]}
        dataframe_inputs_dup_2 = container.DataFrame.from_dict(data=test_data_inputs_dup_2)
        input = pd.concat([dataframe_inputs, dataframe_inputs_dup, dataframe_inputs_dup_2], axis=1)

        hyperparams_class = rename_duplicate_columns.RenameDuplicateColumnsPrimitive.metadata.query()['primitive_code'][
            'class_type_arguments']['Hyperparams']

        primitive = rename_duplicate_columns.RenameDuplicateColumnsPrimitive(hyperparams=hyperparams_class.defaults())

        call_result = primitive.produce(inputs=input)
        dataframe_renamed = call_result.value
        self.assertEqual(dataframe_renamed.columns.values.tolist(),
                         ['col1', 'col2', 'col3', 'col1.1', 'col2.1', 'col1.2', 'col2.2', 'col3.1'])

    def test_monotonic_dup_col_name(self):
        """This test is added because of issue #73"""
        test_data_inputs = {'a': [1.0, 2.0, 3.0],
                            'b': [100, 200, 300]}
        dataframe_inputs = container.DataFrame.from_dict(data=test_data_inputs)
        test_data_inputs_dup = {'b': [1.0, 2.0, 3.0],
                                'c': [4.0, 5.0, 6.0]}
        dataframe_inputs_dup = container.DataFrame.from_dict(data=test_data_inputs_dup)
        input = pd.concat([dataframe_inputs, dataframe_inputs_dup], axis=1)

        hyperparams_class = rename_duplicate_columns.RenameDuplicateColumnsPrimitive.metadata.query()['primitive_code'][
            'class_type_arguments']['Hyperparams']

        primitive = rename_duplicate_columns.RenameDuplicateColumnsPrimitive(hyperparams=hyperparams_class.defaults())

        call_result = primitive.produce(inputs=input)
        dataframe_renamed = call_result.value
        self.assertEqual(dataframe_renamed.columns.values.tolist(),
                         ['a', 'b', 'b.1', 'c'])

    def test_no_change(self):
        test_data_inputs = {'col0': [1.0, 2.0, 3.0],
                            'col1': [4.0, 5.0, 6.0],
                            'col2': [100, 200, 300]}
        dataframe_inputs = container.DataFrame.from_dict(data=test_data_inputs)
        test_data_inputs = {'col3': [1.0, 2.0, 3.0],
                            'col4': [4.0, 5.0, 6.0],
                            'col5': [100, 200, 300]}
        dataframe_inputs_2 = container.DataFrame.from_dict(data=test_data_inputs)

        inputs = pd.concat([dataframe_inputs, dataframe_inputs_2], axis=1)
        hyperparams_class = rename_duplicate_columns.RenameDuplicateColumnsPrimitive.metadata.query()['primitive_code'][
            'class_type_arguments']['Hyperparams']

        primitive = rename_duplicate_columns.RenameDuplicateColumnsPrimitive(hyperparams=hyperparams_class.defaults())

        call_result = primitive.produce(inputs=inputs)
        dataframe_renamed = call_result.value

        self.assertEqual(dataframe_renamed.columns.values.tolist(),
                         ['col0', 'col1', 'col2', 'col3', 'col4', 'col5'])

    def test_iris_with_metadata(self):
        dataframe = self._get_iris_columns()
        dataframe_1 = self._get_iris_columns()
        dataframe_concated = dataframe.append_columns(dataframe_1)
        dataframe_concated_bk = dataframe_concated.copy()
        hyperparams_class = rename_duplicate_columns.RenameDuplicateColumnsPrimitive.metadata.query()['primitive_code'][
            'class_type_arguments']['Hyperparams']

        primitive = rename_duplicate_columns.RenameDuplicateColumnsPrimitive(hyperparams=hyperparams_class.defaults())

        call_result = primitive.produce(inputs=dataframe_concated)
        dataframe_renamed = call_result.value
        names = ['d3mIndex', 'sepalLength', 'sepalWidth', 'petalLength', 'petalWidth', 'species',
                 'd3mIndex.1', 'sepalLength.1', 'sepalWidth.1', 'petalLength.1', 'petalWidth.1',
                 'species.1']
        self.assertEqual(dataframe_renamed.columns.values.tolist(), names)
        self.assertTrue(dataframe_concated.equals(dataframe_concated_bk))
        self.assertTrue(dataframe_concated.metadata.to_internal_json_structure(),
                        dataframe_concated_bk.metadata.to_internal_json_structure())

        for i, column_name in enumerate(dataframe_renamed.columns):
            self.assertEqual(dataframe_renamed.metadata.query_column(i)['other_name'],
                             column_name.split(primitive.hyperparams['separator'])[0])
            self.assertEqual(dataframe_renamed.metadata.query_column(i)['name'], names[i])
