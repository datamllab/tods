import unittest
import os

import numpy

from d3m import container
from d3m.metadata import base as metadata_base

from common_primitives import dataset_to_dataframe, extract_columns_semantic_types, horizontal_concat


class HorizontalConcatPrimitiveTestCase(unittest.TestCase):
    def test_basic(self):
        test_data_inputs = {'col1': [1.0, 2.0, 3.0]}
        dataframe_inputs = container.DataFrame(data=test_data_inputs, generate_metadata=True)

        test_data_targets = {'col2': [1, 2 ,3]}
        dataframe_targets = container.DataFrame(data=test_data_targets, generate_metadata=True)

        hyperparams_class = horizontal_concat.HorizontalConcatPrimitive.metadata.get_hyperparams()

        primitive = horizontal_concat.HorizontalConcatPrimitive(hyperparams=hyperparams_class.defaults())

        call_result = primitive.produce(left=dataframe_inputs, right=dataframe_targets)

        dataframe_concat = call_result.value

        self.assertEqual(dataframe_concat.values.tolist(), [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])

        self._test_basic_metadata(dataframe_concat.metadata)

    def _test_basic_metadata(self, metadata):
        self.assertEqual(metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length'], 2)
        self.assertEqual(metadata.query((metadata_base.ALL_ELEMENTS, 0))['name'], 'col1')
        self.assertEqual(metadata.query((metadata_base.ALL_ELEMENTS, 0))['structural_type'], numpy.float64)
        self.assertEqual(metadata.query((metadata_base.ALL_ELEMENTS, 1))['name'], 'col2')
        self.assertEqual(metadata.query((metadata_base.ALL_ELEMENTS, 1))['structural_type'], numpy.int64)

    def _get_iris(self):
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'iris_dataset_1', 'datasetDoc.json'))

        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        # We set semantic types like runtime would.
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 5), 'https://metadata.datadrivendiscovery.org/types/Target')
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 5), 'https://metadata.datadrivendiscovery.org/types/TrueTarget')
        dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 5), 'https://metadata.datadrivendiscovery.org/types/Attribute')

        hyperparams_class = dataset_to_dataframe.DatasetToDataFramePrimitive.metadata.get_hyperparams()

        primitive = dataset_to_dataframe.DatasetToDataFramePrimitive(hyperparams=hyperparams_class.defaults())

        call_metadata = primitive.produce(inputs=dataset)

        dataframe = call_metadata.value

        return dataframe

    def _get_iris_columns(self):
        dataframe = self._get_iris()

        hyperparams_class = extract_columns_semantic_types.ExtractColumnsBySemanticTypesPrimitive.metadata.get_hyperparams()

        primitive = extract_columns_semantic_types.ExtractColumnsBySemanticTypesPrimitive(hyperparams=hyperparams_class.defaults().replace({'semantic_types': ('https://metadata.datadrivendiscovery.org/types/PrimaryKey',)}))

        call_metadata = primitive.produce(inputs=dataframe)

        index = call_metadata.value

        primitive = extract_columns_semantic_types.ExtractColumnsBySemanticTypesPrimitive(hyperparams=hyperparams_class.defaults().replace({'semantic_types': ('https://metadata.datadrivendiscovery.org/types/Attribute',)}))

        call_metadata = primitive.produce(inputs=dataframe)

        attributes = call_metadata.value

        primitive = extract_columns_semantic_types.ExtractColumnsBySemanticTypesPrimitive(hyperparams=hyperparams_class.defaults().replace({'semantic_types': ('https://metadata.datadrivendiscovery.org/types/SuggestedTarget',)}))

        call_metadata = primitive.produce(inputs=dataframe)

        targets = call_metadata.value

        return dataframe, index, attributes, targets

    def test_iris(self):
        dataframe, index, attributes, targets = self._get_iris_columns()

        hyperparams_class = horizontal_concat.HorizontalConcatPrimitive.metadata.get_hyperparams()

        primitive = horizontal_concat.HorizontalConcatPrimitive(hyperparams=hyperparams_class.defaults())

        call_metadata = primitive.produce(left=index, right=attributes)

        call_metadata = primitive.produce(left=call_metadata.value, right=targets)

        new_dataframe = call_metadata.value

        self.assertEqual(dataframe.values.tolist(), new_dataframe.values.tolist())

        self._test_iris_metadata(dataframe.metadata, new_dataframe.metadata)

    def _test_iris_metadata(self, metadata, new_metadata):
        self.assertEqual(metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length'], new_metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length'])

        for i in range(new_metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length']):
            self.assertEqual(metadata.query((metadata_base.ALL_ELEMENTS, i)), new_metadata.query((metadata_base.ALL_ELEMENTS, i)), i)

    def _get_iris_columns_with_index(self):
        dataframe = self._get_iris()

        hyperparams_class = extract_columns_semantic_types.ExtractColumnsBySemanticTypesPrimitive.metadata.get_hyperparams()

        primitive = extract_columns_semantic_types.ExtractColumnsBySemanticTypesPrimitive(hyperparams=hyperparams_class.defaults().replace({'semantic_types': ('https://metadata.datadrivendiscovery.org/types/PrimaryKey',)}))

        call_metadata = primitive.produce(inputs=dataframe)

        index = call_metadata.value

        primitive = extract_columns_semantic_types.ExtractColumnsBySemanticTypesPrimitive(hyperparams=hyperparams_class.defaults().replace({'semantic_types': ('https://metadata.datadrivendiscovery.org/types/PrimaryKey', 'https://metadata.datadrivendiscovery.org/types/Attribute')}))

        call_metadata = primitive.produce(inputs=dataframe)

        attributes = call_metadata.value

        primitive = extract_columns_semantic_types.ExtractColumnsBySemanticTypesPrimitive(hyperparams=hyperparams_class.defaults().replace({'semantic_types': ('https://metadata.datadrivendiscovery.org/types/PrimaryKey', 'https://metadata.datadrivendiscovery.org/types/SuggestedTarget')}))

        call_metadata = primitive.produce(inputs=dataframe)

        targets = call_metadata.value

        return dataframe, index, attributes, targets

    def test_iris_with_index_removed(self):
        dataframe, index, attributes, targets = self._get_iris_columns_with_index()

        hyperparams_class = horizontal_concat.HorizontalConcatPrimitive.metadata.get_hyperparams()

        primitive = horizontal_concat.HorizontalConcatPrimitive(hyperparams=hyperparams_class.defaults().replace({'use_index': False}))

        call_metadata = primitive.produce(left=index, right=attributes)

        call_metadata = primitive.produce(left=call_metadata.value, right=targets)

        new_dataframe = call_metadata.value

        self.assertEqual(dataframe.values.tolist(), new_dataframe.values.tolist())

        self._test_iris_with_index_removed_metadata(dataframe.metadata, new_dataframe.metadata)

    def _test_iris_with_index_removed_metadata(self, metadata, new_metadata):
        self.assertEqual(metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length'], new_metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length'])

        for i in range(new_metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length']):
            self.assertEqual(metadata.query((metadata_base.ALL_ELEMENTS, i)), new_metadata.query((metadata_base.ALL_ELEMENTS, i)), i)

    def test_iris_with_index_reorder(self):
        dataframe, index, attributes, targets = self._get_iris_columns_with_index()

        # Let's make problems.
        attributes = attributes.sort_values(by='sepalLength').reset_index(drop=True)

        hyperparams_class = horizontal_concat.HorizontalConcatPrimitive.metadata.get_hyperparams()

        primitive = horizontal_concat.HorizontalConcatPrimitive(hyperparams=hyperparams_class.defaults())

        call_metadata = primitive.produce(left=index, right=attributes)

        call_metadata = primitive.produce(left=call_metadata.value, right=targets)

        new_dataframe = call_metadata.value

        self.assertEqual(dataframe.values.tolist(), new_dataframe.values.tolist())

        self._test_iris_with_index_reorder_metadata(dataframe.metadata, new_dataframe.metadata)

    def _test_iris_with_index_reorder_metadata(self, metadata, new_metadata):
        self.assertEqual(metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length'], new_metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length'])

        for i in range(new_metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length']):
            self.assertEqual(metadata.query((metadata_base.ALL_ELEMENTS, i)), new_metadata.query((metadata_base.ALL_ELEMENTS, i)), i)


if __name__ == '__main__':
    unittest.main()
