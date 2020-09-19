import os
import logging
import unittest

import numpy

from d3m import container
from d3m.metadata import base as metadata_base

from common_primitives import cast_to_type, column_parser, dataset_to_dataframe, extract_columns_semantic_types


class CastToTypePrimitiveTestCase(unittest.TestCase):
    def test_basic(self):
        inputs = container.DataFrame({'a': [1, 2, 3], 'b': ['a', 'b', 'c']}, generate_metadata=True)

        self.assertEqual(inputs.dtypes['a'], numpy.int64)
        self.assertEqual(inputs.dtypes['b'], object)

        hyperparams_class = cast_to_type.CastToTypePrimitive.metadata.get_hyperparams()

        primitive = cast_to_type.CastToTypePrimitive(hyperparams=hyperparams_class.defaults().replace({'type_to_cast': 'str'}))

        call_metadata = primitive.produce(inputs=inputs)

        self.assertIsInstance(call_metadata.value, container.DataFrame)

        self.assertEqual(len(call_metadata.value.dtypes), 2)
        self.assertEqual(call_metadata.value.dtypes['a'], object)
        self.assertEqual(call_metadata.value.dtypes['b'], object)

        self.assertEqual(call_metadata.value.metadata.query((metadata_base.ALL_ELEMENTS, 0))['structural_type'], str)
        self.assertEqual(call_metadata.value.metadata.query((metadata_base.ALL_ELEMENTS, 1))['structural_type'], str)
        self.assertEqual(call_metadata.value.metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length'], 2)

        primitive = cast_to_type.CastToTypePrimitive(hyperparams=hyperparams_class.defaults().replace({'type_to_cast': 'float'}))

        with self.assertLogs(level=logging.WARNING) as cm:
            call_metadata = primitive.produce(inputs=inputs)

        self.assertEqual(len(call_metadata.value.dtypes), 1)
        self.assertEqual(call_metadata.value.dtypes['a'], float)

        self.assertEqual(call_metadata.value.metadata.query((metadata_base.ALL_ELEMENTS, 0))['structural_type'], float)
        self.assertEqual(call_metadata.value.metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length'], 1)

        self.assertEqual(len(cm.records), 1)
        self.assertEqual(cm.records[0].msg, "Not all columns can be cast to type '%(type)s'. Skipping columns: %(columns)s")

        primitive = cast_to_type.CastToTypePrimitive(hyperparams=hyperparams_class.defaults().replace({'exclude_columns': (0,), 'type_to_cast': 'float'}))

        with self.assertRaisesRegex(ValueError, 'No columns to be cast to type'):
            primitive.produce(inputs=inputs)

    def test_objects(self):
        hyperparams_class = cast_to_type.CastToTypePrimitive.metadata.get_hyperparams()

        inputs = container.DataFrame({'a': [1, 2, 3], 'b': [{'a': 1}, {'b': 1}, {'c': 1}]}, {
            'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
            'structural_type': container.DataFrame,
            'dimension': {
                'length': 3,
            },
        }, generate_metadata=False)
        inputs.metadata = inputs.metadata.update((metadata_base.ALL_ELEMENTS,), {
            'dimension': {
                'length': 2,
            },
        })
        inputs.metadata = inputs.metadata.update((metadata_base.ALL_ELEMENTS, 0), {
            'structural_type': int,
        })
        inputs.metadata = inputs.metadata.update((metadata_base.ALL_ELEMENTS, 1), {
            'structural_type': dict,
        })

        self.assertEqual(inputs.dtypes['a'], numpy.int64)
        self.assertEqual(inputs.dtypes['b'], object)

        primitive = cast_to_type.CastToTypePrimitive(hyperparams=hyperparams_class.defaults().replace({'type_to_cast': 'str'}))

        call_metadata = primitive.produce(inputs=inputs)

        self.assertEqual(len(call_metadata.value.dtypes), 2)
        self.assertEqual(call_metadata.value.dtypes['a'], object)
        self.assertEqual(call_metadata.value.dtypes['b'], object)

        self.assertEqual(call_metadata.value.metadata.query((metadata_base.ALL_ELEMENTS, 0))['structural_type'], str)
        self.assertEqual(call_metadata.value.metadata.query((metadata_base.ALL_ELEMENTS, 1))['structural_type'], str)
        self.assertEqual(call_metadata.value.metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length'], 2)

        primitive = cast_to_type.CastToTypePrimitive(hyperparams=hyperparams_class.defaults().replace({'type_to_cast': 'float'}))

        with self.assertLogs(level=logging.WARNING) as cm:
            call_metadata = primitive.produce(inputs=inputs)

        self.assertEqual(len(call_metadata.value.dtypes), 1)
        self.assertEqual(call_metadata.value.dtypes['a'], float)

        self.assertEqual(call_metadata.value.metadata.query((metadata_base.ALL_ELEMENTS, 0))['structural_type'], float)
        self.assertEqual(call_metadata.value.metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length'], 1)

        self.assertEqual(len(cm.records), 1)
        self.assertEqual(cm.records[0].msg, "Not all columns can be cast to type '%(type)s'. Skipping columns: %(columns)s")

    def test_data(self):
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'iris_dataset_1', 'datasetDoc.json'))

        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        hyperparams_class = dataset_to_dataframe.DatasetToDataFramePrimitive.metadata.get_hyperparams()
        primitive = dataset_to_dataframe.DatasetToDataFramePrimitive(hyperparams=hyperparams_class.defaults())
        dataframe = primitive.produce(inputs=dataset).value

        hyperparams_class = column_parser.ColumnParserPrimitive.metadata.get_hyperparams()
        primitive = column_parser.ColumnParserPrimitive(hyperparams=hyperparams_class.defaults())
        dataframe = primitive.produce(inputs=dataframe).value

        hyperparams_class = extract_columns_semantic_types.ExtractColumnsBySemanticTypesPrimitive.metadata.get_hyperparams()
        primitive = extract_columns_semantic_types.ExtractColumnsBySemanticTypesPrimitive(hyperparams=hyperparams_class.defaults())
        attributes = primitive.produce(inputs=dataframe).value

        hyperparams_class = cast_to_type.CastToTypePrimitive.metadata.get_hyperparams()
        primitive = cast_to_type.CastToTypePrimitive(hyperparams=hyperparams_class.defaults().replace({'type_to_cast': 'float'}))
        cast_attributes = primitive.produce(inputs=attributes).value

        self.assertEqual(cast_attributes.values.dtype, numpy.float64)


if __name__ == '__main__':
    unittest.main()
