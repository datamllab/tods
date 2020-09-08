import os
import unittest

from d3m import container, utils
from d3m.metadata import base as metadata_base

from common_primitives import normalize_graphs, denormalize, dataset_map, column_parser, normalize_column_references, simple_profiler

import utils as test_utils


class NormalizeGraphsPrimitiveTestCase(unittest.TestCase):
    def _parse_columns(self, dataset):
        hyperparams_class = dataset_map.DataFrameDatasetMapPrimitive.metadata.get_hyperparams()

        primitive = dataset_map.DataFrameDatasetMapPrimitive(
            # We have to make an instance of the primitive ourselves.
            hyperparams=hyperparams_class.defaults().replace({
                'primitive': column_parser.ColumnParserPrimitive(
                    hyperparams=column_parser.ColumnParserPrimitive.metadata.get_hyperparams().defaults(),
                ),
                'resources': 'all',
            }),

        )

        return primitive.produce(inputs=dataset).value

    def _normalize_column_references(self, dataset):
        hyperparams_class = normalize_column_references.NormalizeColumnReferencesPrimitive.metadata.get_hyperparams()

        primitive = normalize_column_references.NormalizeColumnReferencesPrimitive(
            hyperparams=hyperparams_class.defaults(),
        )

        return primitive.produce(inputs=dataset).value

    def _get_dataset_1(self):
        dataset_doc_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'graph_dataset_1', 'datasetDoc.json')
        )

        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        # We set semantic types like runtime would.
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 2), 'https://metadata.datadrivendiscovery.org/types/Target')
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 2), 'https://metadata.datadrivendiscovery.org/types/TrueTarget')
        dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 2), 'https://metadata.datadrivendiscovery.org/types/Attribute')

        metadata_before = dataset.metadata.to_internal_json_structure()

        normalized_dataset = self._normalize_column_references(dataset)

        hyperparams_class = normalize_graphs.NormalizeGraphsPrimitive.metadata.get_hyperparams()

        primitive = normalize_graphs.NormalizeGraphsPrimitive(
            hyperparams=hyperparams_class.defaults(),
        )

        normalized_dataset = primitive.produce(inputs=normalized_dataset).value

        hyperparams_class = dataset_map.DataFrameDatasetMapPrimitive.metadata.get_hyperparams()

        primitive = dataset_map.DataFrameDatasetMapPrimitive(
            # We have to make an instance of the primitive ourselves.
            hyperparams=hyperparams_class.defaults().replace({
                'primitive': simple_profiler.SimpleProfilerPrimitive(
                    hyperparams=simple_profiler.SimpleProfilerPrimitive.metadata.get_hyperparams().defaults().replace({
                        'detect_semantic_types': [
                            'http://schema.org/Boolean', 'https://metadata.datadrivendiscovery.org/types/CategoricalData',
                            'http://schema.org/Integer', 'http://schema.org/Float', 'http://schema.org/Text',
                            'https://metadata.datadrivendiscovery.org/types/FloatVector', 'http://schema.org/DateTime',
                            'https://metadata.datadrivendiscovery.org/types/Attribute',
                            'https://metadata.datadrivendiscovery.org/types/Time',
                            'https://metadata.datadrivendiscovery.org/types/UnknownType',
                        ],
                    }),
                ),
                'resources': 'all',
            }),

        )

        primitive.set_training_data(inputs=normalized_dataset)
        primitive.fit()
        normalized_dataset = primitive.produce(inputs=normalized_dataset).value

        normalized_dataset = self._parse_columns(normalized_dataset)

        hyperparams_class = denormalize.DenormalizePrimitive.metadata.get_hyperparams()

        primitive = denormalize.DenormalizePrimitive(
            hyperparams=hyperparams_class.defaults(),
        )

        normalized_dataset = primitive.produce(inputs=normalized_dataset).value

        # To make metadata match in recorded structural types.
        normalized_dataset.metadata = normalized_dataset.metadata.generate(normalized_dataset)

        self.assertEqual(metadata_before, dataset.metadata.to_internal_json_structure())

        return normalized_dataset

    def _get_dataset_2(self):
        dataset_doc_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'graph_dataset_2', 'datasetDoc.json')
        )

        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        # We set semantic types like runtime would.
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 4), 'https://metadata.datadrivendiscovery.org/types/Target')
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 4), 'https://metadata.datadrivendiscovery.org/types/TrueTarget')
        dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 4), 'https://metadata.datadrivendiscovery.org/types/Attribute')

        metadata_before = dataset.metadata.to_internal_json_structure()

        normalized_dataset = self._normalize_column_references(dataset)

        hyperparams_class = normalize_graphs.NormalizeGraphsPrimitive.metadata.get_hyperparams()

        primitive = normalize_graphs.NormalizeGraphsPrimitive(
            hyperparams=hyperparams_class.defaults(),
        )

        normalized_dataset = primitive.produce(inputs=normalized_dataset).value

        normalized_dataset = self._parse_columns(normalized_dataset)

        # To make metadata match in recorded structural types.
        normalized_dataset.metadata = normalized_dataset.metadata.generate(normalized_dataset)

        self.assertEqual(metadata_before, dataset.metadata.to_internal_json_structure())

        return normalized_dataset

    def test_basic(self):
        self.maxDiff = None

        dataset_1 = self._get_dataset_1()
        dataset_2 = self._get_dataset_2()

        # Making some changes to make resulting datasets the same.
        dataset_2['G1_edges'] = dataset_2['edgeList']
        del dataset_2['edgeList']

        dataset_2.metadata = dataset_2.metadata.copy_to(dataset_2.metadata, ('edgeList',), ('G1_edges',))
        dataset_2.metadata = dataset_2.metadata.remove(('edgeList',), recursive=True)

        for field in ['description', 'digest', 'id', 'location_uris', 'name']:
            dataset_1.metadata = dataset_1.metadata.update((), {field: metadata_base.NO_VALUE})
            dataset_2.metadata = dataset_2.metadata.update((), {field: metadata_base.NO_VALUE})

        dataset_1_metadata = test_utils.effective_metadata(dataset_1.metadata)
        dataset_2_metadata = test_utils.effective_metadata(dataset_2.metadata)

        # Removing an ALL_ELEMENTS selector which does not really apply to any element anymore
        # (it is overridden by more specific selectors).
        del dataset_1_metadata[3]

        self.assertEqual(dataset_1_metadata, dataset_2_metadata)

        self.assertCountEqual(dataset_1.keys(), dataset_2.keys())

        for resource_id in dataset_1.keys():
            self.assertTrue(dataset_1[resource_id].equals(dataset_2[resource_id]), resource_id)

    def test_idempotent_dataset_1(self):
        dataset = self._get_dataset_1()

        hyperparams_class = normalize_graphs.NormalizeGraphsPrimitive.metadata.get_hyperparams()

        primitive = normalize_graphs.NormalizeGraphsPrimitive(
            hyperparams=hyperparams_class.defaults(),
        )

        normalized_dataset = primitive.produce(inputs=dataset).value

        self.assertEqual(utils.to_json_structure(dataset.metadata.to_internal_simple_structure()), normalized_dataset.metadata.to_internal_json_structure())

        self.assertCountEqual(dataset.keys(), normalized_dataset.keys())

        for resource_id in dataset.keys():
            self.assertTrue(dataset[resource_id].equals(normalized_dataset[resource_id]), resource_id)

    def test_idempotent_dataset_2(self):
        dataset = self._get_dataset_2()

        hyperparams_class = normalize_graphs.NormalizeGraphsPrimitive.metadata.get_hyperparams()

        primitive = normalize_graphs.NormalizeGraphsPrimitive(
            hyperparams=hyperparams_class.defaults(),
        )

        normalized_dataset = primitive.produce(inputs=dataset).value

        self.assertEqual(dataset.metadata.to_internal_json_structure(), normalized_dataset.metadata.to_internal_json_structure())

        self.assertCountEqual(dataset.keys(), normalized_dataset.keys())

        for resource_id in dataset.keys():
            self.assertTrue(dataset[resource_id].equals(normalized_dataset[resource_id]), resource_id)


if __name__ == '__main__':
    unittest.main()
