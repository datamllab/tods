import collections
import datetime
import filecmp
import glob
import json
import os
import os.path
import shutil
import sys
import tempfile
import unittest
import uuid

import frozendict
import numpy
from sklearn import datasets

COMMON_PRIMITIVES_DIR = os.path.join(os.path.dirname(__file__), 'common-primitives')
# NOTE: This insertion should appear before any code attempting to resolve or load primitives,
# so the git submodule version of `common-primitives` is looked at first.
sys.path.insert(0, COMMON_PRIMITIVES_DIR)

from common_primitives.column_parser import ColumnParserPrimitive
from common_primitives.dataset_to_dataframe import DatasetToDataFramePrimitive

from d3m import container, exceptions, utils
from d3m.container import dataset
from d3m.metadata import base as metadata_base, pipeline_run


def convert_metadata(metadata):
    return json.loads(json.dumps(metadata, cls=utils.JsonEncoder))


def make_regular_dict_and_list(obj):
    if isinstance(obj, (collections.OrderedDict, frozendict.FrozenOrderedDict, frozendict.frozendict)):
        obj = dict(obj)
    if isinstance(obj, tuple):
        obj = list(obj)

    if isinstance(obj, list):
        obj = [make_regular_dict_and_list(o) for o in obj]

    if isinstance(obj, dict):
        obj = {k: make_regular_dict_and_list(v) for k, v in obj.items()}

    return obj


def _normalize_dataset_description(dataset_description, dataset_name=None):
    for key in ('digest', 'datasetVersion', 'datasetSchemaVersion', 'redacted'):
        dataset_description['about'].pop(key, None)

    for i, r in enumerate(dataset_description.get('dataResources', [])):
        for j, c in enumerate(r.get('columns', [])):
            if 'attribute' in c['role'] and len(c['role']) > 1:
                k = c['role'].index('attribute')
                c['role'].pop(k)
                dataset_description['dataResources'][i]['columns'][j] = c

    if dataset_name == 'audio_dataset_1':
        del dataset_description['dataResources'][1]['columns'][2]['refersTo']
        del dataset_description['dataResources'][1]['columns'][3]['refersTo']

    if dataset_name == 'dataset_TEST':
        dataset_description['about']['datasetID'] = 'object_dataset_1_TEST'

    dataset_description.pop('qualities', None)

    return dataset_description


class TestDataset(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_d3m(self):
        dataset_doc_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'iris_dataset_1', 'datasetDoc.json')
        )

        ds = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        self._test_d3m(ds, dataset_doc_path)

        pipeline_run.validate_dataset(ds.to_json_structure(canonical=True))
        metadata_base.CONTAINER_SCHEMA_VALIDATOR.validate(ds.to_json_structure(canonical=True))

    def _test_d3m(self, ds, dataset_doc_path):
        ds.metadata.check(ds)

        for row in ds['learningData']:
            for cell in row:
                # Nothing should be parsed from a string.
                self.assertIsInstance(cell, str, dataset_doc_path)

        self.assertEqual(len(ds['learningData']), 150, dataset_doc_path)
        self.assertEqual(len(ds['learningData'].iloc[0]), 6, dataset_doc_path)

        self.maxDiff = None

        self.assertEqual(
            convert_metadata(ds.metadata.query(())),
            {
                'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
                'structural_type': 'd3m.container.dataset.Dataset',
                'id': 'iris_dataset_1',
                'name': 'Iris Dataset',
                'location_uris': ['file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path)],
                'source': {'license': 'CC', 'redacted': False, 'human_subjects_research': False},
                'dimension': {
                    'length': 1,
                    'name': 'resources',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/DatasetResource'],
                },
                'digest': '6191a49372f185f530920ffa35a3c4a78034ec47247aa23474537c449d37323b',
                'version': '4.0.0',
            },
            dataset_doc_path,
        )

        self.assertEqual(
            convert_metadata(ds.metadata.query(('learningData',))),
            {
                'structural_type': 'd3m.container.pandas.DataFrame',
                'semantic_types': [
                    'https://metadata.datadrivendiscovery.org/types/Table',
                    'https://metadata.datadrivendiscovery.org/types/DatasetEntryPoint',
                ],
                'dimension': {
                    'name': 'rows',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                    'length': 150,
                },
            },
            dataset_doc_path,
        )

        self.assertEqual(
            convert_metadata(ds.metadata.query(('learningData', metadata_base.ALL_ELEMENTS))),
            {
                'dimension': {
                    'name': 'columns',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                    'length': 6,
                }
            },
            dataset_doc_path,
        )

        self.assertEqual(
            convert_metadata(ds.metadata.query(('learningData', metadata_base.ALL_ELEMENTS, 0))),
            {
                'name': 'd3mIndex',
                'structural_type': 'str',
                'semantic_types': [
                    'http://schema.org/Integer',
                    'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                ],
            },
            dataset_doc_path,
        )

        for i in range(1, 5):
            self.assertEqual(
                convert_metadata(ds.metadata.query(('learningData', metadata_base.ALL_ELEMENTS, i))),
                {
                    'name': ['sepalLength', 'sepalWidth', 'petalLength', 'petalWidth'][i - 1],
                    'structural_type': 'str',
                    'semantic_types': [
                        'http://schema.org/Float',
                        'https://metadata.datadrivendiscovery.org/types/Attribute',
                    ],
                },
                dataset_doc_path,
            )

        self.assertEqual(
            convert_metadata(ds.metadata.query(('learningData', metadata_base.ALL_ELEMENTS, 5))),
            {
                'name': 'species',
                'structural_type': 'str',
                'semantic_types': [
                    'https://metadata.datadrivendiscovery.org/types/CategoricalData',
                    'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',
                    'https://metadata.datadrivendiscovery.org/types/Attribute',
                ],
            },
            dataset_doc_path,
        )

    def test_d3m_lazy(self):
        dataset_doc_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'iris_dataset_1', 'datasetDoc.json')
        )

        ds = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path), lazy=True)

        ds.metadata.check(ds)

        self.assertTrue(len(ds) == 0)
        self.assertTrue(ds.is_lazy())

        self.maxDiff = None

        self.assertEqual(
            convert_metadata(ds.metadata.query(())),
            {
                'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
                'structural_type': 'd3m.container.dataset.Dataset',
                'id': 'iris_dataset_1',
                'name': 'Iris Dataset',
                'location_uris': ['file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path)],
                'source': {'license': 'CC', 'redacted': False, 'human_subjects_research': False},
                'dimension': {
                    'length': 0,
                    'name': 'resources',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/DatasetResource'],
                },
                'digest': '6191a49372f185f530920ffa35a3c4a78034ec47247aa23474537c449d37323b',
                'version': '4.0.0',
            },
        )

        self.assertEqual(convert_metadata(ds.metadata.query(('learningData',))), {})
        self.assertEqual(convert_metadata(ds.metadata.query(('learningData', metadata_base.ALL_ELEMENTS))), {})
        self.assertEqual(convert_metadata(ds.metadata.query(('learningData', metadata_base.ALL_ELEMENTS, 0))), {})

        ds.load_lazy()

        self.assertFalse(ds.is_lazy())

        self._test_d3m(ds, dataset_doc_path)

    def test_d3m_minimal_metadata(self):
        dataset_doc_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'iris_dataset_3', 'datasetDoc.json')
        )

        ds = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        self._test_d3m_minimal_metadata(ds, dataset_doc_path)

        pipeline_run.validate_dataset(ds.to_json_structure(canonical=True))
        metadata_base.CONTAINER_SCHEMA_VALIDATOR.validate(ds.to_json_structure(canonical=True))

    def _test_d3m_minimal_metadata(self, ds, dataset_doc_path):
        ds.metadata.check(ds)

        for row in ds['learningData']:
            for cell in row:
                # Nothing should be parsed from a string.
                self.assertIsInstance(cell, str, dataset_doc_path)

        self.assertEqual(len(ds['learningData']), 150, dataset_doc_path)
        self.assertEqual(len(ds['learningData'].iloc[0]), 6, dataset_doc_path)

        self.maxDiff = None

        self.assertEqual(
            convert_metadata(ds.metadata.query(())),
            {
                'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
                'structural_type': 'd3m.container.dataset.Dataset',
                'id': 'iris_dataset_3',
                'name': 'Iris Dataset with minimal metadata',
                'location_uris': ['file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path)],
                'source': {'license': 'CC', 'redacted': False, 'human_subjects_research': False},
                'dimension': {
                    'length': 1,
                    'name': 'resources',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/DatasetResource'],
                },
                'digest': '4a0b43c5e5a76919b42b2066015ba0962512beb8600919dfffa4e2ad604e446d',
                'version': '4.0.0',
            },
            dataset_doc_path,
        )

        self.assertEqual(
            convert_metadata(ds.metadata.query(('learningData',))),
            {
                'structural_type': 'd3m.container.pandas.DataFrame',
                'semantic_types': [
                    'https://metadata.datadrivendiscovery.org/types/Table',
                    'https://metadata.datadrivendiscovery.org/types/DatasetEntryPoint',
                ],
                'dimension': {
                    'name': 'rows',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                    'length': 150,
                },
            },
            dataset_doc_path,
        )

        self.assertEqual(
            convert_metadata(ds.metadata.query(('learningData', metadata_base.ALL_ELEMENTS))),
            {
                'dimension': {
                    'name': 'columns',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                    'length': 6,
                }
            },
            dataset_doc_path,
        )

        self.assertEqual(
            convert_metadata(ds.metadata.query(('learningData', metadata_base.ALL_ELEMENTS, 0))),
            {
                'name': 'd3mIndex',
                'structural_type': 'str',
                'semantic_types': [
                    'http://schema.org/Integer',
                    'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                ],
            },
            dataset_doc_path,
        )

        for i in range(1, 6):
            self.assertEqual(
                convert_metadata(ds.metadata.query(('learningData', metadata_base.ALL_ELEMENTS, i))),
                {
                    'name': ['sepalLength', 'sepalWidth', 'petalLength', 'petalWidth', 'species'][i - 1],
                    'structural_type': 'str',
                    'semantic_types': [
                        'https://metadata.datadrivendiscovery.org/types/UnknownType',
                    ],
                },
                dataset_doc_path,
            )

    def test_d3m_saver(self):
        at_least_one = False

        for dirpath, dirnames, filenames in os.walk(
            os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'datasets'))
        ):
            if 'datasetDoc.json' in filenames:
                # Do not traverse further (to not parse "datasetDoc.json" or "problemDoc.json" if they
                # exists in raw data filename).
                dirnames[:] = []

                dataset_path = os.path.join(os.path.abspath(dirpath), 'datasetDoc.json')
                dataset_name = dataset_path.split(os.path.sep)[-2]

                # We skip "graph_dataset_1" because saving changes GML file to a file collection.
                # We skip "iris_dataset_2" and "iris_dataset_3" because when loading we add additional metadata.
                if 'graph_dataset_1' not in dataset_path and 'iris_dataset_3' not in dataset_path and 'iris_dataset_2' not in dataset_path:
                    self._test_d3m_saver(dataset_path, dataset_name)
                    self._test_d3m_saver_digest(dataset_path, dataset_name)
                at_least_one = True

        self.assertTrue(at_least_one)

    def test_d3m_saver_update_column_description(self):
        source_dataset_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'audio_dataset_1')
        )
        source_dataset_uri = 'file://{dataset_path}'.format(
            dataset_path=os.path.join(source_dataset_path, 'datasetDoc.json')
        )

        output_dataset_path = os.path.join(self.test_dir, 'audio_dataset_1')
        output_dataset_uri = 'file://{dataset_path}'.format(
            dataset_path=os.path.join(output_dataset_path, 'datasetDoc.json')
        )

        selector = ('learningData', metadata_base.ALL_ELEMENTS, 0)
        new_metadata = {'description': 'Audio files'}
        ds = container.Dataset.load(source_dataset_uri)
        ds.metadata = ds.metadata.update(selector, new_metadata)
        ds.save(output_dataset_uri)
        ds2 = container.Dataset.load(output_dataset_uri)

        self.assertEqual(convert_metadata(ds.metadata.query(selector)), convert_metadata(ds2.metadata.query(selector)))

    def test_d3m_saver_file_columns(self):
        source_dataset_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'multivariate_dataset_1')
        )
        source_dataset_uri = 'file://{dataset_path}'.format(
            dataset_path=os.path.join(source_dataset_path, 'datasetDoc.json')
        )

        output_dataset_path = os.path.join(self.test_dir, 'multivariate_dataset_1')
        output_dataset_uri = 'file://{dataset_path}'.format(
            dataset_path=os.path.join(output_dataset_path, 'datasetDoc.json')
        )

        ds = container.Dataset.load(source_dataset_uri)
        ds.save(output_dataset_uri)

        with open(os.path.join(source_dataset_path, 'datasetDoc.json'), 'r') as f:
            source_dataset_description = _normalize_dataset_description(json.load(f))

        with open(os.path.join(output_dataset_path, 'datasetDoc.json'), 'r') as f:
            output_dataset_description = _normalize_dataset_description(json.load(f))

        self.assertEqual(source_dataset_description, output_dataset_description)

        source_files = [
            x
            for x in glob.iglob(os.path.join(source_dataset_path, '**'), recursive=True)
            if os.path.isfile(x) and os.path.basename(x) != 'datasetDoc.json'
        ]
        output_files = [
            x
            for x in glob.iglob(os.path.join(output_dataset_path, '**'), recursive=True)
            if os.path.isfile(x) and os.path.basename(x) != 'datasetDoc.json'
        ]

        for x, y in zip(source_files, output_files):
            self.assertTrue(filecmp.cmp(x, y, shallow=False), (x, y))

        source_relative_filepaths = [os.path.relpath(x, source_dataset_path) for x in source_files]
        output_relative_filepaths = [os.path.relpath(x, output_dataset_path) for x in output_files]
        self.assertEqual(source_relative_filepaths, output_relative_filepaths)

    def test_load_sklearn_save_d3m(self):
        self.maxDiff = None

        for dataset_path in ['boston', 'breast_cancer', 'diabetes', 'digits', 'iris', 'linnerud']:
            source_dataset_uri = 'sklearn://{dataset_path}'.format(dataset_path=dataset_path)
            output_dateset_doc_path = os.path.join(self.test_dir, 'sklearn', dataset_path, 'datasetDoc.json')
            output_dateset_doc_uri = 'file://{output_dateset_doc_path}'.format(
                output_dateset_doc_path=output_dateset_doc_path
            )

            sklearn_dataset = container.Dataset.load(source_dataset_uri)
            sklearn_dataset.save(output_dateset_doc_uri)

            self.assertTrue(os.path.exists(output_dateset_doc_path))

            d3m_dataset = container.Dataset.load(output_dateset_doc_uri)

            sklearn_metadata = make_regular_dict_and_list(sklearn_dataset.metadata.to_internal_simple_structure())
            d3m_metadata = make_regular_dict_and_list(d3m_dataset.metadata.to_internal_simple_structure())

            del sklearn_metadata[0]['metadata']['digest']
            del d3m_metadata[0]['metadata']['digest']
            del sklearn_metadata[0]['metadata']['location_uris']
            del d3m_metadata[0]['metadata']['location_uris']

            # When saving, we convert all columns to string type.
            for metadata_index in range(3, len(sklearn_metadata)):
                sklearn_metadata[metadata_index]['metadata']['structural_type'] = str

            # Additional metadata added when saving.
            sklearn_metadata.insert(
                3,
                {
                    'metadata': {'structural_type': str},
                    'selector': ['learningData', metadata_base.ALL_ELEMENTS, metadata_base.ALL_ELEMENTS],
                },
            )

            self.assertEqual(sklearn_metadata, d3m_metadata)

    def test_load_csv_save_d3m(self):
        source_csv_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'iris_dataset_1', 'tables', 'learningData.csv')
        )
        output_csv_path = os.path.join(
            self.test_dir, 'load_csv_save_d3m', 'iris_dataset_1', 'tables', 'learningData.csv'
        )
        source_csv_uri = 'file://{source_csv_path}'.format(source_csv_path=source_csv_path)

        output_dateset_doc_path = os.path.join(self.test_dir, 'load_csv_save_d3m', 'iris_dataset_1', 'datasetDoc.json')
        output_dateset_doc_uri = 'file://{output_dateset_doc_path}'.format(
            output_dateset_doc_path=output_dateset_doc_path
        )

        csv_dataset = container.Dataset.load(source_csv_uri)
        csv_dataset.save(output_dateset_doc_uri)

        self.assertTrue(os.path.exists(output_dateset_doc_path))
        self.assertTrue(os.path.exists(output_csv_path))
        self.assertTrue(filecmp.cmp(source_csv_path, output_csv_path))

        d3m_dataset = container.Dataset.load(output_dateset_doc_uri)

        csv_metadata = make_regular_dict_and_list(csv_dataset.metadata.to_internal_simple_structure())
        d3m_metadata = make_regular_dict_and_list(d3m_dataset.metadata.to_internal_simple_structure())

        del csv_metadata[0]['metadata']['digest']
        del d3m_metadata[0]['metadata']['digest']
        del csv_metadata[0]['metadata']['location_uris']
        del d3m_metadata[0]['metadata']['location_uris']
        del d3m_metadata[0]['metadata']['approximate_stored_size']

        # Additional metadata added when saving.
        csv_metadata.insert(
            3,
            {
                'metadata': {'structural_type': str},
                'selector': ['learningData', metadata_base.ALL_ELEMENTS, metadata_base.ALL_ELEMENTS],
            },
        )

        self.assertEqual(csv_metadata, d3m_metadata)

    def _test_d3m_saver(self, dataset_path, dataset_name):
        self.maxDiff = None

        try:
            input_dataset_doc_path = dataset_path
            output_dateset_doc_path = os.path.join(self.test_dir, dataset_name, 'datasetDoc.json')

            with open(input_dataset_doc_path, 'r', encoding='utf8') as f:
                input_dataset_description = json.load(f)

            ds = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=input_dataset_doc_path))
            ds.save('file://{dataset_doc_path}'.format(dataset_doc_path=output_dateset_doc_path))

            with open(output_dateset_doc_path) as f:
                output_dataset_description = json.load(f)

            input_dataset_description = _normalize_dataset_description(input_dataset_description)
            output_dataset_description = _normalize_dataset_description(output_dataset_description, dataset_name)

            self.assertDictEqual(input_dataset_description, output_dataset_description, dataset_name)

            source_files = [
                x for x in glob.iglob(os.path.join(os.path.dirname(input_dataset_doc_path), '**'), recursive=True)
            ]
            output_files = [
                x for x in glob.iglob(os.path.join(os.path.dirname(output_dateset_doc_path), '**'), recursive=True)
            ]

            source_relative_filepaths = [
                os.path.relpath(x, os.path.dirname(input_dataset_doc_path)) for x in source_files
            ]
            output_relative_filepaths = [
                os.path.relpath(x, os.path.dirname(output_dateset_doc_path)) for x in output_files
            ]

            self.assertEqual(source_relative_filepaths, output_relative_filepaths, dataset_name)

            source_files = [x for x in source_files if (x != input_dataset_doc_path) and os.path.isfile(x)]
            output_files = [x for x in output_files if (x != output_dateset_doc_path) and os.path.isfile(x)]

            for x, y in zip(source_files, output_files):
                if dataset_name == 'dataset_TEST' and os.path.basename(x) == 'learningData.csv':
                    continue

                self.assertTrue(filecmp.cmp(x, y, shallow=False), (dataset_name, x, y))

        finally:
            shutil.rmtree(os.path.join(self.test_dir, dataset_name), ignore_errors=True)

    def _test_d3m_saver_digest(self, dataset_path, dataset_name):
        self.maxDiff = None

        try:
            # Load original dataset and store it's digest
            original_dataset_uri = 'file://{dataset_path}'.format(dataset_path=dataset_path)
            original_dataset = container.Dataset.load(original_dataset_uri)
            original_dateset_digest = original_dataset.metadata.query(())['digest']

            # Save the dataset to a new location
            output_dataset_path = os.path.join(self.test_dir, dataset_name)
            output_dataset_uri = 'file://{dataset_path}'.format(
                dataset_path=os.path.join(output_dataset_path, 'datasetDoc.json')
            )
            original_dataset.save(output_dataset_uri)

            # Load the dataset from the new location and store the digest
            output_dataset = container.Dataset.load(output_dataset_uri)
            output_dataset_digest = output_dataset.metadata.query(())['digest']

            # Remove digest from the in-memory dataset and store the dataset to a new location
            output_dataset.metadata = output_dataset.metadata.update((), {'digest': metadata_base.NO_VALUE})
            new_output_dataset_path = os.path.join(self.test_dir, dataset_name + '_new')
            output_dataset_uri = 'file://{dataset_path}'.format(
                dataset_path=os.path.join(new_output_dataset_path, 'datasetDoc.json')
            )
            output_dataset.save(output_dataset_uri)

            # Load digest from the stored datasetDoc.json
            with open(os.path.join(new_output_dataset_path, 'datasetDoc.json'), 'r') as f:
                saved_digest = json.load(f)['about']['digest']

            # Calculate dataset digest with the reference function
            reference_dataset_digest = dataset.get_d3m_dataset_digest(
                os.path.join(output_dataset_path, 'datasetDoc.json')
            )

            self.assertEqual(output_dataset_digest, saved_digest)
            self.assertEqual(output_dataset_digest, reference_dataset_digest)

        finally:
            shutil.rmtree(os.path.join(self.test_dir, dataset_name), ignore_errors=True)
            shutil.rmtree(os.path.join(self.test_dir, dataset_name + '_new'), ignore_errors=True)

    def test_d3m_preserve_edge_list_resource_type(self):
        source_dataset_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'graph_dataset_2')
        )
        source_dataset_uri = 'file://{dataset_path}'.format(
            dataset_path=os.path.join(source_dataset_path, 'datasetDoc.json')
        )

        output_dataset_path = os.path.join(self.test_dir, 'graph_dataset_2')
        output_dataset_uri = 'file://{dataset_path}'.format(
            dataset_path=os.path.join(output_dataset_path, 'datasetDoc.json')
        )

        ds_1 = container.Dataset.load(source_dataset_uri)
        ds_1.save(output_dataset_uri)
        ds_2 = container.Dataset.load(output_dataset_uri)

        selector = ('edgeList',)

        self.assertEqual(
            convert_metadata(ds_1.metadata.query(selector)), convert_metadata(ds_2.metadata.query(selector))
        )

    def test_d3m_saver_qualities(self):
        self.maxDiff = None

        source_dataset_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'audio_dataset_1')
        )
        source_dataset_uri = 'file://{dataset_path}'.format(
            dataset_path=os.path.join(source_dataset_path, 'datasetDoc.json')
        )
        output_dataset_path = os.path.join(self.test_dir, 'audio_dataset_1')
        output_dataset_uri = 'file://{dataset_path}'.format(
            dataset_path=os.path.join(output_dataset_path, 'datasetDoc.json')
        )

        ds = container.Dataset.load(source_dataset_uri)
        # Insert a non-standard dataset value to test quality saving / loading.
        ds.metadata = ds.metadata.update((), {'additional_quality': 'some value'})
        ds.save(output_dataset_uri)
        ds2 = container.Dataset.load(output_dataset_uri)

        ds.metadata = ds.metadata.update((), {'location_uris': ''})
        ds.metadata = ds.metadata.update((), {'digest': ''})
        ds.metadata = ds.metadata.update(('0', metadata_base.ALL_ELEMENTS, 0), {'location_base_uris': ''})

        ds2.metadata = ds2.metadata.update((), {'location_uris': ''})
        ds2.metadata = ds2.metadata.update((), {'digest': ''})
        ds2.metadata = ds2.metadata.update(('0', metadata_base.ALL_ELEMENTS, 0), {'location_base_uris': ''})

        ds_metadata = make_regular_dict_and_list(ds.metadata.to_internal_simple_structure())
        ds2_metadata = make_regular_dict_and_list(ds2.metadata.to_internal_simple_structure())

        # Additional metadata added when saving.
        ds_metadata.insert(
            3,
            {
                'metadata': {'structural_type': str},
                'selector': ['0', metadata_base.ALL_ELEMENTS, metadata_base.ALL_ELEMENTS],
            },
        )
        ds_metadata.insert(
            7,
            {
                'metadata': {'structural_type': str},
                'selector': ['learningData', metadata_base.ALL_ELEMENTS, metadata_base.ALL_ELEMENTS],
            },
        )

        self.assertEqual(ds_metadata, ds2_metadata)

        ds2 = container.Dataset.load(output_dataset_uri, lazy=True)

        ds2.metadata = ds2.metadata.update((), {'location_uris': ''})
        ds2.metadata = ds2.metadata.update((), {'digest': ''})
        ds2.metadata = ds2.metadata.update(('0', metadata_base.ALL_ELEMENTS, 0), {'location_base_uris': ''})

        ds_metadata = make_regular_dict_and_list(ds.metadata.query(()))
        ds2_metadata = make_regular_dict_and_list(ds2.metadata.query(()))

        ds_metadata['dimension']['length'] = 0

        self.assertEqual(ds_metadata, ds2_metadata)

    def test_d3m_saver_synthetic_dataset(self):
        dataset_path = os.path.abspath(os.path.join(self.test_dir, 'synthetic_dataset_1', 'datasetDoc.json'))
        dataset_uri = 'file://{dataset_path}'.format(dataset_path=dataset_path)

        df = container.DataFrame([[0]], columns=['col_1'], generate_metadata=False)
        ds = container.Dataset(resources={'someData': df}, generate_metadata=True)

        ds.metadata = ds.metadata.update(
            (),
            {
                'custom_metadata_1': 'foo',
                'custom_metadata_2': datetime.datetime(2019, 6, 6),
                'deleted_metadata': metadata_base.NO_VALUE,
            },
        )

        with self.assertRaises(exceptions.InvalidMetadataError):
            ds.save(dataset_uri)

        ds.metadata = ds.metadata.update((), {'id': 'synthetic_dataset_1', 'name': 'Synthetic dataset 1'})

        ds.save(dataset_uri)
        ds2 = container.Dataset.load(dataset_uri)

        self.assertEqual(
            make_regular_dict_and_list(ds2.metadata.to_internal_simple_structure()),
            [
                {
                    'selector': [],
                    'metadata': {
                        'custom_metadata_1': 'foo',
                        'custom_metadata_2': datetime.datetime(2019, 6, 6),
                        'deleted_metadata': metadata_base.NO_VALUE,
                        'digest': 'bc41e654599e31169061ce5f6b99133e6220eea2a83c53f55c653e4d9a4b67e2',
                        'dimension': {
                            'length': 1,
                            'name': 'resources',
                            'semantic_types': ['https://metadata.datadrivendiscovery.org/types/DatasetResource'],
                        },
                        'id': 'synthetic_dataset_1',
                        'location_uris': [dataset_uri],
                        'name': 'Synthetic dataset 1',
                        'schema': 'https://metadata.datadrivendiscovery.org/schemas/v0/container.json',
                        'structural_type': container.Dataset,
                    },
                },
                {
                    'selector': ['someData'],
                    'metadata': {
                        'dimension': {
                            'length': 1,
                            'name': 'rows',
                            'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                        },
                        'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Table'],
                        'structural_type': container.DataFrame,
                    },
                },
                {
                    'selector': ['someData', metadata_base.ALL_ELEMENTS],
                    'metadata': {
                        'dimension': {
                            'length': 1,
                            'name': 'columns',
                            'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                        }
                    },
                },
                {
                    'selector': ['someData', metadata_base.ALL_ELEMENTS, metadata_base.ALL_ELEMENTS],
                    'metadata': {'structural_type': str},
                },
                {
                    'selector': ['someData', metadata_base.ALL_ELEMENTS, 0],
                    'metadata': {
                        'name': 'col_1',
                        'semantic_types': ['http://schema.org/Integer'],
                        'structural_type': str,
                    },
                },
            ],
        )

    def test_d3m_saver_synthetic_dataset_2(self):
        self.maxDiff = None

        dataset_path = os.path.abspath(os.path.join(self.test_dir, 'synthetic_dataset_2', 'datasetDoc.json'))
        dataset_uri = 'file://{dataset_path}'.format(dataset_path=dataset_path)

        df = container.DataFrame({'col_1': [0], 'col_2': [0.0]}, generate_metadata=True)
        synthetic_dataset = container.Dataset(resources={'learningData': df}, generate_metadata=True)

        with self.assertRaises(exceptions.InvalidMetadataError):
            synthetic_dataset.save(dataset_uri)

        synthetic_dataset.metadata = synthetic_dataset.metadata.update(
            (), {'id': 'synthetic_dataset_2', 'name': 'Synthetic dataset 2'}
        )

        synthetic_dataset.save(dataset_uri)

        loaded_dataset = container.Dataset.load(dataset_uri)

        hyperparams_class = DatasetToDataFramePrimitive.metadata.get_hyperparams()
        primitive = DatasetToDataFramePrimitive(hyperparams=hyperparams_class.defaults())
        loaded_dataframe = primitive.produce(inputs=loaded_dataset).value

        hyperparams_class = ColumnParserPrimitive.metadata.get_hyperparams()
        primitive = ColumnParserPrimitive(hyperparams=hyperparams_class.defaults())
        loaded_dataframe = primitive.produce(inputs=loaded_dataframe).value

        self.assertEqual(
            make_regular_dict_and_list(loaded_dataframe.metadata.to_internal_simple_structure()),
            [
                {
                    'selector': [],
                    'metadata': {
                        'dimension': {
                            'length': 1,
                            'name': 'rows',
                            'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                        },
                        'schema': 'https://metadata.datadrivendiscovery.org/schemas/v0/container.json',
                        'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Table'],
                        'structural_type': container.DataFrame,
                    },
                },
                {
                    'selector': [metadata_base.ALL_ELEMENTS],
                    'metadata': {
                        'dimension': {
                            'length': 2,
                            'name': 'columns',
                            'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                        }
                    },
                },
                {
                    'selector': [metadata_base.ALL_ELEMENTS, metadata_base.ALL_ELEMENTS],
                    'metadata': {'structural_type': str},
                },
                {
                    'selector': [metadata_base.ALL_ELEMENTS, 0],
                    'metadata': {
                        'name': 'col_1',
                        'semantic_types': ['http://schema.org/Integer'],
                        'structural_type': int,
                    },
                },
                {
                    'selector': [metadata_base.ALL_ELEMENTS, 1],
                    'metadata': {
                        'name': 'col_2',
                        'semantic_types': ['http://schema.org/Float'],
                        'structural_type': float,
                    },
                },
            ],
        )

    def test_d3m_saver_unknown_type(self):
        metadata = metadata_base.DataMetadata()

        metadata = metadata.update(
            (),
            {
                'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
                'structural_type': container.DataFrame,
                'id': 'multi_source_1',
                'version': '1.0',
                'name': 'A multi source dataset',
                'source': {'license': 'CC0', 'redacted': False, 'human_subjects_research': False},
                'dimension': {
                    'length': 1,
                    'name': 'resources',
                    'semantic_types': [
                        'https://metadata.datadrivendiscovery.org/types/DatasetResource',
                        'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',
                        'https://metadata.datadrivendiscovery.org/types/Attribute',
                    ],
                },
            },
        )

        metadata = metadata.update(
            ('learningData',),
            {
                'structural_type': container.DataFrame,
                'semantic_types': [
                    'https://metadata.datadrivendiscovery.org/types/Table',
                    'https://metadata.datadrivendiscovery.org/types/DatasetEntryPoint',
                ],
                'dimension': {
                    'name': 'rows',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                    'length': 3,
                },
            },
        )

        metadata = metadata.update(
            ('learningData', metadata_base.ALL_ELEMENTS),
            {
                'dimension': {
                    'name': 'columns',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                    'length': 3,
                }
            },
        )

        metadata = metadata.update(
            ('learningData', metadata_base.ALL_ELEMENTS, 0),
            {
                'name': 'd3mIndex',
                'structural_type': str,
                'semantic_types': [
                    'http://schema.org/Integer',
                    'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                ],
            },
        )

        metadata = metadata.update(
            ('learningData', metadata_base.ALL_ELEMENTS, 1),
            {
                'name': 'sepalLength',
                'structural_type': str,
                'semantic_types': [
                    'https://metadata.datadrivendiscovery.org/types/UnknownType',
                    'https://metadata.datadrivendiscovery.org/types/Attribute',
                ],
            },
        )

        metadata = metadata.update(
            ('learningData', metadata_base.ALL_ELEMENTS, 2),
            {
                'name': 'species',
                'structural_type': str,
                'semantic_types': [
                    'http://schema.org/Float',
                    'https://metadata.datadrivendiscovery.org/types/UnknownType',
                    'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',
                ],
            },
        )

        dataset_path = os.path.abspath(os.path.join(self.test_dir, 'unknown_columns_1', 'datasetDoc.json'))
        dataset_uri = 'file://{dataset_path}'.format(dataset_path=dataset_path)

        df = container.DataFrame([[0, 0.1, 'Iris-setosa']], columns=['d3mIndex', 'sepalLength', 'species'])
        ds = container.Dataset(resources={'learningData': df}, metadata=metadata)

        with self.assertRaises(exceptions.InvalidMetadataError):
            ds.save(dataset_uri)

        metadata = metadata.update(
            ('learningData', metadata_base.ALL_ELEMENTS, 2),
            {
                'name': 'species',
                'structural_type': str,
                'semantic_types': [
                    'https://metadata.datadrivendiscovery.org/types/UnknownType',
                    'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',
                ],
            },
        )

        ds = container.Dataset(resources={'learningData': df}, metadata=metadata)
        ds.save(dataset_uri)

        ds = container.Dataset.load(dataset_uri)

        self.assertEqual(
            convert_metadata(ds.metadata.query(('learningData', metadata_base.ALL_ELEMENTS, 0))),
            {
                'name': 'd3mIndex',
                'structural_type': 'str',
                'semantic_types': [
                    'http://schema.org/Integer',
                    'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                ],
            },
        )

        self.assertEqual(
            convert_metadata(ds.metadata.query(('learningData', metadata_base.ALL_ELEMENTS, 1))),
            {
                'name': 'sepalLength',
                'structural_type': 'str',
                'semantic_types': [
                    'https://metadata.datadrivendiscovery.org/types/UnknownType',
                    'https://metadata.datadrivendiscovery.org/types/Attribute',
                ],
            },
        )

        self.assertEqual(
            convert_metadata(ds.metadata.query(('learningData', metadata_base.ALL_ELEMENTS, 2))),
            {
                'name': 'species',
                'structural_type': 'str',
                'semantic_types': [
                    'https://metadata.datadrivendiscovery.org/types/UnknownType',
                    'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',
                    'https://metadata.datadrivendiscovery.org/types/Attribute',
                ],
            },
        )

    def test_d3m_saver_multi_source(self):
        shutil.copytree(
            os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'raw_dataset_1')),
            os.path.join(self.test_dir, 'raw_dataset_1'),
        )
        shutil.copytree(
            os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'image_dataset_1')),
            os.path.join(self.test_dir, 'image_dataset_1'),
        )
        shutil.copytree(
            os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'image_dataset_1')),
            os.path.join(self.test_dir, 'image_dataset_2'),
        )

        metadata = metadata_base.DataMetadata()

        metadata = metadata.update(
            (),
            {
                'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
                'structural_type': container.DataFrame,
                'id': 'multi_source_1',
                'version': '1.0',
                'name': 'A multi source dataset',
                'source': {'license': 'CC0', 'redacted': False, 'human_subjects_research': False},
                'dimension': {
                    'length': 1,
                    'name': 'resources',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/DatasetResource'],
                },
            },
        )

        metadata = metadata.update(
            ('learningData',),
            {
                'structural_type': container.DataFrame,
                'semantic_types': [
                    'https://metadata.datadrivendiscovery.org/types/Table',
                    'https://metadata.datadrivendiscovery.org/types/FilesCollection',
                ],
                'dimension': {
                    'name': 'rows',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                    'length': 3,
                },
            },
        )

        metadata = metadata.update(
            ('learningData', metadata_base.ALL_ELEMENTS),
            {
                'dimension': {
                    'name': 'columns',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                    'length': 1,
                }
            },
        )

        metadata = metadata.update(
            ('learningData', metadata_base.ALL_ELEMENTS, 0),
            {
                'media_types': ['image/jpeg', 'image/png', 'text/csv'],
                'name': 'filename',
                'semantic_types': [
                    'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                    'https://metadata.datadrivendiscovery.org/types/FileName',
                    'https://metadata.datadrivendiscovery.org/types/UnspecifiedStructure',
                ],
                'structural_type': str,
            },
        )

        metadata = metadata.update(
            ('learningData', 0, 0),
            {
                'location_base_uris': [
                    'file://{dataset_doc_path}'.format(
                        dataset_doc_path=os.path.join(self.test_dir, 'raw_dataset_1', 'raw') + '/'
                    )
                ],
                'media_types': ['text/csv'],
            },
        )

        metadata = metadata.update(
            ('learningData', 1, 0),
            {
                'location_base_uris': [
                    'file://{dataset_doc_path}'.format(
                        dataset_doc_path=os.path.join(self.test_dir, 'image_dataset_1', 'media') + '/'
                    )
                ],
                'media_types': ['image/png'],
            },
        )

        metadata = metadata.update(
            ('learningData', 2, 0),
            {
                'location_base_uris': [
                    'file://{dataset_doc_path}'.format(
                        dataset_doc_path=os.path.join(self.test_dir, 'image_dataset_2', 'media') + '/'
                    )
                ],
                'media_types': ['image/jpeg'],
            },
        )

        df = container.DataFrame({'filename': ['complementaryData.csv', 'cifar10_bird_1.png', '001_HandPhoto_left_01.jpg']})

        ds = container.Dataset(resources={'learningData': df}, metadata=metadata)
        data_path = os.path.abspath(os.path.join(self.test_dir, 'multi_source_1', 'datasetDoc.json'))
        ds.save('file://' + data_path)

        self.assertTrue(os.path.exists(data_path))
        with open(data_path, 'r', encoding='utf') as data_file:
            description = json.load(data_file)

        self.assertEqual(
            description['dataResources'],
            [
                {
                    'resID': 'learningData',
                    'isCollection': True,
                    'resFormat': {'image/jpeg': ['jpg'], 'image/png': ['png'], 'text/csv': ['csv']},
                    'resType': 'raw',
                    'resPath': 'files/',
                }
            ],
        )

        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'multi_source_1', 'files', 'complementaryData.csv')))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'multi_source_1', 'files', 'cifar10_bird_1.png')))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'multi_source_1', 'files', '001_HandPhoto_left_01.jpg')))

    def test_csv_with_d3m_index(self):
        dataset_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'iris_dataset_1', 'tables', 'learningData.csv')
        )

        dataset_id = '219a5e7b-4499-4160-9b72-9cfa53c4924d'
        dataset_name = 'Iris Dataset'

        ds = container.Dataset.load(
            'file://{dataset_path}'.format(dataset_path=dataset_path), dataset_id=dataset_id, dataset_name=dataset_name
        )

        self._test_csv_with_d3m_index(ds, dataset_path, dataset_id, dataset_name)

    def _test_csv_with_d3m_index(self, ds, dataset_path, dataset_id, dataset_name):
        ds.metadata.check(ds)

        for row in ds['learningData']:
            for cell in row:
                # Nothing should be parsed from a string.
                self.assertIsInstance(cell, str)

        self.assertEqual(len(ds['learningData']), 150, dataset_name)
        self.assertEqual(len(ds['learningData'].iloc[0]), 6, dataset_name)

        self.maxDiff = None

        self.assertEqual(
            convert_metadata(ds.metadata.query(())),
            {
                'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
                'structural_type': 'd3m.container.dataset.Dataset',
                'id': dataset_id,
                'name': dataset_name,
                'stored_size': 4961,
                'location_uris': ['file://localhost{dataset_path}'.format(dataset_path=dataset_path)],
                'dimension': {
                    'length': 1,
                    'name': 'resources',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/DatasetResource'],
                },
                'digest': 'a5e827f2fb60639f1eb7b9bd3b849b0db9c308ba74d0479c20aaeaad77ccda48',
            },
            dataset_name,
        )

        self.assertEqual(
            convert_metadata(ds.metadata.query(('learningData',))),
            {
                'structural_type': 'd3m.container.pandas.DataFrame',
                'semantic_types': [
                    'https://metadata.datadrivendiscovery.org/types/Table',
                    'https://metadata.datadrivendiscovery.org/types/DatasetEntryPoint',
                ],
                'dimension': {
                    'name': 'rows',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                    'length': 150,
                },
            },
            dataset_name,
        )

        self.assertEqual(
            convert_metadata(ds.metadata.query(('learningData', metadata_base.ALL_ELEMENTS))),
            {
                'dimension': {
                    'name': 'columns',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                    'length': 6,
                }
            },
            dataset_name,
        )

        self.assertEqual(
            convert_metadata(ds.metadata.query(('learningData', metadata_base.ALL_ELEMENTS, 0))),
            {
                'name': 'd3mIndex',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/UnknownType'],
                'structural_type': 'str',
            },
            dataset_name,
        )

        for i in range(1, 6):
            self.assertEqual(
                convert_metadata(ds.metadata.query(('learningData', metadata_base.ALL_ELEMENTS, i))),
                {
                    'name': ['sepalLength', 'sepalWidth', 'petalLength', 'petalWidth', 'species'][i - 1],
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/UnknownType'],
                    'structural_type': 'str',
                },
                dataset_name,
            )

    def test_csv_lazy_with_d3m_index(self):
        dataset_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'iris_dataset_1', 'tables', 'learningData.csv')
        )

        dataset_id = '219a5e7b-4499-4160-9b72-9cfa53c4924d'
        dataset_name = 'Iris Dataset'

        ds = container.Dataset.load(
            'file://{dataset_path}'.format(dataset_path=dataset_path),
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            lazy=True,
        )

        ds.metadata.check(ds)

        self.assertTrue(len(ds) == 0)
        self.assertTrue(ds.is_lazy())

        self.maxDiff = None

        self.assertEqual(
            convert_metadata(ds.metadata.query(())),
            {
                'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
                'structural_type': 'd3m.container.dataset.Dataset',
                'id': dataset_id,
                'name': dataset_name,
                'location_uris': ['file://localhost{dataset_path}'.format(dataset_path=dataset_path)],
                'dimension': {
                    'length': 0,
                    'name': 'resources',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/DatasetResource'],
                },
            },
        )

        self.assertEqual(convert_metadata(ds.metadata.query(('learningData',))), {})
        self.assertEqual(convert_metadata(ds.metadata.query(('learningData', metadata_base.ALL_ELEMENTS))), {})
        self.assertEqual(convert_metadata(ds.metadata.query(('learningData', metadata_base.ALL_ELEMENTS, 0))), {})

        ds.load_lazy()

        self.assertFalse(ds.is_lazy())

        self._test_csv_with_d3m_index(ds, dataset_path, dataset_id, dataset_name)

    def test_sklearn(self):
        for dataset_path in ['boston', 'breast_cancer', 'diabetes', 'digits', 'iris', 'linnerud']:
            container.Dataset.load('sklearn://{dataset_path}'.format(dataset_path=dataset_path))

        dataset_uri = 'sklearn://iris'
        dataset_id = str(uuid.uuid3(uuid.NAMESPACE_URL, dataset_uri))
        dataset_name = 'Iris Dataset'

        ds = container.Dataset.load(dataset_uri, dataset_id=dataset_id, dataset_name=dataset_name)

        self._test_sklearn(ds, dataset_uri)

    def _test_sklearn(self, ds, dataset_uri):
        ds.metadata.check(ds)

        self.assertEqual(len(ds['learningData']), 150, dataset_uri)
        self.assertEqual(len(ds['learningData'].iloc[0]), 6, dataset_uri)

        self.maxDiff = None

        self.assertEqual(
            convert_metadata(ds.metadata.query(())),
            {
                'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
                'structural_type': 'd3m.container.dataset.Dataset',
                'id': '44f6efaa-72e7-383e-9369-64bd7168fb26',
                'name': 'Iris Dataset',
                'location_uris': [dataset_uri],
                'description': datasets.load_iris()['DESCR'],
                'dimension': {
                    'length': 1,
                    'name': 'resources',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/DatasetResource'],
                },
                'digest': '2cd0dd490ba383fe08a9f89514f6688bb5cb77d4a7da140e9458e7c534eb82f4',
            },
            dataset_uri,
        )

        self.assertEqual(
            convert_metadata(ds.metadata.query(('learningData',))),
            {
                'structural_type': 'd3m.container.pandas.DataFrame',
                'semantic_types': [
                    'https://metadata.datadrivendiscovery.org/types/Table',
                    'https://metadata.datadrivendiscovery.org/types/DatasetEntryPoint',
                ],
                'dimension': {
                    'name': 'rows',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                    'length': 150,
                },
            },
            dataset_uri,
        )

        self.assertEqual(
            convert_metadata(ds.metadata.query(('learningData', metadata_base.ALL_ELEMENTS))),
            {
                'dimension': {
                    'name': 'columns',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                    'length': 6,
                }
            },
            dataset_uri,
        )

        self.assertEqual(
            convert_metadata(ds.metadata.query(('learningData', metadata_base.ALL_ELEMENTS, 0))),
            {
                'name': 'd3mIndex',
                'structural_type': 'numpy.int64',
                'semantic_types': [
                    'http://schema.org/Integer',
                    'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                ],
            },
            dataset_uri,
        )

        for i in range(1, 5):
            self.assertEqual(
                convert_metadata(ds.metadata.query(('learningData', metadata_base.ALL_ELEMENTS, i))),
                {
                    'name': ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'][i - 1],
                    'structural_type': 'numpy.float64',
                    'semantic_types': [
                        'https://metadata.datadrivendiscovery.org/types/UnknownType',
                        'https://metadata.datadrivendiscovery.org/types/Attribute',
                    ],
                },
                dataset_uri,
            )

        self.assertEqual(
            convert_metadata(ds.metadata.query(('learningData', metadata_base.ALL_ELEMENTS, 5))),
            {
                'name': 'column 4',
                'structural_type': 'str',
                'semantic_types': [
                    'https://metadata.datadrivendiscovery.org/types/CategoricalData',
                    'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',
                    'https://metadata.datadrivendiscovery.org/types/Attribute',
                ],
            },
            dataset_uri,
        )

    @unittest.skip("requires rewrite")
    # TODO: Fix. Currently "generate_metadata" is not called when not loading lazily.
    #       We should just always use auto generation for as much as possible.
    #       Or not, to make sure things are speedy?
    def test_sklearn_lazy(self):
        for dataset_path in ['boston', 'breast_cancer', 'diabetes', 'digits', 'iris', 'linnerud']:
            container.Dataset.load('sklearn://{dataset_path}'.format(dataset_path=dataset_path))

        dataset_uri = 'sklearn://iris'
        dataset_id = str(uuid.uuid3(uuid.NAMESPACE_URL, dataset_uri))
        dataset_name = 'Iris Dataset'

        ds = container.Dataset.load(dataset_uri, dataset_id=dataset_id, dataset_name=dataset_name, lazy=True)

        ds.metadata.check(ds)

        self.assertTrue(len(ds) == 0)
        self.assertTrue(ds.is_lazy())

        self.maxDiff = None

        self.assertEqual(
            convert_metadata(ds.metadata.query(())),
            {
                'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
                'structural_type': 'd3m.container.dataset.Dataset',
                'id': '44f6efaa-72e7-383e-9369-64bd7168fb26',
                'name': 'Iris Dataset',
                'location_uris': [dataset_uri],
                'dimension': {
                    'length': 0,
                    'name': 'resources',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/DatasetResource'],
                },
            },
        )

        self.assertEqual(convert_metadata(ds.metadata.query(('learningData',))), {})
        self.assertEqual(convert_metadata(ds.metadata.query(('learningData', metadata_base.ALL_ELEMENTS))), {})
        self.assertEqual(convert_metadata(ds.metadata.query(('learningData', metadata_base.ALL_ELEMENTS, 0))), {})

        ds.load_lazy()

        self.assertFalse(ds.is_lazy())

        self._test_sklearn(ds, dataset_uri)

    def test_multi_table(self):
        dataset_doc_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'database_dataset_1', 'datasetDoc.json')
        )

        container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

    def test_timeseries(self):
        dataset_doc_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'timeseries_dataset_1', 'datasetDoc.json')
        )

        container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

    def test_audio(self):
        dataset_doc_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'audio_dataset_1', 'datasetDoc.json')
        )

        ds = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        self.maxDiff = None

        self.assertEqual(
            convert_metadata(ds.metadata.query(())),
            {
                'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
                'structural_type': 'd3m.container.dataset.Dataset',
                'id': 'audio_dataset_1',
                'version': '4.0.0',
                'name': 'Audio dataset to be used for tests',
                'location_uris': ['file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path)],
                'source': {'license': 'CC0', 'redacted': False},
                'dimension': {
                    'length': 2,
                    'name': 'resources',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/DatasetResource'],
                },
                'digest': '4eaa4ee8ce18dc066d400d756105aab1ce92895593d09c8be23e08fdd89640e1',
            },
        )

        self.assertEqual(
            convert_metadata(ds.metadata.query(('0',))),
            {
                'structural_type': 'd3m.container.pandas.DataFrame',
                'semantic_types': [
                    'https://metadata.datadrivendiscovery.org/types/Table',
                    'https://metadata.datadrivendiscovery.org/types/FilesCollection',
                ],
                'dimension': {
                    'name': 'rows',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                    'length': 1,
                },
            },
        )

        self.assertEqual(
            convert_metadata(ds.metadata.query(('0', metadata_base.ALL_ELEMENTS))),
            {
                'dimension': {
                    'name': 'columns',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                    'length': 1,
                }
            },
        )

        self.assertEqual(
            convert_metadata(ds.metadata.query(('learningData',))),
            {
                'structural_type': 'd3m.container.pandas.DataFrame',
                'semantic_types': [
                    'https://metadata.datadrivendiscovery.org/types/Table',
                    'https://metadata.datadrivendiscovery.org/types/DatasetEntryPoint',
                ],
                'dimension': {
                    'name': 'rows',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                    'length': 1,
                },
            },
        )

        self.assertEqual(
            convert_metadata(ds.metadata.query(('learningData', metadata_base.ALL_ELEMENTS))),
            {
                'dimension': {
                    'name': 'columns',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                    'length': 5,
                }
            },
        )

        self.assertEqual(
            convert_metadata(ds.metadata.query(('learningData', metadata_base.ALL_ELEMENTS, 0))),
            {
                'name': 'd3mIndex',
                'structural_type': 'str',
                'semantic_types': [
                    'http://schema.org/Integer',
                    'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                ],
            },
        )

        self.assertEqual(
            convert_metadata(ds.metadata.query(('learningData', metadata_base.ALL_ELEMENTS, 1))),
            {
                'name': 'audio_file',
                'structural_type': 'str',
                'semantic_types': [
                    'http://schema.org/Text',
                    'https://metadata.datadrivendiscovery.org/types/Attribute',
                ],
                'foreign_key': {'type': 'COLUMN', 'resource_id': '0', 'column_index': 0},
            },
        )

        self.assertEqual(
            convert_metadata(ds.metadata.query(('learningData', metadata_base.ALL_ELEMENTS, 2))),
            {
                'name': 'start',
                'structural_type': 'str',
                'semantic_types': [
                    'http://schema.org/Float',
                    'https://metadata.datadrivendiscovery.org/types/Boundary',
                    'https://metadata.datadrivendiscovery.org/types/IntervalStart',
                ],
                'boundary_for': {'resource_id': 'learningData', 'column_index': 1},
            },
        )

        self.assertEqual(
            convert_metadata(ds.metadata.query(('learningData', metadata_base.ALL_ELEMENTS, 3))),
            {
                'name': 'end',
                'structural_type': 'str',
                'semantic_types': [
                    'http://schema.org/Float',
                    'https://metadata.datadrivendiscovery.org/types/Boundary',
                    'https://metadata.datadrivendiscovery.org/types/IntervalEnd',
                ],
                'boundary_for': {'resource_id': 'learningData', 'column_index': 1},
            },
        )

        self.assertEqual(
            convert_metadata(ds.metadata.query(('learningData', metadata_base.ALL_ELEMENTS, 4))),
            {
                'name': 'class',
                'structural_type': 'str',
                'semantic_types': [
                    'https://metadata.datadrivendiscovery.org/types/CategoricalData',
                    'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',
                    'https://metadata.datadrivendiscovery.org/types/Attribute',
                ],
            },
        )

    def test_raw(self):
        dataset_doc_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'raw_dataset_1', 'datasetDoc.json')
        )

        ds = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        self.assertEqual(
            convert_metadata(ds.metadata.query(())),
            {
                'schema': 'https://metadata.datadrivendiscovery.org/schemas/v0/container.json',
                'structural_type': 'd3m.container.dataset.Dataset',
                'id': 'raw_dataset_1',
                'name': 'Raw dataset to be used for tests',
                'location_uris': ['file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path)],
                'dimension': {
                    'name': 'resources',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/DatasetResource'],
                    'length': 1,
                },
                'digest': 'e28468d602c30c7da7643aa78840bcaae68a9abb96b48cc98eb51fb94e6fd3af',
                'source': {'redacted': False},
                'version': '4.0.0',
            },
        )
        self.assertEqual(
            convert_metadata(ds.metadata.query(('0', metadata_base.ALL_ELEMENTS, 0))),
            {
                'location_base_uris': [
                    'file://{dataset_path}/raw/'.format(dataset_path=os.path.dirname(dataset_doc_path))
                ],
                'media_types': ['text/csv'],
                'name': 'filename',
                'semantic_types': [
                    'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                    'https://metadata.datadrivendiscovery.org/types/FileName',
                    'https://metadata.datadrivendiscovery.org/types/UnspecifiedStructure',
                ],
                'structural_type': 'str',
            },
        )

    def test_select_rows(self):
        dataset_doc_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'iris_dataset_1', 'datasetDoc.json')
        )
        ds = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        # add metadata for rows 0, 1, 2
        ds.metadata = ds.metadata.update(('learningData', 0), {'a': 0})
        ds.metadata = ds.metadata.update(('learningData', 1), {'b': 1})
        ds.metadata = ds.metadata.update(('learningData', 2), {'c': 2})

        cut_dataset = ds.select_rows({'learningData': [0, 2]})

        # verify that rows are removed from dataframe and re-indexed
        self.assertListEqual([0, 1], list(cut_dataset['learningData'].index))
        self.assertListEqual(['0', '2'], list(cut_dataset['learningData'].d3mIndex))

        # verify that metadata is removed and re-indexed
        self.assertEqual(cut_dataset.metadata.query(('learningData', 0))['a'], 0)
        self.assertEqual(cut_dataset.metadata.query(('learningData', 1))['c'], 2)

    def test_score_workaround(self):
        dataset_doc_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), 'data', 'datasets', 'score_dataset_1', 'dataset_TEST', 'datasetDoc.json'
            )
        )
        ds = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        self.assertEqual(ds.metadata.query_field((), 'id'), 'object_dataset_1_SCORE')

        self.assertEqual(
            ds['learningData'].values.tolist(),
            [
                ['0', 'img_00285.png', 'red', '480,457,480,529,515,529,515,457'],
                ['0', 'img_00285.png', 'black', '10,117,10,329,105,329,105,117'],
                ['1', 'img_00225.png', 'blue', '422,540,422,660,576,660,576,540'],
                ['1', 'img_00225.png', 'red', '739,460,739,545,768,545,768,460'],
            ],
        )

    def test_csv_without_d3m_index(self):
        dataset_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'database_dataset_1', 'tables', 'values.csv')
        )

        dataset_id = '7cd469db-e922-4418-84a9-cda9517251d1'
        dataset_name = 'Database Dataset'

        ds = container.Dataset.load(
            'file://{dataset_path}'.format(dataset_path=dataset_path), dataset_id=dataset_id, dataset_name=dataset_name
        )

        self._test_csv_without_d3m_index(ds, dataset_path, dataset_id, dataset_name)

    def _test_csv_without_d3m_index(self, ds, dataset_path, dataset_id, dataset_name):
        ds.metadata.check(ds)

        for row in ds['learningData']:
            for cell in row:
                # Nothing should be parsed from a string.
                self.assertIsInstance(cell, str)

        self.assertEqual(len(ds['learningData']), 64, dataset_name)
        self.assertEqual(len(ds['learningData'].iloc[0]), 5, dataset_name)

        self.maxDiff = None

        self.assertEqual(
            convert_metadata(ds.metadata.query(())),
            {
                'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
                'structural_type': 'd3m.container.dataset.Dataset',
                'id': dataset_id,
                'name': dataset_name,
                'stored_size': 1794,
                'location_uris': ['file://localhost{dataset_path}'.format(dataset_path=dataset_path)],
                'dimension': {
                    'length': 1,
                    'name': 'resources',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/DatasetResource'],
                },
                'digest': 'b22431ee93c7b5fd6405c813bc67bfe6b2e1718eb6080cc50ff90ef6b2812139',
            },
            dataset_name,
        )

        self.assertEqual(
            convert_metadata(ds.metadata.query(('learningData',))),
            {
                'structural_type': 'd3m.container.pandas.DataFrame',
                'semantic_types': [
                    'https://metadata.datadrivendiscovery.org/types/Table',
                    'https://metadata.datadrivendiscovery.org/types/DatasetEntryPoint',
                ],
                'dimension': {
                    'name': 'rows',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                    'length': 64,
                },
            },
            dataset_name,
        )

        self.assertEqual(
            convert_metadata(ds.metadata.query(('learningData', metadata_base.ALL_ELEMENTS))),
            {
                'dimension': {
                    'name': 'columns',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                    'length': 5,
                }
            },
            dataset_name,
        )

        self.assertEqual(
            convert_metadata(ds.metadata.query(('learningData', metadata_base.ALL_ELEMENTS, 0))),
            {
                'name': 'd3mIndex',
                'semantic_types': [
                    'http://schema.org/Integer',
                    'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                ],
                'structural_type': 'numpy.int64',
            },
            dataset_name,
        )

        for i in range(1, 5):
            self.assertEqual(
                convert_metadata(ds.metadata.query(('learningData', metadata_base.ALL_ELEMENTS, i))),
                {
                    'name': ['code', 'key', 'year', 'value'][i - 1],
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/UnknownType'],
                    'structural_type': 'str',
                },
                dataset_name,
            )

    def test_csv_lazy_without_d3m_index(self):
        dataset_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'database_dataset_1', 'tables', 'values.csv')
        )

        dataset_id = '7cd469db-e922-4418-84a9-cda9517251d1'
        dataset_name = 'Database Dataset'

        ds = container.Dataset.load(
            'file://{dataset_path}'.format(dataset_path=dataset_path),
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            lazy=True,
        )

        ds.metadata.check(ds)

        self.assertTrue(len(ds) == 0)
        self.assertTrue(ds.is_lazy())

        self.maxDiff = None

        self.assertEqual(
            convert_metadata(ds.metadata.query(())),
            {
                'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
                'structural_type': 'd3m.container.dataset.Dataset',
                'id': dataset_id,
                'name': dataset_name,
                'location_uris': ['file://localhost{dataset_path}'.format(dataset_path=dataset_path)],
                'dimension': {
                    'length': 0,
                    'name': 'resources',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/DatasetResource'],
                },
            },
        )

        self.assertEqual(convert_metadata(ds.metadata.query(('learningData',))), {})
        self.assertEqual(convert_metadata(ds.metadata.query(('learningData', metadata_base.ALL_ELEMENTS))), {})
        self.assertEqual(convert_metadata(ds.metadata.query(('learningData', metadata_base.ALL_ELEMENTS, 0))), {})

        ds.load_lazy()

        self.assertFalse(ds.is_lazy())

        self._test_csv_without_d3m_index(ds, dataset_path, dataset_id, dataset_name)

    def test_openml(self):
        self.maxDiff = None

        # TODO: Try also with 1414. Do we have to convert date columns to strings or something?
        #       See: https://github.com/openml/openml-data/issues/23
        for dataset_id in [8, 17, 61, 42, 46, 373, 41496]:
            dataset_uri = 'https://www.openml.org/d/{dataset_id}'.format(dataset_id=dataset_id)
            output_dataset_uri = 'file://{dataset_path}'.format(
                dataset_path=os.path.join(self.test_dir, str(dataset_id), 'datasetDoc.json')
            )

            ds_1 = dataset.Dataset.load(dataset_uri=dataset_uri, dataset_id=str(dataset_id))
            ds_1.save(dataset_uri=output_dataset_uri)
            ds_2 = dataset.Dataset.load(dataset_uri=output_dataset_uri)

            self._test_openml_compare_loaded(ds_1, ds_2)

    def _test_openml_compare_loaded(self, ds_1, ds_2):
        keys_to_remove = ['digest', 'location_uris']
        for metadata_key in keys_to_remove:
            ds_1.metadata = ds_1.metadata.update((), {metadata_key: metadata_base.NO_VALUE})
            ds_2.metadata = ds_2.metadata.update((), {metadata_key: metadata_base.NO_VALUE})

        for resource_id in ds_1:
            # Additional metadata added when saving.
            ds_1.metadata = ds_1.metadata.update(
                (resource_id, metadata_base.ALL_ELEMENTS, metadata_base.ALL_ELEMENTS), {'structural_type': str}
            )

            # When saving, we convert all columns to string type.
            for column_index in range(
                ds_1.metadata.query((resource_id, metadata_base.ALL_ELEMENTS))['dimension']['length']
            ):
                ds_1.metadata = ds_1.metadata.update(
                    (resource_id, metadata_base.ALL_ELEMENTS, column_index), {'structural_type': str}
                )

        self.assertEqual(ds_1.metadata.to_internal_json_structure(), ds_2.metadata.to_internal_json_structure())

    def test_openml_nonlazy(self):
        dataset_id = 61
        dataset_name = 'iris'
        dataset_uri = 'https://www.openml.org/d/{dataset_id}'.format(dataset_id=dataset_id)
        ds = dataset.Dataset.load(dataset_uri, dataset_id=str(dataset_id), dataset_name=dataset_name)

        self._openml_check(ds, dataset_uri)

    def _openml_check_top_metadata(self, ds, dataset_uri, resources):
        self.assertEqual(
            convert_metadata(ds.metadata.query(())),
            {
                'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
                'structural_type': 'd3m.container.dataset.Dataset',
                'id': '61',
                'name': 'iris',
                'location_uris': [dataset_uri],
                'description': """**Author**: R.A. Fisher  
**Source**: [UCI](https://archive.ics.uci.edu/ml/datasets/Iris) - 1936 - Donated by Michael Marshall  
**Please cite**:   

**Iris Plants Database**  
This is perhaps the best known database to be found in the pattern recognition literature.  Fisher's paper is a classic in the field and is referenced frequently to this day.  (See Duda & Hart, for example.)  The data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant.  One class is     linearly separable from the other 2; the latter are NOT linearly separable from each other.

Predicted attribute: class of iris plant.  
This is an exceedingly simple domain.  
 
### Attribute Information:
    1. sepal length in cm
    2. sepal width in cm
    3. petal length in cm
    4. petal width in cm
    5. class: 
       -- Iris Setosa
       -- Iris Versicolour
       -- Iris Virginica""",
                'dimension': {
                    'length': resources,
                    'name': 'resources',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/DatasetResource'],
                },
                'keywords': [
                    'study_1',
                    'study_25',
                    'study_4',
                    'study_41',
                    'study_50',
                    'study_52',
                    'study_7',
                    'study_86',
                    'study_88',
                    'study_89',
                    'uci',
                ],
                'source': {
                    'license': 'Public',
                    'name': 'R.A. Fisher',
                    'published': '1936-01-01T00:00:00Z',
                    'uris': [
                        'https://www.openml.org/d/61',
                        'https://archive.ics.uci.edu/ml/datasets/Iris',
                        'http://digital.library.adelaide.edu.au/dspace/handle/2440/15227',
                    ],
                },
                'version': '1',
                'digest': '3b516a917d2f91d898be96391761e9e4aa7c4817bd45c2a89aace3fd6cc88d10',
                'data_metafeatures': {
                    'dimensionality': float(1 / 30),
                    'kurtosis_of_attributes': {
                        'max': 0.2907810623654319,
                        'mean': -0.7507394876837399,
                        'median': -0.9459091062274964,
                        'min': -1.401920800645399,
                        'quartile_1': -1.3863791432688857,
                        'quartile_3': 0.08006978644516216,
                    },
                    'mean_of_attributes': {
                        'max': 5.843333333333334,
                        'mean': 3.4636666666666667,
                        'median': 3.406333333333333,
                        'min': 1.1986666666666665,
                        'quartile_1': 1.6624999999999999,
                        'quartile_3': 5.322166666666667,
                    },
                    'number_distinct_values_of_categorical_attributes': {
                        'max': 3.0,
                        'min': 3.0,
                        'mean': 3.0,
                        'std': 0.0,
                    },
                    'number_of_attributes': 5,
                    'number_of_binary_attributes': 0,
                    'number_of_categorical_attributes': 1,
                    'number_of_instances': 150,
                    'number_of_instances_with_missing_values': 0,
                    'number_of_missing_values': 0,
                    'number_of_numeric_attributes': 4,
                    'ratio_of_binary_attributes': 0.0,
                    'ratio_of_categorical_attributes': 20.0,
                    'ratio_of_instances_with_missing_values': 0.0,
                    'ratio_of_missing_values': 0.0,
                    'ratio_of_numeric_attributes': 80.0,
                    'skew_of_attributes': {
                        'max': 0.33405266217208907,
                        'mean': 0.067375701047788,
                        'median': 0.10495719724642329,
                        'min': -0.2744642524737837,
                        'quartile_1': -0.2320973298913695,
                        'quartile_3': 0.32926723578831013,
                    },
                    'standard_deviation_of_attributes': {
                        'max': 1.7644204199522626,
                        'mean': 0.9473104002482851,
                        'median': 0.795613434839352,
                        'min': 0.43359431136217386,
                        'quartile_1': 0.5159859189468406,
                        'quartile_3': 1.5303318469586626,
                    },


                },
            },
        )

    def _openml_check(self, ds, dataset_uri):
        self.maxDiff = None

        ds.metadata.check(ds)

        self.assertEqual(len(ds['learningData']), 150)
        self.assertEqual(len(ds['learningData'].iloc[0]), 6)
        self.assertEqual(ds['learningData'].iloc[0, 5], 'Iris-setosa')
        self.assertEqual(ds['learningData'].dtypes[5], numpy.object)

        self._openml_check_top_metadata(ds, dataset_uri, 1)

        self.assertEqual(
            convert_metadata(ds.metadata.query(('learningData',))),
            {
                'structural_type': 'd3m.container.pandas.DataFrame',
                'semantic_types': [
                    'https://metadata.datadrivendiscovery.org/types/Table',
                    'https://metadata.datadrivendiscovery.org/types/DatasetEntryPoint',
                ],
                'dimension': {
                    'name': 'rows',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                    'length': 150,
                },
            },
        )

        self.assertEqual(
            convert_metadata(ds.metadata.query(('learningData', metadata_base.ALL_ELEMENTS))),
            {
                'dimension': {
                    'name': 'columns',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                    'length': 6,
                }
            },
        )

        self.assertEqual(
            convert_metadata(ds.metadata.query(('learningData', metadata_base.ALL_ELEMENTS, 0))),
            {
                'name': 'd3mIndex',
                'structural_type': 'int',
                'semantic_types': [
                    'http://schema.org/Integer',
                    'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                ],
            },
        )

        for i in range(1, 5):
            self.assertEqual(
                convert_metadata(ds.metadata.query(('learningData', metadata_base.ALL_ELEMENTS, i))),
                {
                    'name': ['sepallength', 'sepalwidth', 'petallength', 'petalwidth'][i - 1],
                    'structural_type': 'float',
                    'semantic_types': [
                        'http://schema.org/Float',
                        'https://metadata.datadrivendiscovery.org/types/Attribute',
                    ],
                },
            )

        self.assertEqual(
            convert_metadata(ds.metadata.query(('learningData', metadata_base.ALL_ELEMENTS, 5))),
            {
                'name': 'class',
                'structural_type': 'str',
                'semantic_types': [
                    'https://metadata.datadrivendiscovery.org/types/CategoricalData',
                    'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',
                    'https://metadata.datadrivendiscovery.org/types/Attribute',
                ],
            },
        )

    def test_openml_lazy(self):
        self.maxDiff = None

        dataset_id = 61
        dataset_name = 'iris'
        dataset_uri = 'https://www.openml.org/d/{dataset_id}'.format(dataset_id=dataset_id)
        ds = dataset.Dataset.load(dataset_uri, dataset_id=str(dataset_id), dataset_name=dataset_name, lazy=True)

        ds.metadata.check(ds)

        self.assertEqual(len(ds), 0)
        self.assertTrue(ds.is_lazy())

        self._openml_check_top_metadata(ds, dataset_uri, 0)

        self.assertEqual(convert_metadata(ds.metadata.query(('learningData',))), {})
        self.assertEqual(convert_metadata(ds.metadata.query(('learningData', metadata_base.ALL_ELEMENTS))), {})
        self.assertEqual(convert_metadata(ds.metadata.query(('learningData', metadata_base.ALL_ELEMENTS, 0))), {})

        ds.load_lazy()

        self.assertEqual(len(ds), 1)
        self.assertFalse(ds.is_lazy())

        self._openml_check(ds, dataset_uri)


if __name__ == '__main__':
    unittest.main()
