import json
import os

from d3m import utils, container
from d3m.metadata import base as metadata_base

from tods.data_processing import DatasetToDataFramePrimitive


def convert_metadata(metadata):
    return json.loads(json.dumps(metadata, cls=utils.JsonEncoder))


def load_iris_metadata():
    dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'iris_dataset_1', 'datasetDoc.json'))
    dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))
    return dataset


def test_iris_metadata(test_obj, metadata, structural_type, rows_structural_type=None):
    test_obj.maxDiff = None

    test_obj.assertEqual(convert_metadata(metadata.query(())), {
        'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
        'structural_type': structural_type,
        'semantic_types': [
            'https://metadata.datadrivendiscovery.org/types/Table',
        ],
        'dimension': {
            'name': 'rows',
            'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
            'length': 150,
        }
    })

    if rows_structural_type is None:
        test_obj.assertEqual(convert_metadata(metadata.query((metadata_base.ALL_ELEMENTS,))), {
            'dimension': {
                'name': 'columns',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                'length': 6,
            }
        })
    else:
        test_obj.assertEqual(convert_metadata(metadata.query((metadata_base.ALL_ELEMENTS,))), {
            'structural_type': rows_structural_type,
            'dimension': {
                'name': 'columns',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                'length': 6,
            }
        })

    test_obj.assertEqual(convert_metadata(metadata.query((metadata_base.ALL_ELEMENTS, 0))), {
        'name': 'd3mIndex',
        'structural_type': 'str',
        'semantic_types': [
            'http://schema.org/Integer',
            'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
        ],
    })

    for i in range(1, 5):
        test_obj.assertEqual(convert_metadata(metadata.query((metadata_base.ALL_ELEMENTS, i))), {
            'name': ['sepalLength', 'sepalWidth', 'petalLength', 'petalWidth'][i - 1],
            'structural_type': 'str',
            'semantic_types': [
                'http://schema.org/Float',
                'https://metadata.datadrivendiscovery.org/types/Attribute',
            ],
        }, i)

    test_obj.assertEqual(convert_metadata(metadata.query((metadata_base.ALL_ELEMENTS, 5))), {
        'name': 'species',
        'structural_type': 'str',
        'semantic_types': [
            'https://metadata.datadrivendiscovery.org/types/CategoricalData',
            'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',
            'https://metadata.datadrivendiscovery.org/types/Attribute',
        ],
    })


def convert_through_json(data):
    return json.loads(json.dumps(data, cls=utils.JsonEncoder))


def normalize_semantic_types(data):
    if isinstance(data, dict):
        if 'semantic_types' in data:
            # We sort them so that it is easier to compare them.
            data['semantic_types'] = sorted(data['semantic_types'])

        return {key: normalize_semantic_types(value) for key, value in data.items()}

    return data


def effective_metadata(metadata):
    output =  metadata.to_json_structure()

    for entry in output:
        entry['metadata'] = normalize_semantic_types(entry['metadata'])

    return output


def get_dataframe(dataset):
    dataset_hyperparams_class = dataset_to_dataframe.DatasetToDataFramePrimitive.metadata.get_hyperparams()
    dataframe_primitive = dataset_to_dataframe.DatasetToDataFramePrimitive(hyperparams=dataset_hyperparams_class.defaults())
    dataframe = dataframe_primitive.produce(inputs=dataset).value
    return dataframe
