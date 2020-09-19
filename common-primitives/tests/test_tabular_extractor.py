import os
import unittest

from d3m import container, utils
from d3m.metadata import base as metadata_base

from common_primitives import dataset_to_dataframe, column_parser, tabular_extractor

import utils as test_utils


class TabularExtractorPrimitiveTestCase(unittest.TestCase):
    def setUp(self):
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'iris_dataset_1', 'datasetDoc.json'))

        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        # We mark targets as attributes.
        dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 5), 'https://metadata.datadrivendiscovery.org/types/Target')
        dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 5), 'https://metadata.datadrivendiscovery.org/types/TrueTarget')
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 5), 'https://metadata.datadrivendiscovery.org/types/Attribute')

        self.dataset = dataset

        # DatasetToDataFramePrimitive

        df_hyperparams_class = dataset_to_dataframe.DatasetToDataFramePrimitive.metadata.get_hyperparams()

        df_primitive = dataset_to_dataframe.DatasetToDataFramePrimitive(hyperparams=df_hyperparams_class.defaults())

        df_dataframe = df_primitive.produce(inputs=self.dataset).value

        # Set some missing values.
        df_dataframe.iloc[1, 1] = ""
        df_dataframe.iloc[10, 1] = ""
        df_dataframe.iloc[15, 1] = ""

        # ColumnParserPrimitive

        cp_hyperparams_class = column_parser.ColumnParserPrimitive.metadata.get_hyperparams()

        # To simulate how Pandas "read_csv" is reading CSV files, we parse just numbers.
        cp_primitive = column_parser.ColumnParserPrimitive(
            hyperparams=cp_hyperparams_class.defaults().replace({
                'parse_semantic_types': ['http://schema.org/Integer', 'http://schema.org/Float'],
            }),
        )

        self.dataframe = cp_primitive.produce(inputs=df_dataframe).value

    def test_defaults(self):
        te_hyperparams_class = tabular_extractor.AnnotatedTabularExtractorPrimitive.metadata.get_hyperparams()

        # It one-hot encodes categorical columns, it imputes numerical values,
        # and adds missing indicator column for each.
        te_primitive = tabular_extractor.AnnotatedTabularExtractorPrimitive(
            hyperparams=te_hyperparams_class.defaults(),
        )

        te_primitive.set_training_data(inputs=self.dataframe)
        te_primitive.fit()

        dataframe = te_primitive.produce(inputs=self.dataframe).value

        # 1 index column, 4 numerical columns with one indicator column each,
        # 3 columns for one-hot encoding of "target" column and indicator column for that.
        self.assertEqual(dataframe.shape, (150, 13))

        self.assertEqual(test_utils.convert_through_json(utils.to_json_structure(dataframe.metadata.to_internal_simple_structure())), [{
            'selector': [],
            'metadata': {
                'dimension': {
                    'length': 150,
                    'name': 'rows',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                },
                'schema': 'https://metadata.datadrivendiscovery.org/schemas/v0/container.json',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Table'],
                'structural_type': 'd3m.container.pandas.DataFrame',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__'],
            'metadata': {
                'dimension': {
                    'length': 13,
                    'name': 'columns',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 0],
            'metadata': {
                'name': 'd3mIndex',
                'semantic_types': ['http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/PrimaryKey'],
                'structural_type': 'int',
             },
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
                'structural_type': 'numpy.float64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 2],
            'metadata': {
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
                'structural_type': 'numpy.float64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 3],
            'metadata': {
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
                'structural_type': 'numpy.float64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 4],
            'metadata': {
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
                'structural_type': 'numpy.float64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 5],
            'metadata': {
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
                'structural_type': 'numpy.float64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 6],
            'metadata': {
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
                'structural_type': 'numpy.float64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 7],
            'metadata': {
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
                'structural_type': 'numpy.float64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 8],
            'metadata': {
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
                'structural_type': 'numpy.float64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 9],
            'metadata': {
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
                'structural_type': 'numpy.float64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 10],
            'metadata': {
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
                'structural_type': 'numpy.float64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 11],
            'metadata': {
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
                'structural_type': 'numpy.float64',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 12],
            'metadata': {
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
                'structural_type': 'numpy.float64',
            },
        }])


if __name__ == '__main__':
    unittest.main()
