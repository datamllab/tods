import os.path
import unittest



from d3m import container, utils
from d3m.metadata import base as metadata_base

from tods.data_processing import DatasetToDataframe, ExtractColumnsBySemanticTypes

import utils as test_utils


class ExtractColumnsBySemanticTypePrimitiveTestCase(unittest.TestCase):
    def test_basic(self):
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '..', '..', 'datasets', 'anomaly','yahoo_sub_5','TRAIN','dataset_TRAIN', 'datasetDoc.json'))

        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        # We set semantic types like runtime would.
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 7), 'https://metadata.datadrivendiscovery.org/types/Target')
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 7), 'https://metadata.datadrivendiscovery.org/types/TrueTarget')
        dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 7), 'https://metadata.datadrivendiscovery.org/types/Attribute')

        hyperparams_class = DatasetToDataframe.DatasetToDataFramePrimitive.metadata.get_hyperparams()

        primitive = DatasetToDataframe.DatasetToDataFramePrimitive(hyperparams=hyperparams_class.defaults())

        call_metadata = primitive.produce(inputs=dataset)

        dataframe = call_metadata.value

        hyperparams_class = ExtractColumnsBySemanticTypes.ExtractColumnsBySemanticTypesPrimitive.metadata.get_hyperparams()

        primitive = ExtractColumnsBySemanticTypes.ExtractColumnsBySemanticTypesPrimitive(hyperparams=hyperparams_class.defaults().replace({'semantic_types': ('https://metadata.datadrivendiscovery.org/types/Attribute', 'https://metadata.datadrivendiscovery.org/types/PrimaryKey')}))

        call_metadata = primitive.produce(inputs=dataframe)

        dataframe = call_metadata.value

        self._test_metadata(dataframe.metadata)

    def _test_metadata(self, metadata):
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
                'length': 7,
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





if __name__ == '__main__':
    unittest.main()
