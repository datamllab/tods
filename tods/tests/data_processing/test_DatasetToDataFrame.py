import os.path
import unittest



from d3m import container, utils
from d3m.metadata import base as metadata_base

from tods.data_processing import DatasetToDataframe

import utils as test_utils


class ColumnParserPrimitiveTestCase(unittest.TestCase):
    def test_basic(self):
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '..', '..', 'datasets', 'anomaly','yahoo_sub_5','TRAIN','dataset_TRAIN', 'datasetDoc.json'))


        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        hyperparams_class = DatasetToDataframe.DatasetToDataFramePrimitive.metadata.get_hyperparams()

        primitive = DatasetToDataframe.DatasetToDataFramePrimitive(hyperparams=hyperparams_class.defaults())

        call_metadata = primitive.produce(inputs=dataset)

        dataframe = call_metadata.value


        first_row = list(dataframe.itertuples(index=False, name=None))[0]

        self.assertEqual(first_row, ('0', '1', '12183', '0.0', '3.7166666666667', '5', '2109', '0'))

        self.assertEqual([type(o) for o in first_row], [str,str, str,str, str, str, str, str])

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


        self.assertEqual(test_utils.convert_through_json(metadata.query((metadata_base.ALL_ELEMENTS, 0))), {'name': 'd3mIndex', 'structural_type': 'str', 'semantic_types': ['http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/PrimaryKey']})
        self.assertEqual(test_utils.convert_through_json(metadata.query((metadata_base.ALL_ELEMENTS, 1))), {'name': 'timestamp', 'structural_type': 'str', 'semantic_types': ['http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/Attribute']})
        self.assertEqual(test_utils.convert_through_json(metadata.query((metadata_base.ALL_ELEMENTS, 2))), {'name': 'value_0', 'structural_type': 'str', 'semantic_types': ['http://schema.org/Float', 'https://metadata.datadrivendiscovery.org/types/Attribute']})

        self.assertEqual(test_utils.convert_through_json(metadata.query((metadata_base.ALL_ELEMENTS, 3))), {'name': 'value_1', 'structural_type': 'str', 'semantic_types': ['http://schema.org/Float', 'https://metadata.datadrivendiscovery.org/types/Attribute']})

        self.assertEqual(test_utils.convert_through_json(metadata.query((metadata_base.ALL_ELEMENTS, 4))), {'name': 'value_2', 'structural_type': 'str', 'semantic_types': ['http://schema.org/Float', 'https://metadata.datadrivendiscovery.org/types/Attribute']})

        self.assertEqual(test_utils.convert_through_json(metadata.query((metadata_base.ALL_ELEMENTS, 5))), {'name': 'value_3', 'structural_type': 'str', 'semantic_types': ['http://schema.org/Float', 'https://metadata.datadrivendiscovery.org/types/Attribute']})

        self.assertEqual(test_utils.convert_through_json(metadata.query((metadata_base.ALL_ELEMENTS, 6))), {'name': 'value_4', 'structural_type': 'str', 'semantic_types': ['http://schema.org/Float', 'https://metadata.datadrivendiscovery.org/types/Attribute']})

        self.assertEqual(test_utils.convert_through_json(metadata.query((metadata_base.ALL_ELEMENTS, 7))), {'name': 'ground_truth', 'structural_type': 'str', 'semantic_types': ['http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/SuggestedTarget', 'https://metadata.datadrivendiscovery.org/types/Attribute']})



if __name__ == '__main__':
    unittest.main()
