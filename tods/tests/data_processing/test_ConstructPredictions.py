import copy
import os
import unittest

import numpy

from d3m import container
from d3m.metadata import base as metadata_base

from tods.data_processing import DatasetToDataframe , ConstructPredictions , ExtractColumnsBySemanticTypes

import utils as test_utils


class ConstructPredictionsPrimitiveTestCase(unittest.TestCase):
    # TODO: Make this part of metadata API.
    #       Something like setting a semantic type for given columns.
    def _mark_all_targets(self, dataset, targets):
        for target in targets:
            dataset.metadata = dataset.metadata.add_semantic_type((target['resource_id'], metadata_base.ALL_ELEMENTS, target['column_index']), 'https://metadata.datadrivendiscovery.org/types/Target')
            dataset.metadata = dataset.metadata.add_semantic_type((target['resource_id'], metadata_base.ALL_ELEMENTS, target['column_index']), 'https://metadata.datadrivendiscovery.org/types/TrueTarget')
            dataset.metadata = dataset.metadata.remove_semantic_type((target['resource_id'], metadata_base.ALL_ELEMENTS, target['column_index']), 'https://metadata.datadrivendiscovery.org/types/Attribute')

    def _get_yahoo_dataframe(self):
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '..', '..', 'datasets', 'anomaly','yahoo_sub_5','TRAIN','dataset_TRAIN', 'datasetDoc.json'))

        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        self._mark_all_targets(dataset, [{'resource_id': 'learningData', 'column_index': 5}])

        hyperparams_class = DatasetToDataframe.DatasetToDataFramePrimitive.metadata.get_hyperparams()

        primitive = DatasetToDataframe.DatasetToDataFramePrimitive(hyperparams=hyperparams_class.defaults())

        call_metadata = primitive.produce(inputs=dataset)

        dataframe = call_metadata.value

        return dataframe

    def test_correct_order(self):
        dataframe = self._get_yahoo_dataframe()

        hyperparams_class = ExtractColumnsBySemanticTypes.ExtractColumnsBySemanticTypesPrimitive.metadata.get_hyperparams()

        # We extract both the primary index and targets. So it is in the output format already.
        primitive = ExtractColumnsBySemanticTypes.ExtractColumnsBySemanticTypesPrimitive(hyperparams=hyperparams_class.defaults().replace({'semantic_types': ('https://metadata.datadrivendiscovery.org/types/PrimaryKey', 'https://metadata.datadrivendiscovery.org/types/Target',)}))

        call_metadata = primitive.produce(inputs=dataframe)

        targets = call_metadata.value

        # We pretend these are our predictions.
        targets.metadata = targets.metadata.remove_semantic_type((metadata_base.ALL_ELEMENTS, 1), 'https://metadata.datadrivendiscovery.org/types/TrueTarget')
        targets.metadata = targets.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, 1), 'https://metadata.datadrivendiscovery.org/types/PredictedTarget')

        # We switch columns around.
        targets = targets.select_columns([1, 0])

        hyperparams_class = ConstructPredictions.ConstructPredictionsPrimitive.metadata.get_hyperparams()

        construct_primitive = ConstructPredictions.ConstructPredictionsPrimitive(hyperparams=hyperparams_class.defaults())

        call_metadata = construct_primitive.produce(inputs=targets, reference=dataframe)

        dataframe = call_metadata.value

        self.assertEqual(list(dataframe.columns), ['d3mIndex', 'value_3'])

        self._test_metadata(dataframe.metadata)




    def _test_metadata(self, metadata, no_metadata=False):
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

        if no_metadata:
            self.assertEqual(test_utils.convert_through_json(metadata.query((metadata_base.ALL_ELEMENTS, 1))), {
                'name': 'value_3',
                'structural_type': 'str',
                'semantic_types': [
                    'https://metadata.datadrivendiscovery.org/types/Target',
                    'https://metadata.datadrivendiscovery.org/types/PredictedTarget',
                ],
            })

        else:
            self.assertEqual(test_utils.convert_through_json(metadata.query((metadata_base.ALL_ELEMENTS, 1))), {
                'name': 'value_3',
                'structural_type': 'str',
                'semantic_types': [
                    'http://schema.org/Float',
                    'https://metadata.datadrivendiscovery.org/types/Target',
                    'https://metadata.datadrivendiscovery.org/types/PredictedTarget',
                ],
            })

    def test_all_columns(self):
        dataframe = self._get_yahoo_dataframe()

        # We use all columns. Output has to be just index and targets.
        targets = copy.copy(dataframe)

        # We pretend these are our predictions.
        targets.metadata = targets.metadata.remove_semantic_type((metadata_base.ALL_ELEMENTS, 5),
                                                                 'https://metadata.datadrivendiscovery.org/types/TrueTarget')
        targets.metadata = targets.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, 5),
                                                              'https://metadata.datadrivendiscovery.org/types/PredictedTarget')

        hyperparams_class = ConstructPredictions.ConstructPredictionsPrimitive.metadata.get_hyperparams()

        construct_primitive = ConstructPredictions.ConstructPredictionsPrimitive(
            hyperparams=hyperparams_class.defaults())

        call_metadata = construct_primitive.produce(inputs=targets, reference=dataframe)

        dataframe = call_metadata.value

        self.assertEqual(list(dataframe.columns), ['d3mIndex', 'value_3'])

        self._test_metadata(dataframe.metadata)

    def test_missing_index(self):
        dataframe = self._get_yahoo_dataframe()

        # We just use all columns.
        targets = copy.copy(dataframe)

        # We pretend these are our predictions.
        targets.metadata = targets.metadata.remove_semantic_type((metadata_base.ALL_ELEMENTS, 5),
                                                                 'https://metadata.datadrivendiscovery.org/types/TrueTarget')
        targets.metadata = targets.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, 5),
                                                              'https://metadata.datadrivendiscovery.org/types/PredictedTarget')

        # Remove primary index. This one has to be reconstructed.
        targets = targets.remove_columns([0])

        hyperparams_class = ConstructPredictions.ConstructPredictionsPrimitive.metadata.get_hyperparams()

        construct_primitive = ConstructPredictions.ConstructPredictionsPrimitive(
            hyperparams=hyperparams_class.defaults())

        call_metadata = construct_primitive.produce(inputs=targets, reference=dataframe)

        dataframe = call_metadata.value

        self.assertEqual(list(dataframe.columns), ['d3mIndex', 'value_3'])

        self._test_metadata(dataframe.metadata)

    def test_just_targets_no_metadata(self):
        dataframe = self._get_yahoo_dataframe()

        hyperparams_class = ExtractColumnsBySemanticTypes.ExtractColumnsBySemanticTypesPrimitive.metadata.get_hyperparams()

        # We extract just targets.
        primitive = ExtractColumnsBySemanticTypes.ExtractColumnsBySemanticTypesPrimitive(
            hyperparams=hyperparams_class.defaults().replace(
                {'semantic_types': ('https://metadata.datadrivendiscovery.org/types/Target',)}))

        call_metadata = primitive.produce(inputs=dataframe)

        targets = call_metadata.value

        # Remove all metadata.
        targets.metadata = metadata_base.DataMetadata().generate(targets)

        hyperparams_class = ConstructPredictions.ConstructPredictionsPrimitive.metadata.get_hyperparams()

        construct_primitive = ConstructPredictions.ConstructPredictionsPrimitive(
            hyperparams=hyperparams_class.defaults())

        call_metadata = construct_primitive.produce(inputs=targets, reference=dataframe)

        dataframe = call_metadata.value

        self.assertEqual(list(dataframe.columns), ['d3mIndex', 'value_3'])

        self._test_metadata(dataframe.metadata, True)

    def test_float_vector(self):
        dataframe = container.DataFrame({
            'd3mIndex': [0],
            'target': [container.ndarray(numpy.array([3, 5, 9, 10]))],
        }, generate_metadata=True)

        # Update metadata.
        dataframe.metadata = dataframe.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, 0),
                                                                  'https://metadata.datadrivendiscovery.org/types/PrimaryKey')
        dataframe.metadata = dataframe.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, 1),
                                                                  'https://metadata.datadrivendiscovery.org/types/PredictedTarget')

        hyperparams_class = ConstructPredictions.ConstructPredictionsPrimitive.metadata.get_hyperparams()

        construct_primitive = ConstructPredictions.ConstructPredictionsPrimitive(
            hyperparams=hyperparams_class.defaults())

        dataframe = construct_primitive.produce(inputs=dataframe, reference=dataframe).value

        self.assertEqual(list(dataframe.columns), ['d3mIndex', 'target'])

        self.assertEqual(dataframe.values.tolist(), [
            [0, '3,5,9,10'],
        ])

        self.assertEqual(dataframe.metadata.query_column(1), {
            'structural_type': str,
            'name': 'target',
            'semantic_types': (
                'https://metadata.datadrivendiscovery.org/types/PredictedTarget',
            ),
        })


if __name__ == '__main__':
    unittest.main()
