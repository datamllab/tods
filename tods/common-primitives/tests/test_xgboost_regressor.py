import os
import pickle
import unittest

from sklearn.metrics import mean_squared_error

from d3m import container, utils
from d3m.metadata import base as metadata_base

from common_primitives import dataset_to_dataframe, extract_columns_semantic_types, xgboost_regressor, column_parser


class XGBoostRegressorTestCase(unittest.TestCase):
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
        col_index_list = list(range(len(dataframe.columns)))
        _, target = col_index_list.pop(0), col_index_list.pop(3)
        original_target_col = 5
        # We set custom metadata on columns.
        for column_index in col_index_list:
            dataframe.metadata = dataframe.metadata.update_column(column_index, {'custom_metadata': 'attributes'})
        dataframe.metadata = dataframe.metadata.update_column(target, {'custom_metadata': 'targets'})
        dataframe.metadata = dataframe.metadata.remove_semantic_type((metadata_base.ALL_ELEMENTS, target),
                                                                     'https://metadata.datadrivendiscovery.org/types/Attribute')
        dataframe.metadata = dataframe.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, original_target_col),
                                                                  'https://metadata.datadrivendiscovery.org/types/Attribute')
        # We set semantic types like runtime would.
        dataframe.metadata = dataframe.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, target),
                                                                  'https://metadata.datadrivendiscovery.org/types/Target')
        dataframe.metadata = dataframe.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, target),
                                                                  'https://metadata.datadrivendiscovery.org/types/TrueTarget')
        dataframe.metadata = dataframe.metadata.remove_semantic_type((metadata_base.ALL_ELEMENTS, target), 'https://metadata.datadrivendiscovery.org/types/Attribute')

        # Parsing.
        hyperparams_class = \
            column_parser.ColumnParserPrimitive.metadata.query()['primitive_code']['class_type_arguments'][
                'Hyperparams']
        primitive = column_parser.ColumnParserPrimitive(hyperparams=hyperparams_class.defaults())
        dataframe = primitive.produce(inputs=dataframe).value

        hyperparams_class = \
            extract_columns_semantic_types.ExtractColumnsBySemanticTypesPrimitive.metadata.query()['primitive_code'][
                'class_type_arguments']['Hyperparams']

        primitive = extract_columns_semantic_types.ExtractColumnsBySemanticTypesPrimitive(
            hyperparams=hyperparams_class.defaults().replace(
                {'semantic_types': ('https://metadata.datadrivendiscovery.org/types/Attribute',)}))
        attributes = primitive.produce(inputs=dataframe).value

        primitive = extract_columns_semantic_types.ExtractColumnsBySemanticTypesPrimitive(
            hyperparams=hyperparams_class.defaults().replace(
                {'semantic_types': ('https://metadata.datadrivendiscovery.org/types/TrueTarget',)}))
        targets = primitive.produce(inputs=dataframe).value

        return dataframe, attributes, targets

    def test_single_target(self):
        dataframe, attributes, targets = self._get_iris_columns()

        hyperparams_class = \
            xgboost_regressor.XGBoostGBTreeRegressorPrimitive.metadata.query()['primitive_code'][
                'class_type_arguments']['Hyperparams']
        primitive = xgboost_regressor.XGBoostGBTreeRegressorPrimitive(
            hyperparams=hyperparams_class.defaults().replace({'return_result': 'new', 'add_index_columns': False}))

        primitive.set_training_data(inputs=attributes, outputs=targets)
        primitive.fit()

        predictions = primitive.produce(inputs=attributes).value
        mse = mean_squared_error(targets, predictions)
        self.assertLessEqual(mse, 0.01)
        self.assertEqual(predictions.shape, (150, 1))
        self.assertTrue(predictions.metadata.has_semantic_type((metadata_base.ALL_ELEMENTS, 0),
                                                               'https://metadata.datadrivendiscovery.org/types/PredictedTarget'))
        self.assertFalse(predictions.metadata.has_semantic_type((metadata_base.ALL_ELEMENTS, 0),
                                                                'https://metadata.datadrivendiscovery.org/types/TrueTarget'))
        self.assertEqual(predictions.metadata.query_column(0)['name'], 'petalWidth')
        self.assertEqual(predictions.metadata.query_column(0)['custom_metadata'], 'targets')

        self._test_single_target_metadata(predictions.metadata)

        samples = primitive.sample(inputs=attributes).value

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].shape, (150, 1))
        self.assertTrue(samples[0].metadata.has_semantic_type((metadata_base.ALL_ELEMENTS, 0),
                                                              'https://metadata.datadrivendiscovery.org/types/PredictedTarget'))
        self.assertFalse(samples[0].metadata.has_semantic_type((metadata_base.ALL_ELEMENTS, 0),
                                                               'https://metadata.datadrivendiscovery.org/types/TrueTarget'))
        self.assertEqual(samples[0].metadata.query_column(0)['name'], 'petalWidth')
        self.assertEqual(samples[0].metadata.query_column(0)['custom_metadata'], 'targets')

    def test_single_target_continue(self):
        dataframe, attributes, targets = self._get_iris_columns()

        hyperparams_class = \
            xgboost_regressor.XGBoostGBTreeRegressorPrimitive.metadata.query()['primitive_code'][
                'class_type_arguments'][
                'Hyperparams']
        primitive = xgboost_regressor.XGBoostGBTreeRegressorPrimitive(
            hyperparams=hyperparams_class.defaults().replace({'return_result': 'new', 'add_index_columns': False}))

        primitive.set_training_data(inputs=attributes, outputs=targets)
        primitive.fit()
        # reset the training data to make continue_fit() work.
        primitive.set_training_data(inputs=attributes, outputs=targets)
        primitive.continue_fit()
        params = primitive.get_params()
        self.assertEqual(params['booster'].best_ntree_limit,
                         primitive.hyperparams['n_estimators'] + primitive.hyperparams['n_more_estimators'])
        predictions = primitive.produce(inputs=attributes).value
        mse = mean_squared_error(targets, predictions)
        self.assertLessEqual(mse, 0.01)
        self.assertEqual(predictions.shape, (150, 1))
        self.assertTrue(predictions.metadata.has_semantic_type((metadata_base.ALL_ELEMENTS, 0),
                                                               'https://metadata.datadrivendiscovery.org/types/PredictedTarget'))
        self.assertFalse(predictions.metadata.has_semantic_type((metadata_base.ALL_ELEMENTS, 0),
                                                                'https://metadata.datadrivendiscovery.org/types/TrueTarget'))
        self.assertEqual(predictions.metadata.query_column(0)['name'], 'petalWidth')
        self.assertEqual(predictions.metadata.query_column(0)['custom_metadata'], 'targets')

        self._test_single_target_metadata(predictions.metadata)

        samples = primitive.sample(inputs=attributes).value

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].shape, (150, 1))
        self.assertTrue(samples[0].metadata.has_semantic_type((metadata_base.ALL_ELEMENTS, 0),
                                                              'https://metadata.datadrivendiscovery.org/types/PredictedTarget'))
        self.assertFalse(samples[0].metadata.has_semantic_type((metadata_base.ALL_ELEMENTS, 0),
                                                               'https://metadata.datadrivendiscovery.org/types/TrueTarget'))
        self.assertEqual(samples[0].metadata.query_column(0)['name'], 'petalWidth')
        self.assertEqual(samples[0].metadata.query_column(0)['custom_metadata'], 'targets')

    def _test_single_target_metadata(self, predictions_metadata):
        expected_metadata = [{
            'selector': [],
            'metadata': {
                'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
                'structural_type': 'd3m.container.pandas.DataFrame',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Table'],
                'dimension': {
                    'name': 'rows',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                    'length': 150,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__'],
            'metadata': {
                'dimension': {
                    'name': 'columns',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                    'length': 1,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 0],
            'metadata': {
                'structural_type': 'float',
                'name': 'petalWidth',
                'semantic_types': ['http://schema.org/Float',
                                   'https://metadata.datadrivendiscovery.org/types/Target',
                                   'https://metadata.datadrivendiscovery.org/types/PredictedTarget'],
                'custom_metadata': 'targets',
            },
        }]

        self.assertEqual(utils.to_json_structure(predictions_metadata.to_internal_simple_structure()), expected_metadata)

    def test_multiple_targets(self):
        dataframe, attributes, targets = self._get_iris_columns()

        targets = targets.append_columns(targets)

        hyperparams_class = \
            xgboost_regressor.XGBoostGBTreeRegressorPrimitive.metadata.query()['primitive_code'][
                'class_type_arguments']['Hyperparams']
        primitive = xgboost_regressor.XGBoostGBTreeRegressorPrimitive(
            hyperparams=hyperparams_class.defaults().replace({'return_result': 'new', 'add_index_columns': False}))

        primitive.set_training_data(inputs=attributes, outputs=targets)
        primitive.fit()

        predictions = primitive.produce(inputs=attributes).value
        mse = mean_squared_error(targets, predictions)
        self.assertLessEqual(mse, 0.01)

        self.assertEqual(predictions.shape, (150, 2))
        for column_index in range(2):
            self.assertTrue(predictions.metadata.has_semantic_type((metadata_base.ALL_ELEMENTS, column_index),
                                                                   'https://metadata.datadrivendiscovery.org/types/PredictedTarget'))
            self.assertFalse(predictions.metadata.has_semantic_type((metadata_base.ALL_ELEMENTS, column_index),
                                                                    'https://metadata.datadrivendiscovery.org/types/TrueTarget'))
            self.assertEqual(predictions.metadata.query_column(column_index)['name'], 'petalWidth')
            self.assertEqual(predictions.metadata.query_column(column_index)['custom_metadata'], 'targets')

        samples = primitive.sample(inputs=attributes).value

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].shape, (150, 2))
        for column_index in range(2):
            self.assertTrue(samples[0].metadata.has_semantic_type((metadata_base.ALL_ELEMENTS, column_index),
                                                                  'https://metadata.datadrivendiscovery.org/types/PredictedTarget'))
            self.assertFalse(samples[0].metadata.has_semantic_type((metadata_base.ALL_ELEMENTS, column_index),
                                                                   'https://metadata.datadrivendiscovery.org/types/TrueTarget'))
            self.assertEqual(samples[0].metadata.query_column(column_index)['name'], 'petalWidth')
            self.assertEqual(samples[0].metadata.query_column(column_index)['custom_metadata'], 'targets')

        feature_importances = primitive.produce_feature_importances().value

        self.assertEqual(feature_importances.values.tolist(),
            [[0.0049971588887274265,
            0.006304567214101553,
            0.27505698800086975,
            0.7136412858963013]])

    def test_multiple_targets_continue(self):
        dataframe, attributes, targets = self._get_iris_columns()
        second_targets = targets.copy()
        second_targets.rename(columns={'petalWidth': 't-petalWidth'}, inplace=True)
        second_targets.metadata = second_targets.metadata.update_column(0, {'name': 't-petalWidth'})
        targets = targets.append_columns(second_targets)

        hyperparams_class = \
            xgboost_regressor.XGBoostGBTreeRegressorPrimitive.metadata.query()['primitive_code'][
                'class_type_arguments']['Hyperparams']
        primitive = xgboost_regressor.XGBoostGBTreeRegressorPrimitive(
            hyperparams=hyperparams_class.defaults().replace({'return_result': 'new', 'add_index_columns': False}))

        primitive.set_training_data(inputs=attributes, outputs=targets)
        primitive.fit()
        # Set training data again to make continue_fit work
        primitive.set_training_data(inputs=attributes, outputs=targets)
        primitive.continue_fit()
        params = primitive.get_params()
        for estimator in params['estimators']:
            self.assertEqual(estimator.get_booster().best_ntree_limit,
                             primitive.hyperparams['n_estimators'] + primitive.hyperparams['n_more_estimators'])

        predictions = primitive.produce(inputs=attributes).value
        mse = mean_squared_error(targets, predictions)
        self.assertLessEqual(mse, 0.01)
        self.assertEqual(predictions.shape, (150, 2))

        self.assertTrue(predictions.metadata.has_semantic_type((metadata_base.ALL_ELEMENTS, 0),
                                                               'https://metadata.datadrivendiscovery.org/types/PredictedTarget'))
        self.assertFalse(predictions.metadata.has_semantic_type((metadata_base.ALL_ELEMENTS, 0),
                                                                'https://metadata.datadrivendiscovery.org/types/TrueTarget'))
        self.assertEqual(predictions.metadata.query_column(0)['name'], 'petalWidth')
        self.assertEqual(predictions.metadata.query_column(0)['custom_metadata'], 'targets')
        self.assertTrue(predictions.metadata.has_semantic_type((metadata_base.ALL_ELEMENTS, 1),
                                                               'https://metadata.datadrivendiscovery.org/types/PredictedTarget'))
        self.assertFalse(predictions.metadata.has_semantic_type((metadata_base.ALL_ELEMENTS, 1),
                                                                'https://metadata.datadrivendiscovery.org/types/TrueTarget'))
        self.assertEqual(predictions.metadata.query_column(1)['name'], 't-petalWidth')
        self.assertEqual(predictions.metadata.query_column(1)['custom_metadata'], 'targets')

        samples = primitive.sample(inputs=attributes).value

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].shape, (150, 2))
        self.assertTrue(samples[0].metadata.has_semantic_type((metadata_base.ALL_ELEMENTS, 0),
                                                              'https://metadata.datadrivendiscovery.org/types/PredictedTarget'))
        self.assertFalse(samples[0].metadata.has_semantic_type((metadata_base.ALL_ELEMENTS, 0),
                                                               'https://metadata.datadrivendiscovery.org/types/TrueTarget'))
        self.assertEqual(samples[0].metadata.query_column(0)['name'], 'petalWidth')
        self.assertEqual(samples[0].metadata.query_column(0)['custom_metadata'], 'targets')
        self.assertTrue(samples[0].metadata.has_semantic_type((metadata_base.ALL_ELEMENTS, 1),
                                                              'https://metadata.datadrivendiscovery.org/types/PredictedTarget'))
        self.assertFalse(samples[0].metadata.has_semantic_type((metadata_base.ALL_ELEMENTS, 1),
                                                               'https://metadata.datadrivendiscovery.org/types/TrueTarget'))
        self.assertEqual(samples[0].metadata.query_column(1)['name'], 't-petalWidth')
        self.assertEqual(samples[0].metadata.query_column(1)['custom_metadata'], 'targets')

        feature_importances = primitive.produce_feature_importances().value

        self.assertEqual(feature_importances.values.tolist(),
            [[0.003233343129977584,
            0.003926052246242762,
            0.19553671777248383,
            0.7973038554191589]])

    def test_semantic_types(self):
        dataframe, attributes, targets = self._get_iris_columns()

        hyperparams_class = \
            xgboost_regressor.XGBoostGBTreeRegressorPrimitive.metadata.query()['primitive_code'][
                'class_type_arguments']['Hyperparams']
        primitive = xgboost_regressor.XGBoostGBTreeRegressorPrimitive(
            hyperparams=hyperparams_class.defaults().replace({'return_result': 'new', 'add_index_columns': False}))

        primitive.set_training_data(inputs=dataframe, outputs=dataframe)
        primitive.fit()

        predictions = primitive.produce(inputs=dataframe).value

        self.assertEqual(predictions.shape, (150, 1))
        self.assertTrue(predictions.metadata.has_semantic_type((metadata_base.ALL_ELEMENTS, 0),
                                                               'https://metadata.datadrivendiscovery.org/types/PredictedTarget'))
        self.assertFalse(predictions.metadata.has_semantic_type((metadata_base.ALL_ELEMENTS, 0),
                                                                'https://metadata.datadrivendiscovery.org/types/TrueTarget'))
        self.assertEqual(predictions.metadata.query_column(0)['name'], 'petalWidth')
        self.assertEqual(predictions.metadata.query_column(0)['custom_metadata'], 'targets')

        samples = primitive.sample(inputs=attributes).value

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].shape, (150, 1))
        self.assertTrue(samples[0].metadata.has_semantic_type((metadata_base.ALL_ELEMENTS, 0),
                                                              'https://metadata.datadrivendiscovery.org/types/PredictedTarget'))
        self.assertFalse(samples[0].metadata.has_semantic_type((metadata_base.ALL_ELEMENTS, 0),
                                                               'https://metadata.datadrivendiscovery.org/types/TrueTarget'))
        self.assertEqual(samples[0].metadata.query_column(0)['name'], 'petalWidth')
        self.assertEqual(samples[0].metadata.query_column(0)['custom_metadata'], 'targets')

        feature_importances = primitive.produce_feature_importances().value

        self.assertEqual(feature_importances.values.tolist(),
            [[0.0049971588887274265,
            0.006304567214101553,
            0.27505698800086975,
            0.7136412858963013]])

    def test_return_append(self):
        dataframe, attributes, targets = self._get_iris_columns()

        hyperparams_class = \
            xgboost_regressor.XGBoostGBTreeRegressorPrimitive.metadata.query()['primitive_code'][
                'class_type_arguments']['Hyperparams']
        primitive = xgboost_regressor.XGBoostGBTreeRegressorPrimitive(hyperparams=hyperparams_class.defaults())

        primitive.set_training_data(inputs=dataframe, outputs=dataframe)
        primitive.fit()

        predictions = primitive.produce(inputs=dataframe).value

        self.assertEqual(predictions.shape, (150, 7))
        self.assertTrue(predictions.metadata.has_semantic_type((metadata_base.ALL_ELEMENTS, 6),
                                                               'https://metadata.datadrivendiscovery.org/types/PredictedTarget'))
        self.assertFalse(predictions.metadata.has_semantic_type((metadata_base.ALL_ELEMENTS, 6),
                                                                'https://metadata.datadrivendiscovery.org/types/TrueTarget'))
        self.assertEqual(predictions.metadata.query_column(6)['name'], 'petalWidth')
        self.assertEqual(predictions.metadata.query_column(6)['custom_metadata'], 'targets')

        self._test_return_append_metadata(predictions.metadata)

    def _test_return_append_metadata(self, predictions_metadata):
        self.assertEqual(utils.to_json_structure(predictions_metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'structural_type': 'd3m.container.pandas.DataFrame',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Table'],
                'dimension': {
                    'name': 'rows',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                    'length': 150,
                },
                'schema': 'https://metadata.datadrivendiscovery.org/schemas/v0/container.json',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__'],
            'metadata': {
                'dimension': {
                    'name': 'columns',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                    'length': 7,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 0],
            'metadata': {
                'name': 'd3mIndex',
                'structural_type': 'int',
                'semantic_types': ['http://schema.org/Integer',
                                   'https://metadata.datadrivendiscovery.org/types/PrimaryKey'],
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {
                'name': 'sepalLength',
                'structural_type': 'float',
                'semantic_types': ['http://schema.org/Float',
                                   'https://metadata.datadrivendiscovery.org/types/Attribute'],
                'custom_metadata': 'attributes',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 2],
            'metadata': {
                'name': 'sepalWidth',
                'structural_type': 'float',
                'semantic_types': ['http://schema.org/Float',
                                   'https://metadata.datadrivendiscovery.org/types/Attribute'],
                'custom_metadata': 'attributes',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 3],
            'metadata': {
                'name': 'petalLength',
                'structural_type': 'float',
                'semantic_types': ['http://schema.org/Float',
                                   'https://metadata.datadrivendiscovery.org/types/Attribute'],
                'custom_metadata': 'attributes',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 4],
            'metadata': {
                'name': 'petalWidth',
                'structural_type': 'float',
                'semantic_types': ['http://schema.org/Float',
                                   'https://metadata.datadrivendiscovery.org/types/Target',
                                   'https://metadata.datadrivendiscovery.org/types/TrueTarget'],
                'custom_metadata': 'targets',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 5],
            'metadata': {
                'name': 'species',
                'structural_type': 'int',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/CategoricalData',
                                   'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',
                                   'https://metadata.datadrivendiscovery.org/types/Attribute', ],
                'custom_metadata': 'attributes',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 6],
            'metadata': {
                'structural_type': 'float',
                'name': 'petalWidth',
                'semantic_types': ['http://schema.org/Float',
                                   'https://metadata.datadrivendiscovery.org/types/Target',
                                   'https://metadata.datadrivendiscovery.org/types/PredictedTarget'],
                'custom_metadata': 'targets',
            },
        }])

    def test_return_new(self):
        dataframe, attributes, targets = self._get_iris_columns()

        hyperparams_class = \
            xgboost_regressor.XGBoostGBTreeRegressorPrimitive.metadata.query()['primitive_code'][
                'class_type_arguments']['Hyperparams']
        primitive = xgboost_regressor.XGBoostGBTreeRegressorPrimitive(
            hyperparams=hyperparams_class.defaults().replace({'return_result': 'new'}))

        primitive.set_training_data(inputs=dataframe, outputs=dataframe)
        primitive.fit()

        predictions = primitive.produce(inputs=dataframe).value

        self.assertEqual(predictions.shape, (150, 2))
        self.assertTrue(predictions.metadata.has_semantic_type((metadata_base.ALL_ELEMENTS, 1),
                                                               'https://metadata.datadrivendiscovery.org/types/PredictedTarget'))
        self.assertFalse(predictions.metadata.has_semantic_type((metadata_base.ALL_ELEMENTS, 1),
                                                                'https://metadata.datadrivendiscovery.org/types/TrueTarget'))
        self.assertEqual(predictions.metadata.query_column(1)['name'], 'petalWidth')
        self.assertEqual(predictions.metadata.query_column(1)['custom_metadata'], 'targets')

        self._test_return_new_metadata(predictions.metadata)

    def _test_return_new_metadata(self, predictions_metadata):
        expected_metadata = [{
            'selector': [],
            'metadata': {
                'structural_type': 'd3m.container.pandas.DataFrame',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Table'],
                'dimension': {
                    'name': 'rows',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                    'length': 150,
                },
                'schema': 'https://metadata.datadrivendiscovery.org/schemas/v0/container.json',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__'],
            'metadata': {
                'dimension': {
                    'name': 'columns',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                    'length': 2,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 0],
            'metadata': {
                'name': 'd3mIndex',
                'structural_type': 'int',
                'semantic_types': ['http://schema.org/Integer',
                                   'https://metadata.datadrivendiscovery.org/types/PrimaryKey'],
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {
                'structural_type': 'float',
                'name': 'petalWidth',
                'semantic_types': ['http://schema.org/Float',
                                   'https://metadata.datadrivendiscovery.org/types/Target',
                                   'https://metadata.datadrivendiscovery.org/types/PredictedTarget'],
                'custom_metadata': 'targets',
            },
        }]

        self.assertEqual(utils.to_json_structure(predictions_metadata.to_internal_simple_structure()), expected_metadata)

    def test_return_replace(self):
        dataframe, attributes, targets = self._get_iris_columns()

        hyperparams_class = \
            xgboost_regressor.XGBoostGBTreeRegressorPrimitive.metadata.query()['primitive_code'][
                'class_type_arguments']['Hyperparams']
        primitive = xgboost_regressor.XGBoostGBTreeRegressorPrimitive(
            hyperparams=hyperparams_class.defaults().replace({'return_result': 'replace'}))

        primitive.set_training_data(inputs=dataframe, outputs=dataframe)
        primitive.fit()

        predictions = primitive.produce(inputs=dataframe).value

        self.assertEqual(predictions.shape, (150, 3))
        self.assertTrue(predictions.metadata.has_semantic_type((metadata_base.ALL_ELEMENTS, 1),
                                                               'https://metadata.datadrivendiscovery.org/types/PredictedTarget'))
        self.assertFalse(predictions.metadata.has_semantic_type((metadata_base.ALL_ELEMENTS, 1),
                                                                'https://metadata.datadrivendiscovery.org/types/TrueTarget'))
        self.assertEqual(predictions.metadata.query_column(1)['name'], 'petalWidth')
        self.assertEqual(predictions.metadata.query_column(1)['custom_metadata'], 'targets')

        self._test_return_replace_metadata(predictions.metadata)

    def test_pickle_unpickle(self):
        dataframe, attributes, targets = self._get_iris_columns()

        hyperparams_class = \
            xgboost_regressor.XGBoostGBTreeRegressorPrimitive.metadata.query()['primitive_code'][
                'class_type_arguments'][
                'Hyperparams']
        primitive = xgboost_regressor.XGBoostGBTreeRegressorPrimitive(
            hyperparams=hyperparams_class.defaults().replace({'return_result': 'new', 'add_index_columns': False}))

        primitive.set_training_data(inputs=attributes, outputs=targets)
        primitive.fit()

        before_pickled_prediction = primitive.produce(inputs=attributes).value
        pickle_object = pickle.dumps(primitive)
        primitive = pickle.loads(pickle_object)
        after_unpickled_prediction = primitive.produce(inputs=attributes).value
        self.assertTrue(container.DataFrame.equals(before_pickled_prediction, after_unpickled_prediction))

    def _test_return_replace_metadata(self, predictions_metadata):
        self.assertEqual(utils.to_json_structure(predictions_metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                'structural_type': 'd3m.container.pandas.DataFrame',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Table'],
                'dimension': {
                    'name': 'rows',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                    'length': 150,
                },
                'schema': 'https://metadata.datadrivendiscovery.org/schemas/v0/container.json',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__'],
            'metadata': {
                'dimension': {
                    'name': 'columns',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                    'length': 3,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 0],
            'metadata': {
                'name': 'd3mIndex',
                'structural_type': 'int',
                'semantic_types': ['http://schema.org/Integer',
                                   'https://metadata.datadrivendiscovery.org/types/PrimaryKey'],
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {
                'structural_type': 'float',
                'name': 'petalWidth',
                'semantic_types': ['http://schema.org/Float',
                                   'https://metadata.datadrivendiscovery.org/types/Target',
                                   'https://metadata.datadrivendiscovery.org/types/PredictedTarget'],
                'custom_metadata': 'targets',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 2],
            'metadata': {
                'name': 'petalWidth',
                'structural_type': 'float',
                'semantic_types': ['http://schema.org/Float',
                                   'https://metadata.datadrivendiscovery.org/types/Target',
                                   'https://metadata.datadrivendiscovery.org/types/TrueTarget'],
                'custom_metadata': 'targets',
            },
        }])


if __name__ == '__main__':
    unittest.main()
