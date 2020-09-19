import os
import pickle
import unittest

import numpy as np

from d3m import container, utils
from d3m.metadata import base as metadata_base

from common_primitives import dataset_to_dataframe, extract_columns_semantic_types, lgbm_classifier, column_parser


def _add_categorical_col(attributes):
    rand_str = ['a', 'b', 'c', 'd', 'e']
    attributes = attributes.append_columns(container.DataFrame(data={
        'mock_cat_col': np.random.choice(rand_str, attributes.shape[0])
    }, generate_metadata=True))
    attributes.metadata = attributes.metadata.add_semantic_type([metadata_base.ALL_ELEMENTS, attributes.shape[-1] - 1],
                                                                'https://metadata.datadrivendiscovery.org/types/CategoricalData')
    attributes.metadata = attributes.metadata.add_semantic_type([metadata_base.ALL_ELEMENTS, attributes.shape[-1] - 1],
                                                                'https://metadata.datadrivendiscovery.org/types/Attribute')
    return attributes


def _get_iris():
    dataset_doc_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'iris_dataset_1', 'datasetDoc.json'))

    dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

    hyperparams_class = \
        dataset_to_dataframe.DatasetToDataFramePrimitive.metadata.query()['primitive_code']['class_type_arguments'][
            'Hyperparams']
    primitive = dataset_to_dataframe.DatasetToDataFramePrimitive(hyperparams=hyperparams_class.defaults())

    dataframe = primitive.produce(inputs=dataset).value
    return dataframe


def _get_iris_columns():
    dataframe = _get_iris()

    # We set custom metadata on columns.
    for column_index in range(1, 5):
        dataframe.metadata = dataframe.metadata.update_column(column_index, {'custom_metadata': 'attributes'})
    for column_index in range(5, 6):
        dataframe.metadata = dataframe.metadata.update_column(column_index, {'custom_metadata': 'targets'})

    # We set semantic types like runtime would.
    dataframe.metadata = dataframe.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, 5),
                                                              'https://metadata.datadrivendiscovery.org/types/Target')
    dataframe.metadata = dataframe.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, 5),
                                                              'https://metadata.datadrivendiscovery.org/types/TrueTarget')
    dataframe.metadata = dataframe.metadata.remove_semantic_type((metadata_base.ALL_ELEMENTS, 5),
                                                                 'https://metadata.datadrivendiscovery.org/types/Attribute')
    dataframe = _add_categorical_col(dataframe)

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
            {'semantic_types': ('https://metadata.datadrivendiscovery.org/types/SuggestedTarget',)}))
    targets = primitive.produce(inputs=dataframe).value

    return dataframe, attributes, targets


class LGBMTestCase(unittest.TestCase):
    attributes: container.DataFrame = None
    targets: container.DataFrame = None
    dataframe: container.DataFrame = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.dataframe, cls.attributes, cls.targets = _get_iris_columns()
        cls.excp_attributes = cls.attributes.copy()

    def test_single_target(self):
        self.assertEqual(list(self.targets.columns), ['species'])

        hyperparams_class = \
            lgbm_classifier.LightGBMClassifierPrimitive.metadata.query()['primitive_code']['class_type_arguments'][
                'Hyperparams']
        primitive = lgbm_classifier.LightGBMClassifierPrimitive(
            hyperparams=hyperparams_class.defaults().replace({'return_result': 'new', 'add_index_columns': False}))

        primitive.set_training_data(inputs=self.attributes, outputs=self.targets)
        primitive.fit()

        predictions = primitive.produce(inputs=self.attributes).value

        self.assertEqual(list(predictions.columns), ['species'])

        self.assertEqual(predictions.shape, (150, 1))
        self.assertEqual(predictions.iloc[0, 0], 'Iris-setosa')
        self.assertTrue(predictions.metadata.has_semantic_type((metadata_base.ALL_ELEMENTS, 0),
                                                               'https://metadata.datadrivendiscovery.org/types/PredictedTarget'))
        self.assertFalse(predictions.metadata.has_semantic_type((metadata_base.ALL_ELEMENTS, 0),
                                                                'https://metadata.datadrivendiscovery.org/types/TrueTarget'))
        self.assertEqual(predictions.metadata.query_column(0)['name'], 'species')
        self.assertEqual(predictions.metadata.query_column(0)['custom_metadata'], 'targets')

        self._test_single_target_metadata(predictions.metadata)

        samples = primitive.sample(inputs=self.attributes).value

        self.assertEqual(list(samples[0].columns), ['species'])

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].shape, (150, 1))
        self.assertEqual(samples[0].iloc[0, 0], 'Iris-setosa')
        self.assertTrue(samples[0].metadata.has_semantic_type((metadata_base.ALL_ELEMENTS, 0),
                                                              'https://metadata.datadrivendiscovery.org/types/PredictedTarget'))
        self.assertFalse(samples[0].metadata.has_semantic_type((metadata_base.ALL_ELEMENTS, 0),
                                                               'https://metadata.datadrivendiscovery.org/types/TrueTarget'))
        self.assertEqual(samples[0].metadata.query_column(0)['name'], 'species')
        self.assertEqual(samples[0].metadata.query_column(0)['custom_metadata'], 'targets')

        log_likelihoods = primitive.log_likelihoods(inputs=self.attributes, outputs=self.targets).value

        self.assertEqual(list(log_likelihoods.columns), ['species'])

        self.assertEqual(log_likelihoods.shape, (150, 1))
        self.assertEqual(log_likelihoods.metadata.query_column(0)['name'], 'species')

        log_likelihood = primitive.log_likelihood(inputs=self.attributes, outputs=self.targets).value

        self.assertEqual(list(log_likelihood.columns), ['species'])

        self.assertEqual(log_likelihood.shape, (1, 1))
        self.assertAlmostEqual(log_likelihood.iloc[0, 0], -6.338635478886032)
        self.assertEqual(log_likelihood.metadata.query_column(0)['name'], 'species')

    def test_single_target_continue_fit(self):
        hyperparams_class = \
            lgbm_classifier.LightGBMClassifierPrimitive.metadata.query()['primitive_code']['class_type_arguments'][
                'Hyperparams']
        primitive = lgbm_classifier.LightGBMClassifierPrimitive(
            hyperparams=hyperparams_class.defaults().replace({'return_result': 'new', 'add_index_columns': False}))

        primitive.set_training_data(inputs=self.attributes, outputs=self.targets)
        primitive.fit()
        # reset the training data to make continue_fit() work.
        primitive.set_training_data(inputs=self.attributes, outputs=self.targets)
        primitive.continue_fit()
        params = primitive.get_params()
        self.assertEqual(params['booster'].current_iteration(),
                         primitive.hyperparams['n_estimators'] + primitive.hyperparams['n_more_estimators'])
        predictions = primitive.produce(inputs=self.attributes).value

        self.assertEqual(predictions.shape, (150, 1))
        self.assertEqual(predictions.iloc[0, 0], 'Iris-setosa')
        self.assertTrue(predictions.metadata.has_semantic_type((metadata_base.ALL_ELEMENTS, 0),
                                                               'https://metadata.datadrivendiscovery.org/types/PredictedTarget'))
        self.assertFalse(predictions.metadata.has_semantic_type((metadata_base.ALL_ELEMENTS, 0),
                                                                'https://metadata.datadrivendiscovery.org/types/TrueTarget'))
        self.assertEqual(predictions.metadata.query_column(0)['name'], 'species')
        self.assertEqual(predictions.metadata.query_column(0)['custom_metadata'], 'targets')

        self._test_single_target_metadata(predictions.metadata)

        samples = primitive.sample(inputs=self.attributes).value

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].shape, (150, 1))
        self.assertEqual(samples[0].iloc[0, 0], 'Iris-setosa')
        self.assertTrue(samples[0].metadata.has_semantic_type((metadata_base.ALL_ELEMENTS, 0),
                                                              'https://metadata.datadrivendiscovery.org/types/PredictedTarget'))
        self.assertFalse(samples[0].metadata.has_semantic_type((metadata_base.ALL_ELEMENTS, 0),
                                                               'https://metadata.datadrivendiscovery.org/types/TrueTarget'))
        self.assertEqual(samples[0].metadata.query_column(0)['name'], 'species')
        self.assertEqual(samples[0].metadata.query_column(0)['custom_metadata'], 'targets')

        log_likelihoods = primitive.log_likelihoods(inputs=self.attributes, outputs=self.targets).value

        self.assertEqual(log_likelihoods.shape, (150, 1))
        self.assertEqual(log_likelihoods.metadata.query_column(0)['name'], 'species')

        log_likelihood = primitive.log_likelihood(inputs=self.attributes, outputs=self.targets).value

        self.assertEqual(log_likelihood.shape, (1, 1))
        self.assertAlmostEqual(log_likelihood.iloc[0, 0], -3.723258225143776)
        self.assertEqual(log_likelihoods.metadata.query_column(0)['name'], 'species')

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
                'structural_type': 'str',
                'name': 'species',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/CategoricalData',
                                   'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',
                                   'https://metadata.datadrivendiscovery.org/types/Target',
                                   'https://metadata.datadrivendiscovery.org/types/PredictedTarget'],
                'custom_metadata': 'targets',
            },
        }]

        self.assertEqual(utils.to_json_structure(predictions_metadata.to_internal_simple_structure()), expected_metadata)

    def test_semantic_types(self):
        # dataframe, attributes, targets = self._get_iris_columns()

        hyperparams_class = \
            lgbm_classifier.LightGBMClassifierPrimitive.metadata.query()['primitive_code']['class_type_arguments'][
                'Hyperparams']
        primitive = lgbm_classifier.LightGBMClassifierPrimitive(
            hyperparams=hyperparams_class.defaults().replace({'return_result': 'new', 'add_index_columns': False}))

        primitive.set_training_data(inputs=self.dataframe, outputs=self.dataframe)
        primitive.fit()

        predictions = primitive.produce(inputs=self.dataframe).value

        self.assertEqual(list(predictions.columns), ['species'])

        self.assertEqual(predictions.shape, (150, 1))
        self.assertEqual(predictions.iloc[0, 0], 'Iris-setosa')
        self.assertTrue(predictions.metadata.has_semantic_type((metadata_base.ALL_ELEMENTS, 0),
                                                               'https://metadata.datadrivendiscovery.org/types/PredictedTarget'))
        self.assertFalse(predictions.metadata.has_semantic_type((metadata_base.ALL_ELEMENTS, 0),
                                                                'https://metadata.datadrivendiscovery.org/types/TrueTarget'))
        self.assertEqual(predictions.metadata.query_column(0)['name'], 'species')
        self.assertEqual(predictions.metadata.query_column(0)['custom_metadata'], 'targets')

        samples = primitive.sample(inputs=self.dataframe).value
        self.assertEqual(list(samples[0].columns), ['species'])

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].shape, (150, 1))
        self.assertEqual(samples[0].iloc[0, 0], 'Iris-setosa')
        self.assertTrue(samples[0].metadata.has_semantic_type((metadata_base.ALL_ELEMENTS, 0),
                                                              'https://metadata.datadrivendiscovery.org/types/PredictedTarget'))
        self.assertFalse(samples[0].metadata.has_semantic_type((metadata_base.ALL_ELEMENTS, 0),
                                                               'https://metadata.datadrivendiscovery.org/types/TrueTarget'))
        self.assertEqual(samples[0].metadata.query_column(0)['name'], 'species')
        self.assertEqual(samples[0].metadata.query_column(0)['custom_metadata'], 'targets')

        log_likelihoods = primitive.log_likelihoods(inputs=self.dataframe, outputs=self.dataframe).value
        self.assertEqual(list(log_likelihoods.columns), ['species'])

        self.assertEqual(log_likelihoods.shape, (150, 1))
        self.assertEqual(log_likelihoods.metadata.query_column(0)['name'], 'species')

        log_likelihood = primitive.log_likelihood(inputs=self.dataframe, outputs=self.dataframe).value
        self.assertEqual(list(log_likelihood.columns), ['species'])

        self.assertEqual(log_likelihood.shape, (1, 1))
        self.assertAlmostEqual(log_likelihood.iloc[0, 0], -6.338635478886032)
        self.assertEqual(log_likelihoods.metadata.query_column(0)['name'], 'species')

        feature_importances = primitive.produce_feature_importances().value
        self.assertEqual(list(feature_importances),
                         ['sepalLength', 'sepalWidth', 'petalLength', 'petalWidth', 'mock_cat_col'])
        self.assertEqual(feature_importances.metadata.query_column(0)['name'], 'sepalLength')
        self.assertEqual(feature_importances.metadata.query_column(1)['name'], 'sepalWidth')
        self.assertEqual(feature_importances.metadata.query_column(2)['name'], 'petalLength')
        self.assertEqual(feature_importances.metadata.query_column(3)['name'], 'petalWidth')

        self.assertEqual(feature_importances.values.tolist(),
                         [[0.22740524781341107, 0.18513119533527697, 0.3323615160349854, 0.25510204081632654, 0.0]])

    def test_return_append(self):
        hyperparams_class = \
            lgbm_classifier.LightGBMClassifierPrimitive.metadata.query()['primitive_code']['class_type_arguments'][
                'Hyperparams']
        primitive = lgbm_classifier.LightGBMClassifierPrimitive(hyperparams=hyperparams_class.defaults())

        primitive.set_training_data(inputs=self.dataframe, outputs=self.dataframe)
        primitive.fit()

        predictions = primitive.produce(inputs=self.dataframe).value
        self.assertEqual(list(predictions.columns), [
            'd3mIndex',
            'sepalLength',
            'sepalWidth',
            'petalLength',
            'petalWidth',
            'species',
            'mock_cat_col',
            'species',
        ])
        self.assertEqual(predictions.shape, (150, 8))
        self.assertEqual(predictions.iloc[0, 7], 'Iris-setosa')
        self.assertTrue(predictions.metadata.has_semantic_type((metadata_base.ALL_ELEMENTS, 7),
                                                               'https://metadata.datadrivendiscovery.org/types/PredictedTarget'))
        self.assertFalse(predictions.metadata.has_semantic_type((metadata_base.ALL_ELEMENTS, 7),
                                                                'https://metadata.datadrivendiscovery.org/types/TrueTarget'))
        self.assertEqual(predictions.metadata.query_column(7)['name'], 'species')
        self.assertEqual(predictions.metadata.query_column(7)['custom_metadata'], 'targets')

        self._test_return_append_metadata(predictions.metadata)

    def _test_return_append_metadata(self, predictions_metadata):
        self.assertEqual(utils.to_json_structure(predictions_metadata.to_internal_simple_structure()), [{
            'metadata': {'dimension': {'length': 150,
                                       'name': 'rows',
                                       'semantic_types': [
                                           'https://metadata.datadrivendiscovery.org/types/TabularRow']},
                         'schema': 'https://metadata.datadrivendiscovery.org/schemas/v0/container.json',
                         'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Table'],
                         'structural_type': 'd3m.container.pandas.DataFrame'},
            'selector': []},
            {'metadata': {'dimension': {'length': 8,
                                        'name': 'columns',
                                        'semantic_types': [
                                            'https://metadata.datadrivendiscovery.org/types/TabularColumn']}},
             'selector': ['__ALL_ELEMENTS__']},
            {'metadata': {'name': 'd3mIndex',
                          'semantic_types': ['http://schema.org/Integer',
                                             'https://metadata.datadrivendiscovery.org/types/PrimaryKey'],
                          'structural_type': 'int'},
             'selector': ['__ALL_ELEMENTS__', 0]},
            {'metadata': {'custom_metadata': 'attributes',
                          'name': 'sepalLength',
                          'semantic_types': ['http://schema.org/Float',
                                             'https://metadata.datadrivendiscovery.org/types/Attribute'],
                          'structural_type': 'float'},
             'selector': ['__ALL_ELEMENTS__', 1]},
            {'metadata': {'custom_metadata': 'attributes',
                          'name': 'sepalWidth',
                          'semantic_types': ['http://schema.org/Float',
                                             'https://metadata.datadrivendiscovery.org/types/Attribute'],
                          'structural_type': 'float'},
             'selector': ['__ALL_ELEMENTS__', 2]},
            {'metadata': {'custom_metadata': 'attributes',
                          'name': 'petalLength',
                          'semantic_types': ['http://schema.org/Float',
                                             'https://metadata.datadrivendiscovery.org/types/Attribute'],
                          'structural_type': 'float'},
             'selector': ['__ALL_ELEMENTS__', 3]},
            {'metadata': {'custom_metadata': 'attributes',
                          'name': 'petalWidth',
                          'semantic_types': ['http://schema.org/Float',
                                             'https://metadata.datadrivendiscovery.org/types/Attribute'],
                          'structural_type': 'float'},
             'selector': ['__ALL_ELEMENTS__', 4]},
            {'metadata': {'custom_metadata': 'targets',
                          'name': 'species',
                          'semantic_types': [
                              'https://metadata.datadrivendiscovery.org/types/CategoricalData',
                              'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',
                              'https://metadata.datadrivendiscovery.org/types/Target',
                              'https://metadata.datadrivendiscovery.org/types/TrueTarget'],
                          'structural_type': 'str'},
             'selector': ['__ALL_ELEMENTS__', 5]},
            {'metadata': {'name': 'mock_cat_col',
                          'semantic_types': [
                              'https://metadata.datadrivendiscovery.org/types/CategoricalData',
                              'https://metadata.datadrivendiscovery.org/types/Attribute'],
                          'structural_type': 'int'},
             'selector': ['__ALL_ELEMENTS__', 6]},
            {'metadata': {'custom_metadata': 'targets',
                          'name': 'species',
                          'semantic_types': [
                              'https://metadata.datadrivendiscovery.org/types/CategoricalData',
                              'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',
                              'https://metadata.datadrivendiscovery.org/types/Target',
                              'https://metadata.datadrivendiscovery.org/types/PredictedTarget'],
                          'structural_type': 'str'},
             'selector': ['__ALL_ELEMENTS__', 7]}]
                         )

    def test_return_new(self):
        hyperparams_class = \
            lgbm_classifier.LightGBMClassifierPrimitive.metadata.query()['primitive_code']['class_type_arguments'][
                'Hyperparams']
        primitive = lgbm_classifier.LightGBMClassifierPrimitive(
            hyperparams=hyperparams_class.defaults().replace({'return_result': 'new'}))

        primitive.set_training_data(inputs=self.dataframe, outputs=self.dataframe)
        primitive.fit()

        predictions = primitive.produce(inputs=self.dataframe).value

        self.assertEqual(list(predictions.columns), [
            'd3mIndex',
            'species',
        ])

        self.assertEqual(predictions.shape, (150, 2))
        self.assertEqual(predictions.iloc[0, 1], 'Iris-setosa')
        self.assertTrue(predictions.metadata.has_semantic_type((metadata_base.ALL_ELEMENTS, 1),
                                                               'https://metadata.datadrivendiscovery.org/types/PredictedTarget'))
        self.assertFalse(predictions.metadata.has_semantic_type((metadata_base.ALL_ELEMENTS, 1),
                                                                'https://metadata.datadrivendiscovery.org/types/TrueTarget'))
        self.assertEqual(predictions.metadata.query_column(1)['name'], 'species')
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
                'structural_type': 'str',
                'name': 'species',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/CategoricalData',
                                   'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',
                                   'https://metadata.datadrivendiscovery.org/types/Target',
                                   'https://metadata.datadrivendiscovery.org/types/PredictedTarget'],
                'custom_metadata': 'targets',
            },
        }]

        self.assertEqual(utils.to_json_structure(predictions_metadata.to_internal_simple_structure()), expected_metadata)

    def test_return_replace(self):
        hyperparams_class = \
            lgbm_classifier.LightGBMClassifierPrimitive.metadata.query()['primitive_code']['class_type_arguments'][
                'Hyperparams']
        primitive = lgbm_classifier.LightGBMClassifierPrimitive(
            hyperparams=hyperparams_class.defaults().replace({'return_result': 'replace'}))

        primitive.set_training_data(inputs=self.dataframe, outputs=self.dataframe)
        primitive.fit()

        predictions = primitive.produce(inputs=self.dataframe).value
        self.assertEqual(list(predictions.columns), [
            'd3mIndex',
            'species',
            'species',
        ])
        self.assertEqual(predictions.shape, (150, 3))
        self.assertEqual(predictions.iloc[0, 1], 'Iris-setosa')
        self.assertTrue(predictions.metadata.has_semantic_type((metadata_base.ALL_ELEMENTS, 1),
                                                               'https://metadata.datadrivendiscovery.org/types/PredictedTarget'))
        self.assertFalse(predictions.metadata.has_semantic_type((metadata_base.ALL_ELEMENTS, 1),
                                                                'https://metadata.datadrivendiscovery.org/types/TrueTarget'))
        self.assertEqual(predictions.metadata.query_column(1)['name'], 'species')
        self.assertEqual(predictions.metadata.query_column(1)['custom_metadata'], 'targets')

        self._test_return_replace_metadata(predictions.metadata)

    def test_pickle_unpickle(self):
        hyperparams_class = \
            lgbm_classifier.LightGBMClassifierPrimitive.metadata.query()['primitive_code']['class_type_arguments'][
                'Hyperparams']
        primitive = lgbm_classifier.LightGBMClassifierPrimitive(
            hyperparams=hyperparams_class.defaults().replace({'return_result': 'new', 'add_index_columns': False}))

        primitive.set_training_data(inputs=self.attributes, outputs=self.targets)
        primitive.fit()

        before_pickled_prediction = primitive.produce(inputs=self.attributes).value
        pickle_object = pickle.dumps(primitive)
        primitive = pickle.loads(pickle_object)
        after_unpickled_prediction = primitive.produce(inputs=self.attributes).value
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
                'structural_type': 'str',
                'name': 'species',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/CategoricalData',
                                   'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',
                                   'https://metadata.datadrivendiscovery.org/types/Target',
                                   'https://metadata.datadrivendiscovery.org/types/PredictedTarget'],
                'custom_metadata': 'targets',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 2],
            'metadata': {
                'name': 'species',
                'structural_type': 'str',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/CategoricalData',
                                   'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',
                                   'https://metadata.datadrivendiscovery.org/types/Target',
                                   'https://metadata.datadrivendiscovery.org/types/TrueTarget'],
                'custom_metadata': 'targets',
            },
        }])


if __name__ == '__main__':
    unittest.main()
