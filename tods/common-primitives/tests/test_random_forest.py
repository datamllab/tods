import logging
import os
import pickle
import unittest

from d3m import container, utils
from d3m.metadata import base as metadata_base

from common_primitives import dataset_to_dataframe, extract_columns_semantic_types, random_forest, column_parser


class RandomForestTestCase(unittest.TestCase):
    def _get_iris(self):
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'iris_dataset_1', 'datasetDoc.json'))

        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        hyperparams_class = dataset_to_dataframe.DatasetToDataFramePrimitive.metadata.get_hyperparams()
        primitive = dataset_to_dataframe.DatasetToDataFramePrimitive(hyperparams=hyperparams_class.defaults())

        dataframe = primitive.produce(inputs=dataset).value

        return dataframe

    def _get_iris_columns(self):
        dataframe = self._get_iris()

        # We set custom metadata on columns.
        for column_index in range(1, 5):
            dataframe.metadata = dataframe.metadata.update_column(column_index, {'custom_metadata': 'attributes'})
        for column_index in range(5, 6):
            dataframe.metadata = dataframe.metadata.update_column(column_index, {'custom_metadata': 'targets'})

        # We set semantic types like runtime would.
        dataframe.metadata = dataframe.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, 5), 'https://metadata.datadrivendiscovery.org/types/Target')
        dataframe.metadata = dataframe.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, 5), 'https://metadata.datadrivendiscovery.org/types/TrueTarget')
        dataframe.metadata = dataframe.metadata.remove_semantic_type((metadata_base.ALL_ELEMENTS, 5), 'https://metadata.datadrivendiscovery.org/types/Attribute')

        # Parsing.
        hyperparams_class = column_parser.ColumnParserPrimitive.metadata.get_hyperparams()
        primitive = column_parser.ColumnParserPrimitive(hyperparams=hyperparams_class.defaults())
        dataframe = primitive.produce(inputs=dataframe).value

        hyperparams_class = extract_columns_semantic_types.ExtractColumnsBySemanticTypesPrimitive.metadata.get_hyperparams()

        primitive = extract_columns_semantic_types.ExtractColumnsBySemanticTypesPrimitive(hyperparams=hyperparams_class.defaults().replace({'semantic_types': ('https://metadata.datadrivendiscovery.org/types/Attribute',)}))
        attributes = primitive.produce(inputs=dataframe).value

        primitive = extract_columns_semantic_types.ExtractColumnsBySemanticTypesPrimitive(hyperparams=hyperparams_class.defaults().replace({'semantic_types': ('https://metadata.datadrivendiscovery.org/types/SuggestedTarget',)}))
        targets = primitive.produce(inputs=dataframe).value

        return dataframe, attributes, targets

    def test_single_target(self):
        dataframe, attributes, targets = self._get_iris_columns()

        self.assertEqual(list(targets.columns), ['species'])

        hyperparams_class = random_forest.RandomForestClassifierPrimitive.metadata.get_hyperparams()
        primitive = random_forest.RandomForestClassifierPrimitive(hyperparams=hyperparams_class.defaults().replace({'return_result': 'new', 'add_index_columns': False}))

        primitive.set_training_data(inputs=attributes, outputs=targets)
        primitive.fit()

        predictions = primitive.produce(inputs=attributes).value

        self.assertEqual(list(predictions.columns), ['species'])

        self.assertEqual(predictions.shape, (150, 1))
        self.assertEqual(predictions.iloc[0, 0], 'Iris-setosa')
        self.assertTrue(predictions.metadata.has_semantic_type((metadata_base.ALL_ELEMENTS, 0), 'https://metadata.datadrivendiscovery.org/types/PredictedTarget'))
        self.assertFalse(predictions.metadata.has_semantic_type((metadata_base.ALL_ELEMENTS, 0), 'https://metadata.datadrivendiscovery.org/types/TrueTarget'))
        self.assertEqual(predictions.metadata.query_column(0)['name'], 'species')
        self.assertEqual(predictions.metadata.query_column(0)['custom_metadata'], 'targets')

        self._test_single_target_metadata(predictions.metadata)

        samples = primitive.sample(inputs=attributes).value

        self.assertEqual(list(samples[0].columns), ['species'])

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].shape, (150, 1))
        self.assertEqual(samples[0].iloc[0, 0], 'Iris-setosa')
        self.assertTrue(samples[0].metadata.has_semantic_type((metadata_base.ALL_ELEMENTS, 0), 'https://metadata.datadrivendiscovery.org/types/PredictedTarget'))
        self.assertFalse(samples[0].metadata.has_semantic_type((metadata_base.ALL_ELEMENTS, 0), 'https://metadata.datadrivendiscovery.org/types/TrueTarget'))
        self.assertEqual(samples[0].metadata.query_column(0)['name'], 'species')
        self.assertEqual(samples[0].metadata.query_column(0)['custom_metadata'], 'targets')

        log_likelihoods = primitive.log_likelihoods(inputs=attributes, outputs=targets).value

        self.assertEqual(list(log_likelihoods.columns), ['species'])

        self.assertEqual(log_likelihoods.shape, (150, 1))
        self.assertEqual(log_likelihoods.metadata.query_column(0)['name'], 'species')

        log_likelihood = primitive.log_likelihood(inputs=attributes, outputs=targets).value

        self.assertEqual(list(log_likelihood.columns), ['species'])

        self.assertEqual(log_likelihood.shape, (1, 1))
        self.assertAlmostEqual(log_likelihood.iloc[0, 0], -3.72702785304761)
        self.assertEqual(log_likelihoods.metadata.query_column(0)['name'], 'species')

        feature_importances = primitive.produce_feature_importances().value

        self.assertEqual(list(feature_importances), ['sepalLength', 'sepalWidth', 'petalLength', 'petalWidth'])
        self.assertEqual(feature_importances.metadata.query_column(0)['name'], 'sepalLength')
        self.assertEqual(feature_importances.metadata.query_column(1)['name'], 'sepalWidth')
        self.assertEqual(feature_importances.metadata.query_column(2)['name'], 'petalLength')
        self.assertEqual(feature_importances.metadata.query_column(3)['name'], 'petalWidth')

        self.assertEqual(feature_importances.values.tolist(), [[0.09090795402103087,
            0.024531041234715757,
            0.46044473961715215,
            0.42411626512710127,
        ]])

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
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/CategoricalData', 'https://metadata.datadrivendiscovery.org/types/SuggestedTarget', 'https://metadata.datadrivendiscovery.org/types/Target', 'https://metadata.datadrivendiscovery.org/types/PredictedTarget'],
                'custom_metadata': 'targets',
            },
        }]

        self.assertEqual(utils.to_json_structure(predictions_metadata.to_internal_simple_structure()), expected_metadata)

    def test_multiple_targets(self):
        dataframe, attributes, targets = self._get_iris_columns()

        targets = targets.append_columns(targets)

        self.assertEqual(list(targets.columns), ['species', 'species'])

        hyperparams_class = random_forest.RandomForestClassifierPrimitive.metadata.get_hyperparams()
        primitive = random_forest.RandomForestClassifierPrimitive(hyperparams=hyperparams_class.defaults().replace({'return_result': 'new', 'add_index_columns': False}))

        primitive.set_training_data(inputs=attributes, outputs=targets)
        primitive.fit()

        predictions = primitive.produce(inputs=attributes).value

        self.assertEqual(list(predictions.columns), ['species', 'species'])

        self.assertEqual(predictions.shape, (150, 2))
        for column_index in range(2):
            self.assertEqual(predictions.iloc[0, column_index], 'Iris-setosa')
            self.assertTrue(predictions.metadata.has_semantic_type((metadata_base.ALL_ELEMENTS, column_index), 'https://metadata.datadrivendiscovery.org/types/PredictedTarget'))
            self.assertFalse(predictions.metadata.has_semantic_type((metadata_base.ALL_ELEMENTS, column_index), 'https://metadata.datadrivendiscovery.org/types/TrueTarget'))
            self.assertEqual(predictions.metadata.query_column(column_index)['name'], 'species')
            self.assertEqual(predictions.metadata.query_column(column_index)['custom_metadata'], 'targets')

        samples = primitive.sample(inputs=attributes).value

        self.assertEqual(list(samples[0].columns), ['species', 'species'])

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].shape, (150, 2))
        for column_index in range(2):
            self.assertEqual(samples[0].iloc[0, column_index], 'Iris-setosa')
            self.assertTrue(samples[0].metadata.has_semantic_type((metadata_base.ALL_ELEMENTS, column_index), 'https://metadata.datadrivendiscovery.org/types/PredictedTarget'))
            self.assertFalse(samples[0].metadata.has_semantic_type((metadata_base.ALL_ELEMENTS, column_index), 'https://metadata.datadrivendiscovery.org/types/TrueTarget'))
            self.assertEqual(samples[0].metadata.query_column(column_index)['name'], 'species')
            self.assertEqual(samples[0].metadata.query_column(column_index)['custom_metadata'], 'targets')

        log_likelihoods = primitive.log_likelihoods(inputs=attributes, outputs=targets).value

        self.assertEqual(list(log_likelihoods.columns), ['species', 'species'])

        self.assertEqual(log_likelihoods.shape, (150, 2))
        for column_index in range(2):
            self.assertEqual(log_likelihoods.metadata.query_column(column_index)['name'], 'species')

        log_likelihood = primitive.log_likelihood(inputs=attributes, outputs=targets).value

        self.assertEqual(list(log_likelihood.columns), ['species', 'species'])

        self.assertEqual(log_likelihood.shape, (1, 2))
        for column_index in range(2):
            self.assertAlmostEqual(log_likelihood.iloc[0, column_index], -3.72702785304761)
            self.assertEqual(log_likelihoods.metadata.query_column(column_index)['name'], 'species')

        feature_importances = primitive.produce_feature_importances().value

        self.assertEqual(list(feature_importances), ['sepalLength', 'sepalWidth', 'petalLength', 'petalWidth'])
        self.assertEqual(feature_importances.metadata.query_column(0)['name'], 'sepalLength')
        self.assertEqual(feature_importances.metadata.query_column(1)['name'], 'sepalWidth')
        self.assertEqual(feature_importances.metadata.query_column(2)['name'], 'petalLength')
        self.assertEqual(feature_importances.metadata.query_column(3)['name'], 'petalWidth')

        self.assertEqual(feature_importances.values.tolist(), [[0.09090795402103087,
            0.024531041234715757,
            0.46044473961715215,
            0.42411626512710127,
        ]])

    def test_semantic_types(self):
        dataframe, attributes, targets = self._get_iris_columns()

        hyperparams_class = random_forest.RandomForestClassifierPrimitive.metadata.get_hyperparams()
        primitive = random_forest.RandomForestClassifierPrimitive(hyperparams=hyperparams_class.defaults().replace({'return_result': 'new', 'add_index_columns': False}))

        primitive.set_training_data(inputs=dataframe, outputs=dataframe)
        primitive.fit()

        predictions = primitive.produce(inputs=dataframe).value

        self.assertEqual(list(predictions.columns), ['species'])

        self.assertEqual(predictions.shape, (150, 1))
        self.assertEqual(predictions.iloc[0, 0], 'Iris-setosa')
        self.assertTrue(predictions.metadata.has_semantic_type((metadata_base.ALL_ELEMENTS, 0), 'https://metadata.datadrivendiscovery.org/types/PredictedTarget'))
        self.assertFalse(predictions.metadata.has_semantic_type((metadata_base.ALL_ELEMENTS, 0), 'https://metadata.datadrivendiscovery.org/types/TrueTarget'))
        self.assertEqual(predictions.metadata.query_column(0)['name'], 'species')
        self.assertEqual(predictions.metadata.query_column(0)['custom_metadata'], 'targets')

        samples = primitive.sample(inputs=dataframe).value

        self.assertEqual(list(samples[0].columns), ['species'])

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].shape, (150, 1))
        self.assertEqual(samples[0].iloc[0, 0], 'Iris-setosa')
        self.assertTrue(samples[0].metadata.has_semantic_type((metadata_base.ALL_ELEMENTS, 0), 'https://metadata.datadrivendiscovery.org/types/PredictedTarget'))
        self.assertFalse(samples[0].metadata.has_semantic_type((metadata_base.ALL_ELEMENTS, 0), 'https://metadata.datadrivendiscovery.org/types/TrueTarget'))
        self.assertEqual(samples[0].metadata.query_column(0)['name'], 'species')
        self.assertEqual(samples[0].metadata.query_column(0)['custom_metadata'], 'targets')

        log_likelihoods = primitive.log_likelihoods(inputs=dataframe, outputs=dataframe).value

        self.assertEqual(list(log_likelihoods.columns), ['species'])

        self.assertEqual(log_likelihoods.shape, (150, 1))
        self.assertEqual(log_likelihoods.metadata.query_column(0)['name'], 'species')

        log_likelihood = primitive.log_likelihood(inputs=dataframe, outputs=dataframe).value

        self.assertEqual(list(log_likelihood.columns), ['species'])

        self.assertEqual(log_likelihood.shape, (1, 1))
        self.assertAlmostEqual(log_likelihood.iloc[0, 0], -3.72702785304761)
        self.assertEqual(log_likelihoods.metadata.query_column(0)['name'], 'species')

        feature_importances = primitive.produce_feature_importances().value

        self.assertEqual(list(feature_importances), ['sepalLength', 'sepalWidth', 'petalLength', 'petalWidth'])
        self.assertEqual(feature_importances.metadata.query_column(0)['name'], 'sepalLength')
        self.assertEqual(feature_importances.metadata.query_column(1)['name'], 'sepalWidth')
        self.assertEqual(feature_importances.metadata.query_column(2)['name'], 'petalLength')
        self.assertEqual(feature_importances.metadata.query_column(3)['name'], 'petalWidth')

        self.assertEqual(feature_importances.values.tolist(), [[0.09090795402103087,
            0.024531041234715757,
            0.46044473961715215,
            0.42411626512710127,
        ]])

    def test_return_append(self):
        dataframe, attributes, targets = self._get_iris_columns()

        hyperparams_class = random_forest.RandomForestClassifierPrimitive.metadata.get_hyperparams()
        primitive = random_forest.RandomForestClassifierPrimitive(hyperparams=hyperparams_class.defaults())

        primitive.set_training_data(inputs=dataframe, outputs=dataframe)
        primitive.fit()

        predictions = primitive.produce(inputs=dataframe).value

        self.assertEqual(list(predictions.columns), [
            'd3mIndex',
            'sepalLength',
            'sepalWidth',
            'petalLength',
            'petalWidth',
            'species',
            'species',
        ])

        self.assertEqual(predictions.shape, (150, 7))
        self.assertEqual(predictions.iloc[0, 6], 'Iris-setosa')
        self.assertTrue(predictions.metadata.has_semantic_type((metadata_base.ALL_ELEMENTS, 6), 'https://metadata.datadrivendiscovery.org/types/PredictedTarget'))
        self.assertFalse(predictions.metadata.has_semantic_type((metadata_base.ALL_ELEMENTS, 6), 'https://metadata.datadrivendiscovery.org/types/TrueTarget'))
        self.assertEqual(predictions.metadata.query_column(6)['name'], 'species')
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
                'semantic_types': ['http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/PrimaryKey'],
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {
                'name': 'sepalLength',
                'structural_type': 'float',
                'semantic_types': ['http://schema.org/Float', 'https://metadata.datadrivendiscovery.org/types/Attribute'],
                'custom_metadata': 'attributes',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 2],
            'metadata': {
                'name': 'sepalWidth',
                'structural_type': 'float',
                'semantic_types': ['http://schema.org/Float', 'https://metadata.datadrivendiscovery.org/types/Attribute'],
                'custom_metadata': 'attributes',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 3],
            'metadata': {
                'name': 'petalLength',
                'structural_type': 'float',
                'semantic_types': ['http://schema.org/Float', 'https://metadata.datadrivendiscovery.org/types/Attribute'],
                'custom_metadata': 'attributes',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 4],
            'metadata': {
                'name': 'petalWidth',
                'structural_type': 'float',
                'semantic_types': ['http://schema.org/Float', 'https://metadata.datadrivendiscovery.org/types/Attribute'],
                'custom_metadata': 'attributes',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 5],
            'metadata': {
                'name': 'species',
                'structural_type': 'str',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/CategoricalData', 'https://metadata.datadrivendiscovery.org/types/SuggestedTarget', 'https://metadata.datadrivendiscovery.org/types/Target', 'https://metadata.datadrivendiscovery.org/types/TrueTarget'],
                'custom_metadata': 'targets',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 6],
            'metadata': {
                'structural_type': 'str',
                'name': 'species',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/CategoricalData', 'https://metadata.datadrivendiscovery.org/types/SuggestedTarget', 'https://metadata.datadrivendiscovery.org/types/Target', 'https://metadata.datadrivendiscovery.org/types/PredictedTarget'],
                'custom_metadata': 'targets',
            },
        }])

    def test_return_new(self):
        dataframe, attributes, targets = self._get_iris_columns()

        hyperparams_class = random_forest.RandomForestClassifierPrimitive.metadata.get_hyperparams()
        primitive = random_forest.RandomForestClassifierPrimitive(hyperparams=hyperparams_class.defaults().replace({'return_result': 'new'}))

        primitive.set_training_data(inputs=dataframe, outputs=dataframe)
        primitive.fit()

        predictions = primitive.produce(inputs=dataframe).value

        self.assertEqual(list(predictions.columns), [
            'd3mIndex',
            'species',
        ])

        self.assertEqual(predictions.shape, (150, 2))
        self.assertEqual(predictions.iloc[0, 1], 'Iris-setosa')
        self.assertTrue(predictions.metadata.has_semantic_type((metadata_base.ALL_ELEMENTS, 1), 'https://metadata.datadrivendiscovery.org/types/PredictedTarget'))
        self.assertFalse(predictions.metadata.has_semantic_type((metadata_base.ALL_ELEMENTS, 1), 'https://metadata.datadrivendiscovery.org/types/TrueTarget'))
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
                'semantic_types': ['http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/PrimaryKey'],
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {
                'structural_type': 'str',
                'name': 'species',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/CategoricalData', 'https://metadata.datadrivendiscovery.org/types/SuggestedTarget', 'https://metadata.datadrivendiscovery.org/types/Target', 'https://metadata.datadrivendiscovery.org/types/PredictedTarget'],
                'custom_metadata': 'targets',
            },
        }]

        self.assertEqual(utils.to_json_structure(predictions_metadata.to_internal_simple_structure()), expected_metadata)

    def test_return_replace(self):
        dataframe, attributes, targets = self._get_iris_columns()

        hyperparams_class = random_forest.RandomForestClassifierPrimitive.metadata.get_hyperparams()
        primitive = random_forest.RandomForestClassifierPrimitive(hyperparams=hyperparams_class.defaults().replace({'return_result': 'replace'}))

        primitive.set_training_data(inputs=dataframe, outputs=dataframe)
        primitive.fit()

        predictions = primitive.produce(inputs=dataframe).value

        self.assertEqual(list(predictions.columns), [
            'd3mIndex',
            'species',
            'species',
        ])

        self.assertEqual(predictions.shape, (150, 3))
        self.assertEqual(predictions.iloc[0, 1], 'Iris-setosa')
        self.assertTrue(predictions.metadata.has_semantic_type((metadata_base.ALL_ELEMENTS, 1), 'https://metadata.datadrivendiscovery.org/types/PredictedTarget'))
        self.assertFalse(predictions.metadata.has_semantic_type((metadata_base.ALL_ELEMENTS, 1), 'https://metadata.datadrivendiscovery.org/types/TrueTarget'))
        self.assertEqual(predictions.metadata.query_column(1)['name'], 'species')
        self.assertEqual(predictions.metadata.query_column(1)['custom_metadata'], 'targets')

        self._test_return_replace_metadata(predictions.metadata)

    def test_get_set_params(self):
        dataframe, attributes, targets = self._get_iris_columns()

        hyperparams_class = random_forest.RandomForestClassifierPrimitive.metadata.get_hyperparams()
        primitive = random_forest.RandomForestClassifierPrimitive(
            hyperparams=hyperparams_class.defaults().replace({'return_result': 'new', 'add_index_columns': False}))

        primitive.set_training_data(inputs=attributes, outputs=targets)
        primitive.fit()

        before_set_prediction = primitive.produce(inputs=attributes).value
        params = primitive.get_params()
        primitive.set_params(params=params)
        after_set_prediction = primitive.produce(inputs=attributes).value
        self.assertTrue(container.DataFrame.equals(before_set_prediction, after_set_prediction))

    def test_pickle_unpickle(self):
        dataframe, attributes, targets = self._get_iris_columns()

        hyperparams_class = random_forest.RandomForestClassifierPrimitive.metadata.get_hyperparams()
        primitive = random_forest.RandomForestClassifierPrimitive(
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
                'semantic_types': ['http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/PrimaryKey'],
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {
                'structural_type': 'str',
                'name': 'species',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/CategoricalData', 'https://metadata.datadrivendiscovery.org/types/SuggestedTarget', 'https://metadata.datadrivendiscovery.org/types/Target', 'https://metadata.datadrivendiscovery.org/types/PredictedTarget'],
                'custom_metadata': 'targets',
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 2],
            'metadata': {
                'name': 'species',
                'structural_type': 'str',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/CategoricalData', 'https://metadata.datadrivendiscovery.org/types/SuggestedTarget', 'https://metadata.datadrivendiscovery.org/types/Target', 'https://metadata.datadrivendiscovery.org/types/TrueTarget'],
                'custom_metadata': 'targets',
            },
        }])

    def test_empty_data(self):
        dataframe, attributes, targets = self._get_iris_columns()

        hyperparams_class = random_forest.RandomForestClassifierPrimitive.metadata.get_hyperparams()
        primitive = random_forest.RandomForestClassifierPrimitive(hyperparams=hyperparams_class.defaults())

        just_index_dataframe = dataframe.select_columns([0])
        no_attributes_dataframe = dataframe.select_columns([0, 5])

        primitive.set_training_data(inputs=just_index_dataframe, outputs=just_index_dataframe)

        with self.assertRaises(Exception):
            primitive.fit()

        primitive.set_training_data(inputs=no_attributes_dataframe, outputs=no_attributes_dataframe)

        with self.assertRaises(Exception):
            primitive.fit()

        primitive = random_forest.RandomForestClassifierPrimitive(hyperparams=hyperparams_class.defaults().replace({
            'error_on_no_columns': False,
            'return_result': 'replace',
        }))

        primitive.set_training_data(inputs=just_index_dataframe, outputs=just_index_dataframe)

        with self.assertLogs(primitive.logger, level=logging.WARNING) as cm:
            primitive.fit()

        self.assertEqual(len(cm.records), 2)
        self.assertEqual(cm.records[0].msg, "No inputs columns.")
        self.assertEqual(cm.records[1].msg, "No outputs columns.")

        # Test pickling.
        pickle_object = pickle.dumps(primitive)
        pickle.loads(pickle_object)

        with self.assertLogs(primitive.logger, level=logging.WARNING) as cm:
            predictions = primitive.produce(inputs=just_index_dataframe).value

        self.assertEqual(len(cm.records), 1)
        self.assertEqual(cm.records[0].msg, "No inputs columns.")

        self.assertEqual(list(predictions.columns), [
            'd3mIndex',
        ])
        self.assertEqual(predictions.shape, (150, 1))

        self.assertEqual(predictions.metadata.to_internal_json_structure(), just_index_dataframe.metadata.to_internal_json_structure())

        primitive = random_forest.RandomForestClassifierPrimitive(hyperparams=hyperparams_class.defaults().replace({
            'error_on_no_columns': False,
            'return_result': 'replace',
        }))

        primitive.set_training_data(inputs=no_attributes_dataframe, outputs=no_attributes_dataframe)

        with self.assertLogs(primitive.logger, level=logging.WARNING) as cm:
            primitive.fit()

        self.assertEqual(len(cm.records), 1)
        self.assertEqual(cm.records[0].msg, "No inputs columns.")

        # Test pickling.
        pickle_object = pickle.dumps(primitive)
        pickle.loads(pickle_object)

        with self.assertLogs(primitive.logger, level=logging.WARNING) as cm:
            predictions = primitive.produce(inputs=no_attributes_dataframe).value

        self.assertEqual(len(cm.records), 1)
        self.assertEqual(cm.records[0].msg, "No inputs columns.")

        self.assertEqual(list(predictions.columns), [
            'd3mIndex',
            'species',
        ])
        self.assertEqual(predictions.shape, (150, 2))

        self.assertEqual(predictions.metadata.to_internal_json_structure(), no_attributes_dataframe.metadata.to_internal_json_structure())

        primitive = random_forest.RandomForestClassifierPrimitive(hyperparams=hyperparams_class.defaults().replace({
            'error_on_no_columns': False,
            'return_result': 'new',
        }))

        primitive.set_training_data(inputs=no_attributes_dataframe, outputs=no_attributes_dataframe)

        with self.assertLogs(primitive.logger, level=logging.WARNING) as cm:
            primitive.fit()

        self.assertEqual(len(cm.records), 1)
        self.assertEqual(cm.records[0].msg, "No inputs columns.")

        # Test pickling.
        pickle_object = pickle.dumps(primitive)
        pickle.loads(pickle_object)

        with self.assertLogs(primitive.logger, level=logging.WARNING) as cm:
            with self.assertRaises(ValueError):
                primitive.produce(inputs=no_attributes_dataframe)

        self.assertEqual(len(cm.records), 1)
        self.assertEqual(cm.records[0].msg, "No inputs columns.")

        primitive = random_forest.RandomForestClassifierPrimitive(hyperparams=hyperparams_class.defaults().replace({
            'error_on_no_columns': False,
            'return_result': 'append',
        }))

        primitive.set_training_data(inputs=no_attributes_dataframe, outputs=no_attributes_dataframe)

        with self.assertLogs(primitive.logger, level=logging.WARNING) as cm:
            primitive.fit()

        # Test pickling.
        pickle_object = pickle.dumps(primitive)
        pickle.loads(pickle_object)

        self.assertEqual(len(cm.records), 1)
        self.assertEqual(cm.records[0].msg, "No inputs columns.")

        with self.assertLogs(primitive.logger, level=logging.WARNING) as cm:
            predictions = primitive.produce(inputs=no_attributes_dataframe).value

        self.assertEqual(len(cm.records), 1)
        self.assertEqual(cm.records[0].msg, "No inputs columns.")

        self.assertEqual(list(predictions.columns), [
            'd3mIndex',
            'species',
        ])
        self.assertEqual(predictions.shape, (150, 2))

        self.assertEqual(predictions.metadata.to_internal_json_structure(), no_attributes_dataframe.metadata.to_internal_json_structure())


if __name__ == '__main__':
    unittest.main()
