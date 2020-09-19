import os
import time
import unittest
import numpy as np
import pickle
from d3m import container, exceptions, utils
from d3m.metadata import base as metadata_base

from common_primitives import dataset_to_dataframe, extract_columns_semantic_types, one_hot_maker, column_parser


def _copy_target_as_categorical_feature(attributes, targets):
    attributes = targets.append_columns(attributes)
    for column_name in targets.columns.values:
        column_mask = attributes.columns.get_loc(column_name)
        if isinstance(column_mask, int):
            column_index = column_mask
        else:
            column_index = np.where(column_mask)[0][-1].item()
        attributes.metadata = attributes.metadata.remove_semantic_type(
            (metadata_base.ALL_ELEMENTS, column_index),
            'https://metadata.datadrivendiscovery.org/types/SuggestedTarget')
        attributes.metadata = attributes.metadata.remove_semantic_type(
            (metadata_base.ALL_ELEMENTS, column_index),
            'https://metadata.datadrivendiscovery.org/types/Target')
        attributes.metadata = attributes.metadata.remove_semantic_type(
            (metadata_base.ALL_ELEMENTS, column_index),
            'https://metadata.datadrivendiscovery.org/types/TrueTarget')
        attributes.metadata = attributes.metadata.add_semantic_type(
            (metadata_base.ALL_ELEMENTS, column_index),
            'https://metadata.datadrivendiscovery.org/types/CategoricalData')
        attributes.metadata = attributes.metadata.add_semantic_type(
            (metadata_base.ALL_ELEMENTS, column_index),
            'https://metadata.datadrivendiscovery.org/types/Attribute')
        attributes.metadata = attributes.metadata.update_column(column_index,
                                                                {'custom_metadata': metadata_base.NO_VALUE})
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


class OneHotTestCase(unittest.TestCase):
    attributes: container.DataFrame = None
    excp_attributes: container.DataFrame = None
    targets: container.DataFrame = None
    dataframe: container.DataFrame = None
    unseen_species: str = 'Unseen-Species'
    missing_value: float = np.NaN

    @classmethod
    def setUpClass(cls) -> None:
        cls.dataframe, cls.attributes, cls.targets = _get_iris_columns()
        cls.attributes = _copy_target_as_categorical_feature(attributes=cls.attributes, targets=cls.targets)
        cls.excp_attributes = cls.attributes.copy()

    def tearDown(self):
        self.attributes.iloc[:3, 0] = 'Iris-setosa'
        self.excp_attributes.iloc[:3, 0] = 'Iris-setosa'

    def test_fit_produce(self):
        attributes = _copy_target_as_categorical_feature(self.attributes,
                                                         self.targets.rename(columns={'species': '2-species'}))
        attributes.metadata = attributes.metadata.update_column(1, {
            'name': '2-species'
        })

        hyperparams_class = \
            one_hot_maker.OneHotMakerPrimitive.metadata.query()['primitive_code']['class_type_arguments'][
                'Hyperparams']
        primitive = one_hot_maker.OneHotMakerPrimitive(
            hyperparams=hyperparams_class.defaults().replace({'return_result': 'replace'}))

        primitive.set_training_data(inputs=attributes)
        primitive.fit()
        after_onehot = primitive.produce(inputs=attributes).value
        # 1 for the original, so we remove it.
        self.assertEqual(after_onehot.shape[1], 2 * (len(self.targets['species'].unique()) - 1) + attributes.shape[1])
        self.assertEqual(after_onehot.shape[0], self.targets.shape[0])
        # 3 unique value for 2 (species, 2-species) 3 * 2 = 6
        self.assertTrue(all(dtype == 'uint8' for dtype in after_onehot.dtypes[:6]))
        self.assertEqual(list(after_onehot.columns.values), [
            'species.Iris-setosa', 'species.Iris-versicolor', 'species.Iris-virginica',
            '2-species.Iris-setosa', '2-species.Iris-versicolor', '2-species.Iris-virginica',
            'sepalLength', 'sepalWidth', 'petalLength', 'petalWidth'])
        self._test_metadata_return_replace(after_onehot.metadata)

    def test_error_unseen_categories_ignore(self):
        # default(ignore) case
        self.excp_attributes.iloc[0, 0] = self.unseen_species
        self.excp_attributes.iloc[1, 0] = self.unseen_species + '-2'
        self.excp_attributes.iloc[2, 0] = np.NaN
        hyperparams_class = \
            one_hot_maker.OneHotMakerPrimitive.metadata.query()['primitive_code']['class_type_arguments'][
                'Hyperparams']
        primitive = one_hot_maker.OneHotMakerPrimitive(
            hyperparams=hyperparams_class.defaults().replace({'return_result': 'replace'}))

        primitive.set_training_data(inputs=self.attributes)
        primitive.fit()
        one_hot_result = primitive.produce(inputs=self.excp_attributes).value
        self.assertEqual(one_hot_result.shape[1], len(self.targets['species'].unique()) + self.attributes.shape[1] - 1)
        self.assertEqual(one_hot_result.shape[0], self.targets.shape[0])
        self.assertTrue(all(dtype == 'uint8' for dtype in one_hot_result.dtypes[:3]))

    def test_error_unseen_categories_error(self):
        # error case
        self.excp_attributes.iloc[0, 0] = self.unseen_species
        self.excp_attributes.iloc[1, 0] = self.unseen_species + '-2'
        self.excp_attributes.iloc[2, 0] = np.NaN
        hyperparams_class = \
            one_hot_maker.OneHotMakerPrimitive.metadata.query()['primitive_code']['class_type_arguments'][
                'Hyperparams']
        primitive = one_hot_maker.OneHotMakerPrimitive(
            hyperparams=hyperparams_class.defaults().replace({'return_result': 'replace', 'handle_unseen': 'error'}))

        primitive.set_training_data(inputs=self.attributes)
        primitive.fit()
        self.assertRaises(exceptions.UnexpectedValueError, primitive.produce, inputs=self.excp_attributes)

    def test_unseen_categories_handle(self):
        # handle case
        self.excp_attributes.iloc[0, 0] = self.unseen_species
        self.excp_attributes.iloc[1, 0] = self.unseen_species + '-2'
        self.excp_attributes.iloc[2, 0] = np.NaN
        hyperparams_class = \
            one_hot_maker.OneHotMakerPrimitive.metadata.query()['primitive_code']['class_type_arguments'][
                'Hyperparams']
        primitive = one_hot_maker.OneHotMakerPrimitive(
            hyperparams=hyperparams_class.defaults().replace({'return_result': 'replace', 'handle_unseen': 'column'}))

        primitive.set_training_data(inputs=self.attributes)
        primitive.fit()
        one_hot_result = primitive.produce(inputs=self.excp_attributes).value
        self.assertEqual(one_hot_result.shape[1],
                         len(self.targets['species'].unique()) + self.attributes.shape[1] - 1 + 1)
        # unseen cell should be 1
        self.assertEqual(one_hot_result.iloc[0, 3], 1)
        self.assertEqual(one_hot_result.shape[0], self.targets.shape[0])
        self.assertTrue(all(dtype == 'uint8' for dtype in one_hot_result.dtypes[:3]))
        self.assertEqual(set(one_hot_result.columns.values), {'petalLength',
                                                              'petalWidth',
                                                              'sepalLength',
                                                              'sepalWidth',
                                                              'species.Iris-setosa',
                                                              'species.Iris-versicolor',
                                                              'species.Iris-virginica',
                                                              'species.Unseen'})
        self._test_metadata_unseen_handle_return_replace(one_hot_result.metadata)

    def test_missing_value_ignore(self):
        self.excp_attributes.iloc[0, 0] = self.missing_value
        self.excp_attributes.iloc[1, 0] = self.missing_value

        # missing present during fit
        hyperparams_class = \
            one_hot_maker.OneHotMakerPrimitive.metadata.query()['primitive_code']['class_type_arguments'][
                'Hyperparams']
        primitive = one_hot_maker.OneHotMakerPrimitive(
            hyperparams=hyperparams_class.defaults().replace({'return_result': 'replace'}))

        primitive.set_training_data(inputs=self.excp_attributes)
        primitive.fit()
        one_hot_result = primitive.produce(inputs=self.excp_attributes).value
        self.assertEqual(one_hot_result.shape[1], len(self.targets['species'].unique()) + self.attributes.shape[1] - 1)
        self.assertEqual(one_hot_result.shape[0], self.targets.shape[0])
        self.assertTrue(all(dtype == 'uint8' for dtype in one_hot_result.dtypes[:3]))
        self.assertEqual(set(one_hot_result.columns.values), {
            'species.Iris-setosa', 'species.Iris-versicolor', 'species.Iris-virginica',
            'sepalLength', 'sepalWidth', 'petalLength', 'petalWidth'})

        hyperparams_class = \
            one_hot_maker.OneHotMakerPrimitive.metadata.query()['primitive_code']['class_type_arguments'][
                'Hyperparams']
        primitive = one_hot_maker.OneHotMakerPrimitive(
            hyperparams=hyperparams_class.defaults().replace({'return_result': 'replace'}))

        primitive.set_training_data(inputs=self.attributes)
        primitive.fit()
        one_hot_result = primitive.produce(inputs=self.excp_attributes).value
        self.assertEqual(one_hot_result.shape[1], len(self.targets['species'].unique()) + self.attributes.shape[1] - 1)
        self.assertEqual(one_hot_result.shape[0], self.targets.shape[0])
        self.assertTrue(all(dtype == 'uint8' for dtype in one_hot_result.dtypes[:3]))
        self.assertEqual(set(one_hot_result.columns.values), {
            'species.Iris-setosa', 'species.Iris-versicolor', 'species.Iris-virginica',
            'sepalLength', 'sepalWidth', 'petalLength', 'petalWidth'})

    def test_missing_value_error(self):
        self.excp_attributes.iloc[0, 0] = np.NaN
        self.excp_attributes.iloc[1, 0] = None
        # error
        hyperparams_class = \
            one_hot_maker.OneHotMakerPrimitive.metadata.query()['primitive_code']['class_type_arguments'][
                'Hyperparams']
        primitive = one_hot_maker.OneHotMakerPrimitive(
            hyperparams=hyperparams_class.defaults().replace({
                'return_result': 'replace',
                'handle_missing_value': 'error',
            }))

        primitive.set_training_data(inputs=self.excp_attributes)
        self.assertRaises(exceptions.MissingValueError, primitive.fit)

    def test_missing_value_column(self):
        self.excp_attributes.iloc[0, 0] = np.NaN
        self.excp_attributes.iloc[1, 0] = np.NaN
        self.excp_attributes.iloc[2, 0] = 'Unseen-Species'
        # column
        hyperparams_class = \
            one_hot_maker.OneHotMakerPrimitive.metadata.query()['primitive_code']['class_type_arguments'][
                'Hyperparams']
        primitive = one_hot_maker.OneHotMakerPrimitive(
            hyperparams=hyperparams_class.defaults().replace({
                'return_result': 'replace',
                'handle_missing_value': 'column',
            }))

        primitive.set_training_data(inputs=self.attributes)
        primitive.fit()
        one_hot_result = primitive.produce(inputs=self.excp_attributes).value
        self.assertEqual(one_hot_result.shape[1],
                         len(self.targets['species'].unique()) + 1 + self.attributes.shape[1] - 1)
        self.assertEqual(one_hot_result.shape[0], self.targets.shape[0])
        self.assertTrue(all(dtype == 'uint8' for dtype in one_hot_result.dtypes[:4]))
        self.assertEqual(set(one_hot_result.columns.values), {'petalLength',
                                                              'petalWidth',
                                                              'sepalLength',
                                                              'sepalWidth',
                                                              'species.Iris-setosa',
                                                              'species.Iris-versicolor',
                                                              'species.Iris-virginica',
                                                              'species.Missing'})

    def test_unseen_column_and_missing_value_column(self):
        self.excp_attributes.iloc[0, 0] = np.NaN
        self.excp_attributes.iloc[1, 0] = np.NaN
        self.excp_attributes.iloc[2, 0] = 'Unseen-Species'
        # column
        hyperparams_class = \
            one_hot_maker.OneHotMakerPrimitive.metadata.query()['primitive_code']['class_type_arguments'][
                'Hyperparams']
        primitive = one_hot_maker.OneHotMakerPrimitive(
            hyperparams=hyperparams_class.defaults().replace({
                'return_result': 'replace',
                'handle_missing_value': 'column',
                'handle_unseen': 'column'
            }))

        primitive.set_training_data(inputs=self.attributes)
        primitive.fit()
        one_hot_result = primitive.produce(inputs=self.excp_attributes).value
        self.assertEqual(one_hot_result.shape[1],
                         len(self.targets['species'].unique()) + 2 + self.attributes.shape[1] - 1)
        self.assertEqual(one_hot_result.shape[0], self.targets.shape[0])
        self.assertTrue(all(dtype == 'uint8' for dtype in one_hot_result.dtypes[:4]))
        self.assertEqual(set(one_hot_result.columns.values), {'petalLength',
                                                              'petalWidth',
                                                              'sepalLength',
                                                              'sepalWidth',
                                                              'species.Iris-setosa',
                                                              'species.Iris-versicolor',
                                                              'species.Iris-virginica',
                                                              'species.Missing',
                                                              'species.Unseen'})

    def test_pickle_unpickle(self):
        hyperparams_class = \
            one_hot_maker.OneHotMakerPrimitive.metadata.query()['primitive_code']['class_type_arguments'][
                'Hyperparams']
        primitive = one_hot_maker.OneHotMakerPrimitive(
            hyperparams=hyperparams_class.defaults().replace({
                'return_result': 'replace',
                'handle_missing_value': 'column',
                'handle_unseen': 'column'
            }))

        primitive.set_training_data(inputs=self.attributes)
        primitive.fit()

        before_pickled_prediction = primitive.produce(inputs=self.attributes).value
        pickle_object = pickle.dumps(primitive)
        primitive = pickle.loads(pickle_object)
        after_unpickled_prediction = primitive.produce(inputs=self.attributes).value
        self.assertTrue(container.DataFrame.equals(before_pickled_prediction, after_unpickled_prediction))

    def _test_metadata_unseen_handle_return_replace(self, after_onehot_metadata):
        self.assertEqual(utils.to_json_structure(after_onehot_metadata.to_internal_simple_structure()), [{
            'metadata': {
                'dimension': {
                    'length': 150,
                    'name': 'rows',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow']
                },
                'schema': 'https://metadata.datadrivendiscovery.org/schemas/v0/container.json',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Table'],
                'structural_type': 'd3m.container.pandas.DataFrame'
            },
            'selector': []
        },
            {
                'metadata': {
                    'dimension': {
                        'length': 8,
                        'name': 'columns',
                        'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn']
                    }
                },
                'selector': ['__ALL_ELEMENTS__']
            },
            {
                'metadata': {
                    'custom_metadata': '__NO_VALUE__',
                    'name': 'species.Iris-setosa',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
                    'structural_type': 'numpy.uint8'
                },
                'selector': ['__ALL_ELEMENTS__', 0]
            },
            {
                'metadata': {
                    'custom_metadata': '__NO_VALUE__',
                    'name': 'species.Iris-versicolor',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
                    'structural_type': 'numpy.uint8'
                },
                'selector': ['__ALL_ELEMENTS__', 1]},
            {
                'metadata': {
                    'custom_metadata': '__NO_VALUE__',
                    'name': 'species.Iris-virginica',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
                    'structural_type': 'numpy.uint8'
                },
                'selector': ['__ALL_ELEMENTS__', 2]},
            {
                'metadata': {'custom_metadata': '__NO_VALUE__',
                             'name': 'species.Unseen',
                             'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
                             'structural_type': 'numpy.uint8'},
                'selector': ['__ALL_ELEMENTS__', 3]
            },
            {
                'metadata': {
                    'custom_metadata': 'attributes',
                    'name': 'sepalLength',
                    'semantic_types': ['http://schema.org/Float',
                                       'https://metadata.datadrivendiscovery.org/types/Attribute'
                                       ],
                    'structural_type': 'float'
                },
                'selector': ['__ALL_ELEMENTS__', 4]
            },
            {
                'metadata': {
                    'custom_metadata': 'attributes',
                    'name': 'sepalWidth',
                    'semantic_types': ['http://schema.org/Float',
                                       'https://metadata.datadrivendiscovery.org/types/Attribute'
                                       ],
                    'structural_type': 'float'
                },
                'selector': ['__ALL_ELEMENTS__', 5]
            },
            {
                'metadata': {
                    'custom_metadata': 'attributes',
                    'name': 'petalLength',
                    'semantic_types': ['http://schema.org/Float',
                                       'https://metadata.datadrivendiscovery.org/types/Attribute'
                                       ],
                    'structural_type': 'float'
                },
                'selector': ['__ALL_ELEMENTS__', 6]
            },
            {
                'metadata': {
                    'custom_metadata': 'attributes',
                    'name': 'petalWidth',
                    'semantic_types': ['http://schema.org/Float',
                                       'https://metadata.datadrivendiscovery.org/types/Attribute'
                                       ],
                    'structural_type': 'float'
                },
                'selector': ['__ALL_ELEMENTS__', 7]
            }
        ])

    def _test_metadata_return_replace(self, after_onehot_metadata):
        self.assertEqual(
            utils.to_json_structure(after_onehot_metadata.to_internal_simple_structure()),
            [{'metadata': {'dimension': {'length': 150,
                                         'name': 'rows',
                                         'semantic_types': [
                                             'https://metadata.datadrivendiscovery.org/types/TabularRow']},
                           'schema': 'https://metadata.datadrivendiscovery.org/schemas/v0/container.json',
                           'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Table'],
                           'structural_type': 'd3m.container.pandas.DataFrame'},
              'selector': []},
             {'metadata': {'dimension': {'length': 10,
                                         'name': 'columns',
                                         'semantic_types': [
                                             'https://metadata.datadrivendiscovery.org/types/TabularColumn']}},
              'selector': ['__ALL_ELEMENTS__']},
             {'metadata': {'custom_metadata': '__NO_VALUE__',
                           'name': 'species.Iris-setosa',
                           'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
                           'structural_type': 'numpy.uint8'},
              'selector': ['__ALL_ELEMENTS__', 0]},
             {'metadata': {'custom_metadata': '__NO_VALUE__',
                           'name': 'species.Iris-versicolor',
                           'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
                           'structural_type': 'numpy.uint8'},
              'selector': ['__ALL_ELEMENTS__', 1]},
             {'metadata': {'custom_metadata': '__NO_VALUE__',
                           'name': 'species.Iris-virginica',
                           'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
                           'structural_type': 'numpy.uint8'},
              'selector': ['__ALL_ELEMENTS__', 2]},
             {'metadata': {'custom_metadata': '__NO_VALUE__',
                           'name': '2-species.Iris-setosa',
                           'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
                           'structural_type': 'numpy.uint8'},
              'selector': ['__ALL_ELEMENTS__', 3]},
             {'metadata': {'custom_metadata': '__NO_VALUE__',
                           'name': '2-species.Iris-versicolor',
                           'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
                           'structural_type': 'numpy.uint8'},
              'selector': ['__ALL_ELEMENTS__', 4]},
             {'metadata': {'custom_metadata': '__NO_VALUE__',
                           'name': '2-species.Iris-virginica',
                           'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
                           'structural_type': 'numpy.uint8'},
              'selector': ['__ALL_ELEMENTS__', 5]},
             {'metadata': {'custom_metadata': 'attributes',
                           'name': 'sepalLength',
                           'semantic_types': ['http://schema.org/Float',
                                              'https://metadata.datadrivendiscovery.org/types/Attribute'],
                           'structural_type': 'float'},
              'selector': ['__ALL_ELEMENTS__', 6]},
             {'metadata': {'custom_metadata': 'attributes',
                           'name': 'sepalWidth',
                           'semantic_types': ['http://schema.org/Float',
                                              'https://metadata.datadrivendiscovery.org/types/Attribute'],
                           'structural_type': 'float'},
              'selector': ['__ALL_ELEMENTS__', 7]},
             {'metadata': {'custom_metadata': 'attributes',
                           'name': 'petalLength',
                           'semantic_types': ['http://schema.org/Float',
                                              'https://metadata.datadrivendiscovery.org/types/Attribute'],
                           'structural_type': 'float'},
              'selector': ['__ALL_ELEMENTS__', 8]},
             {'metadata': {'custom_metadata': 'attributes',
                           'name': 'petalWidth',
                           'semantic_types': ['http://schema.org/Float',
                                              'https://metadata.datadrivendiscovery.org/types/Attribute'],
                           'structural_type': 'float'},
              'selector': ['__ALL_ELEMENTS__', 9]}]
        )


if __name__ == '__main__':
    unittest.main()
