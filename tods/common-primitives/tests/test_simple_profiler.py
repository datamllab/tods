import os.path
import pickle
import unittest

from d3m import container
from d3m.metadata import base as metadata_base

from common_primitives import dataset_to_dataframe, simple_profiler, train_score_split


class SimpleProfilerPrimitiveTestCase(unittest.TestCase):
    def _get_iris(self, set_target_as_categorical):
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'iris_dataset_1', 'datasetDoc.json'))
        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        original_metadata = dataset.metadata

        # We make a very empty metadata.
        dataset.metadata = metadata_base.DataMetadata().generate(dataset)
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 0), 'http://schema.org/Integer')
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 0), 'https://metadata.datadrivendiscovery.org/types/PrimaryKey')
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 5), 'https://metadata.datadrivendiscovery.org/types/Attribute')
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 5), 'https://metadata.datadrivendiscovery.org/types/SuggestedTarget')

        if set_target_as_categorical:
            dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 5), 'https://metadata.datadrivendiscovery.org/types/CategoricalData')
        else:
            dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 5), 'https://metadata.datadrivendiscovery.org/types/UnknownType')

        return dataset, original_metadata

    def _test_metadata(self, original_metadata, dataframe_metadata, set_target_as_categorical):
        for column_index in range(5):
            self.assertCountEqual(original_metadata.query_column_field(column_index, 'semantic_types', at=('learningData',)), dataframe_metadata.query_column_field(column_index, 'semantic_types'), (set_target_as_categorical, column_index))

        self.assertEqual(dataframe_metadata.query_column_field(5, 'semantic_types'), (
            'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',
            'https://metadata.datadrivendiscovery.org/types/CategoricalData',
            'https://metadata.datadrivendiscovery.org/types/Target',
            'https://metadata.datadrivendiscovery.org/types/TrueTarget',
        ), set_target_as_categorical)

    def test_basic(self):
        for set_target_as_categorical in [False, True]:
            dataset, original_metadata = self._get_iris(set_target_as_categorical)

            hyperparams_class = dataset_to_dataframe.DatasetToDataFramePrimitive.metadata.get_hyperparams()

            primitive = dataset_to_dataframe.DatasetToDataFramePrimitive(hyperparams=hyperparams_class.defaults())

            dataframe = primitive.produce(inputs=dataset).value

            hyperparams_class = simple_profiler.SimpleProfilerPrimitive.metadata.get_hyperparams()

            primitive = simple_profiler.SimpleProfilerPrimitive(hyperparams=hyperparams_class.defaults())

            primitive.set_training_data(inputs=dataframe)
            primitive.fit()

            primitive_pickled = pickle.dumps(primitive)
            primitive = pickle.loads(primitive_pickled)

            dataframe = primitive.produce(inputs=dataframe).value

            self._test_metadata(original_metadata, dataframe.metadata, set_target_as_categorical)

    def test_small_test(self):
        for set_target_as_categorical in [False, True]:
            dataset, original_metadata = self._get_iris(set_target_as_categorical)

            hyperparams_class = train_score_split.TrainScoreDatasetSplitPrimitive.metadata.get_hyperparams()

            primitive = train_score_split.TrainScoreDatasetSplitPrimitive(hyperparams=hyperparams_class.defaults().replace({
                'train_score_ratio': 0.9,
                'shuffle': True,
            }))

            primitive.set_training_data(dataset=dataset)
            primitive.fit()

            results = primitive.produce(inputs=container.List([0], generate_metadata=True)).value

            self.assertEqual(len(results), 1)

            train_dataset = results[0]

            self.assertEqual(len(train_dataset['learningData']), 135)

            results = primitive.produce_score_data(inputs=container.List([0], generate_metadata=True)).value

            self.assertEqual(len(results), 1)

            score_dataset = results[0]

            self.assertEqual(len(score_dataset['learningData']), 15)

            hyperparams_class = dataset_to_dataframe.DatasetToDataFramePrimitive.metadata.get_hyperparams()

            primitive = dataset_to_dataframe.DatasetToDataFramePrimitive(hyperparams=hyperparams_class.defaults())

            train_dataframe = primitive.produce(inputs=train_dataset).value

            score_dataframe = primitive.produce(inputs=score_dataset).value

            hyperparams_class = simple_profiler.SimpleProfilerPrimitive.metadata.get_hyperparams()

            primitive = simple_profiler.SimpleProfilerPrimitive(hyperparams=hyperparams_class.defaults())

            primitive.set_training_data(inputs=train_dataframe)
            primitive.fit()
            dataframe = primitive.produce(inputs=score_dataframe).value

            self._test_metadata(original_metadata, dataframe.metadata, set_target_as_categorical)

    def _get_column_semantic_types(self, dataframe):
        number_of_columns = dataframe.metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length']
        generated_semantic_types = [
            dataframe.metadata.query((metadata_base.ALL_ELEMENTS, i))['semantic_types']
            for i in range(number_of_columns)
        ]
        generated_semantic_types = [sorted(x) for x in generated_semantic_types]

        return generated_semantic_types

    def test_iris_csv(self):
        dataset_doc_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'iris_dataset_1', 'tables', 'learningData.csv')
        )
        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        # Use profiler to assign semantic types
        dataframe = self._profile_dataset(dataset=dataset)

        generated_semantic_types = self._get_column_semantic_types(dataframe)

        semantic_types = [
            [
                'http://schema.org/Integer',
                'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
            ],
            ['http://schema.org/Float', 'https://metadata.datadrivendiscovery.org/types/Attribute'],
            ['http://schema.org/Float', 'https://metadata.datadrivendiscovery.org/types/Attribute'],
            ['http://schema.org/Float', 'https://metadata.datadrivendiscovery.org/types/Attribute'],
            ['http://schema.org/Float', 'https://metadata.datadrivendiscovery.org/types/Attribute'],
            [
                'https://metadata.datadrivendiscovery.org/types/Attribute',
                'https://metadata.datadrivendiscovery.org/types/CategoricalData',
            ],
        ]

        self.assertEqual(generated_semantic_types, semantic_types)

    def _profile_dataset(self, dataset, hyperparams=None):
        if hyperparams is None:
            hyperparams = {}

        hyperparams_class = dataset_to_dataframe.DatasetToDataFramePrimitive.metadata.get_hyperparams()
        primitive = dataset_to_dataframe.DatasetToDataFramePrimitive(hyperparams=hyperparams_class.defaults())
        dataframe = primitive.produce(inputs=dataset).value

        hyperparams_class = simple_profiler.SimpleProfilerPrimitive.metadata.get_hyperparams()
        primitive = simple_profiler.SimpleProfilerPrimitive(hyperparams=hyperparams_class.defaults().replace(hyperparams))
        primitive.set_training_data(inputs=dataframe)
        primitive.fit()

        return primitive.produce(inputs=dataframe).value

    def test_boston(self):
        dataset = container.dataset.Dataset.load('sklearn://boston')

        # Use profiler to assign semantic types
        dataframe = self._profile_dataset(dataset=dataset)

        generated_semantic_types = self._get_column_semantic_types(dataframe)

        semantic_types = [
            ['http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/PrimaryKey'],
            ['http://schema.org/Float', 'https://metadata.datadrivendiscovery.org/types/Attribute'],
            ['http://schema.org/Float', 'https://metadata.datadrivendiscovery.org/types/Attribute'],
            ['http://schema.org/Float', 'https://metadata.datadrivendiscovery.org/types/Attribute'],
            ['http://schema.org/Boolean', 'https://metadata.datadrivendiscovery.org/types/Attribute'],
            ['http://schema.org/Float', 'https://metadata.datadrivendiscovery.org/types/Attribute'],
            ['http://schema.org/Float', 'https://metadata.datadrivendiscovery.org/types/Attribute'],
            ['http://schema.org/Float', 'https://metadata.datadrivendiscovery.org/types/Attribute'],
            ['http://schema.org/Float', 'https://metadata.datadrivendiscovery.org/types/Attribute'],
            [
                'https://metadata.datadrivendiscovery.org/types/Attribute',
                'https://metadata.datadrivendiscovery.org/types/CategoricalData',
            ],
            ['http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/Attribute'],
            ['http://schema.org/Float', 'https://metadata.datadrivendiscovery.org/types/Attribute'],
            ['http://schema.org/Float', 'https://metadata.datadrivendiscovery.org/types/Attribute'],
            ['http://schema.org/Float', 'https://metadata.datadrivendiscovery.org/types/Attribute'],
            [
                'http://schema.org/Float',
                'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',
                'https://metadata.datadrivendiscovery.org/types/Target',
                'https://metadata.datadrivendiscovery.org/types/TrueTarget',
            ],
        ]

        self.assertEqual(generated_semantic_types, semantic_types)

    def test_diabetes(self):
        dataset = container.dataset.Dataset.load('sklearn://diabetes')

        # Use profiler to assign semantic types
        dataframe = self._profile_dataset(dataset=dataset)

        generated_semantic_types = self._get_column_semantic_types(dataframe)

        semantic_types = [
            ['http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/PrimaryKey'],
            ['http://schema.org/Float', 'https://metadata.datadrivendiscovery.org/types/Attribute'],
            ['http://schema.org/Float', 'https://metadata.datadrivendiscovery.org/types/Attribute'],
            ['http://schema.org/Float', 'https://metadata.datadrivendiscovery.org/types/Attribute'],
            ['http://schema.org/Float', 'https://metadata.datadrivendiscovery.org/types/Attribute'],
            ['http://schema.org/Float', 'https://metadata.datadrivendiscovery.org/types/Attribute'],
            ['http://schema.org/Float', 'https://metadata.datadrivendiscovery.org/types/Attribute'],
            ['http://schema.org/Float', 'https://metadata.datadrivendiscovery.org/types/Attribute'],
            ['http://schema.org/Float', 'https://metadata.datadrivendiscovery.org/types/Attribute'],
            ['http://schema.org/Float', 'https://metadata.datadrivendiscovery.org/types/Attribute'],
            ['http://schema.org/Float', 'https://metadata.datadrivendiscovery.org/types/Attribute'],
            [
                'http://schema.org/Integer',
                'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',
                'https://metadata.datadrivendiscovery.org/types/Target',
                'https://metadata.datadrivendiscovery.org/types/TrueTarget',
            ],
        ]

        self.assertEqual(generated_semantic_types, semantic_types)

    def test_digits(self):
        self.maxDiff = None

        dataset = container.dataset.Dataset.load('sklearn://digits')

        detect_semantic_types = list(simple_profiler.SimpleProfilerPrimitive.metadata.get_hyperparams().configuration['detect_semantic_types'].get_default())
        # Some pixels have very little different values.
        detect_semantic_types.remove('http://schema.org/Boolean')
        # There are just 16 colors, but we want to see them as integers.
        detect_semantic_types.remove('https://metadata.datadrivendiscovery.org/types/CategoricalData')

        # Use profiler to assign semantic types
        dataframe = self._profile_dataset(dataset=dataset, hyperparams={
            'detect_semantic_types': detect_semantic_types,
        })

        generated_semantic_types = self._get_column_semantic_types(dataframe)

        semantic_types = (
            [['http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/PrimaryKey']]
            + 64
            * [
                [
                    'http://schema.org/Integer',
                    'https://metadata.datadrivendiscovery.org/types/Attribute',
                ]
            ]
            + [
                [
                    'https://metadata.datadrivendiscovery.org/types/CategoricalData',
                    'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',
                    'https://metadata.datadrivendiscovery.org/types/Target',
                    'https://metadata.datadrivendiscovery.org/types/TrueTarget',
                ]
            ]
        )

        self.assertEqual(generated_semantic_types, semantic_types)

    def test_iris(self):
        dataset = container.dataset.Dataset.load('sklearn://iris')

        # Use profiler to assign semantic types
        dataframe = self._profile_dataset(dataset=dataset)

        generated_semantic_types = self._get_column_semantic_types(dataframe)

        semantic_types = [
            ['http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/PrimaryKey'],
            [
                'http://schema.org/Float',
                'https://metadata.datadrivendiscovery.org/types/Attribute',
            ],
            [
                'http://schema.org/Float',
                'https://metadata.datadrivendiscovery.org/types/Attribute',
            ],
            [
                'http://schema.org/Float',
                'https://metadata.datadrivendiscovery.org/types/Attribute',
            ],
            [
                'http://schema.org/Float',
                'https://metadata.datadrivendiscovery.org/types/Attribute',
            ],
            [
                'https://metadata.datadrivendiscovery.org/types/CategoricalData',
                'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',
                'https://metadata.datadrivendiscovery.org/types/Target',
                'https://metadata.datadrivendiscovery.org/types/TrueTarget',
            ],
        ]

        self.assertEqual(generated_semantic_types, semantic_types)

    def test_breast_cancer(self):
        dataset = container.dataset.Dataset.load('sklearn://breast_cancer')

        # Use profiler to assign semantic types
        dataframe = self._profile_dataset(dataset=dataset)

        generated_semantic_types = self._get_column_semantic_types(dataframe)

        semantic_types = (
            [['http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/PrimaryKey']]
            + 30
            * [
                [
                    'http://schema.org/Float',
                    'https://metadata.datadrivendiscovery.org/types/Attribute',
                ]
            ]
            + [
                [
                    'http://schema.org/Boolean',
                    'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',
                    'https://metadata.datadrivendiscovery.org/types/Target',
                    'https://metadata.datadrivendiscovery.org/types/TrueTarget',
                ]
            ]
        )

        self.assertEqual(generated_semantic_types, semantic_types)

    def test_linnerud(self):
        dataset = container.dataset.Dataset.load('sklearn://linnerud')

        # Use profiler to assign semantic types
        dataframe = self._profile_dataset(dataset=dataset)

        generated_semantic_types = self._get_column_semantic_types(dataframe)

        semantic_types = [
            ['http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/PrimaryKey'],
            ['http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/Attribute'],
            ['http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/Attribute'],
            ['http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/Attribute'],
            [
                'http://schema.org/Integer',
                'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',
                # Only the first "SuggestedTarget" column is made into a target.
                'https://metadata.datadrivendiscovery.org/types/Target',
                'https://metadata.datadrivendiscovery.org/types/TrueTarget',
            ],
            [
                'http://schema.org/Integer',
                'https://metadata.datadrivendiscovery.org/types/Attribute',
                'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',
            ],
            [
                'http://schema.org/Integer',
                'https://metadata.datadrivendiscovery.org/types/Attribute',
                'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',
            ],
        ]

        self.assertEqual(generated_semantic_types, semantic_types)

    def test_wine(self):
        dataset = container.dataset.Dataset.load('sklearn://wine')

        # Use profiler to assign semantic types
        dataframe = self._profile_dataset(dataset=dataset)

        generated_semantic_types = self._get_column_semantic_types(dataframe)

        semantic_types = [
            ['http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/PrimaryKey'],
            [
                'http://schema.org/Float',
                'https://metadata.datadrivendiscovery.org/types/Attribute',
            ],
            [
                'http://schema.org/Float',
                'https://metadata.datadrivendiscovery.org/types/Attribute',
            ],
            [
                'http://schema.org/Float',
                'https://metadata.datadrivendiscovery.org/types/Attribute',
            ],
            [
                'http://schema.org/Float',
                'https://metadata.datadrivendiscovery.org/types/Attribute',
            ],
            [
                'http://schema.org/Integer',
                'https://metadata.datadrivendiscovery.org/types/Attribute',
            ],
            [
                'http://schema.org/Float',
                'https://metadata.datadrivendiscovery.org/types/Attribute',
            ],
            [
                'http://schema.org/Float',
                'https://metadata.datadrivendiscovery.org/types/Attribute',
            ],
            [
                'http://schema.org/Float',
                'https://metadata.datadrivendiscovery.org/types/Attribute',
            ],
            [
                'http://schema.org/Float',
                'https://metadata.datadrivendiscovery.org/types/Attribute',
            ],
            [
                'http://schema.org/Float',
                'https://metadata.datadrivendiscovery.org/types/Attribute',
            ],
            [
                'http://schema.org/Float',
                'https://metadata.datadrivendiscovery.org/types/Attribute',
            ],
            [
                'http://schema.org/Float',
                'https://metadata.datadrivendiscovery.org/types/Attribute',
            ],
            [
                'http://schema.org/Integer',
                'https://metadata.datadrivendiscovery.org/types/Attribute',
            ],
            [
                'https://metadata.datadrivendiscovery.org/types/CategoricalData',
                'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',
                'https://metadata.datadrivendiscovery.org/types/Target',
                'https://metadata.datadrivendiscovery.org/types/TrueTarget',
            ],
        ]

        self.assertEqual(generated_semantic_types, semantic_types)


if __name__ == '__main__':
    unittest.main()
