import unittest
import os

from common_primitives import term_filter
from d3m import container

import utils as test_utils


class TermFilterPrimitiveTestCase(unittest.TestCase):
    def test_inclusive(self):
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'database_dataset_1', 'datasetDoc.json'))
        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))
        resource = test_utils.get_dataframe(dataset)

        filter_hyperparams_class = term_filter.TermFilterPrimitive.metadata.get_hyperparams()
        hp = filter_hyperparams_class({
            'column': 1,
            'inclusive': True,
            'terms': ['AAA', 'CCC'],
            'match_whole': True
        })

        filter_primitive = term_filter.TermFilterPrimitive(hyperparams=hp)
        new_df = filter_primitive.produce(inputs=resource).value

        self.assertTrue(set(new_df['code'].unique()) == set(['AAA', 'CCC']))

    def test_exclusive(self):
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'database_dataset_1', 'datasetDoc.json'))
        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))
        resource = test_utils.get_dataframe(dataset)

        filter_hyperparams_class = term_filter.TermFilterPrimitive.metadata.get_hyperparams()
        hp = filter_hyperparams_class({
            'column': 1,
            'inclusive': False,
            'terms': ['AAA', 'CCC'],
            'match_whole': True
        })

        filter_primitive = term_filter.TermFilterPrimitive(hyperparams=hp)
        new_df = filter_primitive.produce(inputs=resource).value

        self.assertTrue(set(new_df['code'].unique()) == set(['BBB']))

    def test_numeric(self):
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'database_dataset_1', 'datasetDoc.json'))
        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))
        resource = test_utils.get_dataframe(dataset)

        # set dataframe type to int to match output of a prior parse columns step
        resource.iloc[:,3] = resource.iloc[:,3].astype(int)

        filter_hyperparams_class = term_filter.TermFilterPrimitive.metadata.get_hyperparams()
        hp = filter_hyperparams_class({
            'column': 3,
            'inclusive': False,
            'terms': ['1990'],
            'match_whole': True
        })

        filter_primitive = term_filter.TermFilterPrimitive(hyperparams=hp)
        new_df = filter_primitive.produce(inputs=resource).value

        matches = new_df[~new_df['year'].astype(str).str.match('1990')]
        self.assertTrue(set(matches['year'].unique()) == set([2000, 2010]))

    def test_partial_no_match(self):
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'database_dataset_1', 'datasetDoc.json'))
        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))
        resource = test_utils.get_dataframe(dataset)

        filter_hyperparams_class = term_filter.TermFilterPrimitive.metadata.get_hyperparams()
        hp = filter_hyperparams_class({
            'column': 1,
            'inclusive': True,
            'terms': ['AA', 'CC'],
            'match_whole': False
        })

        filter_primitive = term_filter.TermFilterPrimitive(hyperparams=hp)
        new_df = filter_primitive.produce(inputs=resource).value

        self.assertTrue(set(new_df['code'].unique()) == set(['AAA', 'CCC']))

    def test_escaped_regex(self):
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'database_dataset_1', 'datasetDoc.json'))
        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))
        resource = test_utils.get_dataframe(dataset)

        filter_hyperparams_class = term_filter.TermFilterPrimitive.metadata.get_hyperparams()
        hp = filter_hyperparams_class({
            'column': 4,
            'inclusive': True,
            'terms': ['40.2'],
            'match_whole': False
        })

        filter_primitive = term_filter.TermFilterPrimitive(hyperparams=hp)
        new_df = filter_primitive.produce(inputs=resource).value

        self.assertListEqual(list(new_df['value']), ['40.2346487255306'])

    def test_row_metadata_removal(self):
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'database_dataset_1', 'datasetDoc.json'))
        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        # add metadata for rows 0 and 1
        dataset.metadata = dataset.metadata.update(('learningData', 1), {'a': 0})
        dataset.metadata = dataset.metadata.update(('learningData', 2), {'b': 1})

        resource = test_utils.get_dataframe(dataset)

        filter_hyperparams_class = term_filter.TermFilterPrimitive.metadata.get_hyperparams()
        hp = filter_hyperparams_class({
            'column': 1,
            'inclusive': False,
            'terms': ['AAA'],
            'match_whole': True
        })

        filter_primitive = term_filter.TermFilterPrimitive(hyperparams=hp)
        new_df = filter_primitive.produce(inputs=resource).value

        # verify that the lenght is correct
        self.assertEqual(len(new_df), new_df.metadata.query(())['dimension']['length'])

        # verify that the rows were re-indexed in the metadata
        self.assertEquals(new_df.metadata.query((0,))['a'], 0)
        self.assertEquals(new_df.metadata.query((1,))['b'], 1)
        self.assertFalse('b' in new_df.metadata.query((2,)))


if __name__ == '__main__':
    unittest.main()
