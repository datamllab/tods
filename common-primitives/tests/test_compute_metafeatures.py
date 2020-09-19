import math
import os
import os.path
import unittest

import numpy

from d3m import container
from d3m.metadata import base as metadata_base

from common_primitives import column_parser, compute_metafeatures, dataset_to_dataframe, denormalize

import utils as test_utils


def round_to_significant_digits(x, n):
    if x == 0:
        return x
    elif not numpy.isfinite(x):
        return x
    else:
        return round(x, -int(math.floor(math.log10(abs(x)))) + (n - 1))


def round_numbers(obj):
    if isinstance(obj, (int, str)):
        return obj
    elif isinstance(obj, float):
        return round_to_significant_digits(obj, 12)
    elif isinstance(obj, list):
        return [round_numbers(el) for el in obj]
    elif isinstance(obj, tuple):
        return tuple(round_numbers(el) for el in obj)
    elif isinstance(obj, dict):
        return {k: round_numbers(v) for k, v in obj.items()}
    else:
        return obj


class ComputeMetafeaturesPrimitiveTestCase(unittest.TestCase):
    def _get_iris(self):
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'iris_dataset_1', 'datasetDoc.json'))

        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        # We set semantic types like runtime would.
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 5), 'https://metadata.datadrivendiscovery.org/types/Target')
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 5), 'https://metadata.datadrivendiscovery.org/types/TrueTarget')
        dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 5), 'https://metadata.datadrivendiscovery.org/types/Attribute')

        dataframe_hyperparams_class = dataset_to_dataframe.DatasetToDataFramePrimitive.metadata.get_hyperparams()
        dataframe_primitive = dataset_to_dataframe.DatasetToDataFramePrimitive(hyperparams=dataframe_hyperparams_class.defaults())
        dataframe = dataframe_primitive.produce(inputs=dataset).value

        column_parser_hyperparams_class = column_parser.ColumnParserPrimitive.metadata.get_hyperparams()
        column_parser_primitive = column_parser.ColumnParserPrimitive(hyperparams=column_parser_hyperparams_class.defaults())
        dataframe = column_parser_primitive.produce(inputs=dataframe).value

        return dataframe

    def _get_database(self, parse_categorical_columns):
        dataset_doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'datasets', 'database_dataset_1', 'datasetDoc.json'))

        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        # We set semantic types like runtime would.
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 4), 'https://metadata.datadrivendiscovery.org/types/Target')
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 4), 'https://metadata.datadrivendiscovery.org/types/TrueTarget')
        dataset.metadata = dataset.metadata.remove_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 4), 'https://metadata.datadrivendiscovery.org/types/Attribute')

        denormalize_hyperparams_class = denormalize.DenormalizePrimitive.metadata.get_hyperparams()
        denormalize_primitive = denormalize.DenormalizePrimitive(hyperparams=denormalize_hyperparams_class.defaults())
        dataset = denormalize_primitive.produce(inputs=dataset).value

        dataframe_hyperparams_class = dataset_to_dataframe.DatasetToDataFramePrimitive.metadata.get_hyperparams()
        dataframe_primitive = dataset_to_dataframe.DatasetToDataFramePrimitive(hyperparams=dataframe_hyperparams_class.defaults())
        dataframe = dataframe_primitive.produce(inputs=dataset).value

        if parse_categorical_columns:
            parse_semantic_types = (
                'http://schema.org/Boolean', 'https://metadata.datadrivendiscovery.org/types/CategoricalData',
                'http://schema.org/Integer', 'http://schema.org/Float',
                'https://metadata.datadrivendiscovery.org/types/FloatVector', 'http://schema.org/DateTime',
            )
        else:
            parse_semantic_types = (
                'http://schema.org/Boolean',
                'http://schema.org/Integer', 'http://schema.org/Float',
                'https://metadata.datadrivendiscovery.org/types/FloatVector', 'http://schema.org/DateTime',
            )

        column_parser_hyperparams_class = column_parser.ColumnParserPrimitive.metadata.get_hyperparams()
        column_parser_primitive = column_parser.ColumnParserPrimitive(hyperparams=column_parser_hyperparams_class.defaults().replace({'parse_semantic_types': parse_semantic_types}))
        dataframe = column_parser_primitive.produce(inputs=dataframe).value

        return dataframe

    def test_iris(self):
        self.maxDiff = None

        dataframe = self._get_iris()

        hyperparams_class = compute_metafeatures.ComputeMetafeaturesPrimitive.metadata.get_hyperparams()
        primitive = compute_metafeatures.ComputeMetafeaturesPrimitive(hyperparams=hyperparams_class.defaults())
        dataframe = primitive.produce(inputs=dataframe).value

        self.assertEqual(round_numbers(test_utils.convert_through_json(dataframe.metadata.query(())['data_metafeatures'])), round_numbers({
            'attribute_counts_by_semantic_type': {
                'http://schema.org/Float': 4,
                'https://metadata.datadrivendiscovery.org/types/Attribute': 4,
            },
            'attribute_counts_by_structural_type': {
                'float': 4,
            },
            'attribute_ratios_by_semantic_type': {
                'http://schema.org/Float': 1.0,
                'https://metadata.datadrivendiscovery.org/types/Attribute': 1.0,
            },
            'attribute_ratios_by_structural_type': {
                'float': 1.0,
            },
            'dimensionality': 0.02666666666666667,
            'entropy_of_attributes': {
                'count': 4,
                'kurtosis': -1.4343159590314425,
                'max': 1.525353510619575,
                'mean': 1.4166844257365265,
                'median': 1.4323995290219738,
                'min': 1.2765851342825842,
                'quartile_1': 1.3565647450899858,
                'quartile_3': 1.4925192096685145,
                'skewness': -0.6047691718752254,
                'std': 0.11070539686522164,
            },
            'entropy_of_numeric_attributes': {
                'count': 4,
                'kurtosis': -1.4343159590314425,
                'max': 1.525353510619575,
                'mean': 1.4166844257365265,
                'median': 1.4323995290219738,
                'min': 1.2765851342825842,
                'quartile_1': 1.3565647450899858,
                'quartile_3': 1.4925192096685145,
                'skewness': -0.6047691718752254,
                'std': 0.11070539686522164,
            },
            'kurtosis_of_attributes': {
                'count': 4,
                'kurtosis': -1.1515850633224236,
                'max': 0.2907810623654279,
                'mean': -0.7507394876837397,
                'median': -0.9459091062274914,
                'min': -1.4019208006454036,
                'quartile_1': -1.3552958285158583,
                'quartile_3': -0.3413527653953726,
                'skewness': 0.8725328682893572,
                'std': 0.7948191385132984,
            },
            'mean_of_attributes': {
                'count': 4,
                'kurtosis': 0.8595879081956515,
                'max': 5.843333333333335,
                'mean': 3.4636666666666684,
                'median': 3.406333333333335,
                'min': 1.1986666666666672,
                'quartile_1': 2.5901666666666676,
                'quartile_3': 4.279833333333335,
                'skewness': 0.17098811780721151,
                'std': 1.919017997329383,
            },
            'number_distinct_values_of_numeric_attributes': {
                'count': 4,
                'kurtosis': -3.0617196548227046,
                'max': 43,
                'mean': 30.75,
                'median': 29.0,
                'min': 22,
                'quartile_1': 22.75,
                'quartile_3': 37.0,
                'skewness': 0.5076458131399395,
                'std': 10.07885575516057,
            },
            'number_of_attributes': 4,
            'number_of_binary_attributes': 0,
            'number_of_categorical_attributes': 0,
            'number_of_discrete_attributes': 0,
            'number_of_instances': 150,
            'number_of_instances_with_missing_values': 0,
            'number_of_instances_with_present_values': 150,
            'number_of_numeric_attributes': 4,
            'number_of_other_attributes': 0,
            'number_of_string_attributes': 0,
            'ratio_of_binary_attributes': 0.0,
            'ratio_of_categorical_attributes': 0.0,
            'ratio_of_discrete_attributes': 0.0,
            'ratio_of_instances_with_missing_values': 0.0,
            'ratio_of_instances_with_present_values': 1.0,
            'ratio_of_numeric_attributes': 1.0,
            'ratio_of_other_attributes': 0.0,
            'ratio_of_string_attributes': 0.0,
            'skew_of_attributes': {
                'count': 4,
                'kurtosis': -4.4981774675194846,
                'max': 0.3340526621720866,
                'mean': 0.06737570104778733,
                'median': 0.10495719724642275,
                'min': -0.27446425247378287,
                'quartile_1': -0.1473634847265412,
                'quartile_3': 0.3196963830207513,
                'skewness': -0.25709026597426626,
                'std': 0.3049355425307816,
            },
            'standard_deviation_of_attributes': {
                'count': 4,
                'kurtosis': 2.65240266862979,
                'max': 1.7644204199522617,
                'mean': 0.9473104002482848,
                'median': 0.7956134348393522,
                'min': 0.4335943113621737,
                'quartile_1': 0.6807691341161745,
                'quartile_3': 1.0621547009714627,
                'skewness': 1.4362343455338735,
                'std': 0.5714610798918619,
            }
        }))
        self.assertFalse('data_metafeatures' in dataframe.metadata.query_column(0))
        self.assertEqual(round_numbers(test_utils.convert_through_json(dataframe.metadata.query_column(1)['data_metafeatures'])), round_numbers({
            'entropy_of_values': 1.525353510619575,
            'number_distinct_values': 35,
            'number_of_missing_values': 0,
            'number_of_negative_numeric_values': 0,
            'number_of_numeric_values': 150,
            'number_of_numeric_values_equal_-1': 0,
            'number_of_numeric_values_equal_0': 0,
            'number_of_numeric_values_equal_1': 0,
            'number_of_positive_numeric_values': 150,
            'number_of_present_values': 150,
            'ratio_of_missing_values': 0.0,
            'ratio_of_negative_numeric_values': 0.0,
            'ratio_of_numeric_values': 1.0,
            'ratio_of_numeric_values_equal_-1': 0.0,
            'ratio_of_numeric_values_equal_0': 0.0,
            'ratio_of_numeric_values_equal_1': 0.0,
            'ratio_of_positive_numeric_values': 1.0,
            'ratio_of_present_values': 1.0,
            'value_counts_aggregate': {
                'count': 5,
                'kurtosis': -0.46949652355057747,
                'max': 42,
                'mean': 30.0,
                'median': 32.0,
                'min': 11,
                'quartile_1': 24.0,
                'quartile_3': 41.0,
                'skewness': -0.7773115383470599,
                'std': 12.90348790056394,
            },
            'value_probabilities_aggregate': {
                'count': 5,
                'kurtosis': -0.4694965235505757,
                'max': 0.28,
                'mean': 0.2,
                'median': 0.21333333333333335,
                'min': 0.07333333333333333,
                'quartile_1': 0.16,
                'quartile_3': 0.2733333333333333,
                'skewness': -0.7773115383470603,
                'std': 0.08602325267042626,
            },
            'values_aggregate': {
                'count': 150,
                'kurtosis': -0.5520640413156395,
                'max': 7.9,
                'mean': 5.843333333333335,
                'median': 5.8,
                'min': 4.3,
                'quartile_1': 5.1,
                'quartile_3': 6.4,
                'skewness': 0.3149109566369728,
                'std': 0.8280661279778629,
            },
        }))
        self.assertEqual(round_numbers(test_utils.convert_through_json(dataframe.metadata.query_column(2)['data_metafeatures'])), round_numbers({
            'entropy_of_values': 1.2765851342825842,
            'number_distinct_values': 23,
            'number_of_missing_values': 0,
            'number_of_negative_numeric_values': 0,
            'number_of_numeric_values': 150,
            'number_of_numeric_values_equal_-1': 0,
            'number_of_numeric_values_equal_0': 0,
            'number_of_numeric_values_equal_1': 0,
            'number_of_positive_numeric_values': 150,
            'number_of_present_values': 150,
            'ratio_of_missing_values': 0.0,
            'ratio_of_negative_numeric_values': 0.0,
            'ratio_of_numeric_values': 1.0,
            'ratio_of_numeric_values_equal_-1': 0.0,
            'ratio_of_numeric_values_equal_0': 0.0,
            'ratio_of_numeric_values_equal_1': 0.0,
            'ratio_of_positive_numeric_values': 1.0,
            'ratio_of_present_values': 1.0,
            'value_counts_aggregate': {
                'count': 5,
                'kurtosis': -0.9899064888741496,
                'max': 69,
                'mean': 30.0,
                'median': 20.0,
                'min': 4,
                'quartile_1': 11.0,
                'quartile_3': 46.0,
                'skewness': 0.8048211570183503,
                'std': 26.99073915253156,
            },
            'value_probabilities_aggregate': {
                'count': 5,
                'kurtosis': -0.9899064888741478,
                'max': 0.46,
                'mean': 0.19999999999999998,
                'median': 0.13333333333333333,
                'min': 0.02666666666666667,
                'quartile_1': 0.07333333333333333,
                'quartile_3': 0.30666666666666664,
                'skewness': 0.8048211570183509,
                'std': 0.17993826101687704,
            },
            'values_aggregate': {
                'count': 150,
                'kurtosis': 0.2907810623654279,
                'max': 4.4,
                'mean': 3.0540000000000007,
                'median': 3.0,
                'min': 2.0,
                'quartile_1': 2.8,
                'quartile_3': 3.3,
                'skewness': 0.3340526621720866,
                'std': 0.4335943113621737,
            },
        }))
        self.assertEqual(round_numbers(test_utils.convert_through_json(dataframe.metadata.query_column(3)['data_metafeatures'])), round_numbers({
            'entropy_of_values': 1.38322461535912,
            'number_distinct_values': 43,
            'number_of_missing_values': 0,
            'number_of_negative_numeric_values': 0,
            'number_of_numeric_values': 150,
            'number_of_numeric_values_equal_-1': 0,
            'number_of_numeric_values_equal_0': 0,
            'number_of_numeric_values_equal_1': 1,
            'number_of_positive_numeric_values': 150,
            'number_of_present_values': 150,
            'ratio_of_missing_values': 0.0,
            'ratio_of_negative_numeric_values': 0.0,
            'ratio_of_numeric_values': 1.0,
            'ratio_of_numeric_values_equal_-1': 0.0,
            'ratio_of_numeric_values_equal_0': 0.0,
            'ratio_of_numeric_values_equal_1': 0.006666666666666667,
            'ratio_of_positive_numeric_values': 1.0,
            'ratio_of_present_values': 1.0,
            'value_counts_aggregate': {
                'count': 5,
                'kurtosis': -1.875313335089766,
                'max': 50,
                'mean': 30.0,
                'median': 34.0,
                'min': 3,
                'quartile_1': 16.0,
                'quartile_3': 47.0,
                'skewness': -0.4786622161186872,
                'std': 20.18662923818635,
            },
            'value_probabilities_aggregate': {
                'count': 5,
                'kurtosis': -1.8753133350897668,
                'max': 0.3333333333333333,
                'mean': 0.2,
                'median': 0.22666666666666666,
                'min': 0.02,
                'quartile_1': 0.10666666666666667,
                'quartile_3': 0.31333333333333335,
                'skewness': -0.4786622161186876,
                'std': 0.13457752825457567,
            },
            'values_aggregate': {
                'count': 150,
                'kurtosis': -1.4019208006454036,
                'max': 6.9,
                'mean': 3.7586666666666693,
                'median': 4.35,
                'min': 1.0,
                'quartile_1': 1.6,
                'quartile_3': 5.1,
                'skewness': -0.27446425247378287,
                'std': 1.7644204199522617,
            },
        }))
        self.assertEqual(round_numbers(test_utils.convert_through_json(dataframe.metadata.query_column(4)['data_metafeatures'])), round_numbers({
            'entropy_of_values': 1.4815744426848276,
            'number_distinct_values': 22,
            'number_of_missing_values': 0,
            'number_of_negative_numeric_values': 0,
            'number_of_numeric_values': 150,
            'number_of_numeric_values_equal_-1': 0,
            'number_of_numeric_values_equal_0': 0,
            'number_of_numeric_values_equal_1': 7,
            'number_of_positive_numeric_values': 150,
            'number_of_present_values': 150,
            'ratio_of_missing_values': 0.0,
            'ratio_of_negative_numeric_values': 0.0,
            'ratio_of_numeric_values': 1.0,
            'ratio_of_numeric_values_equal_-1': 0.0,
            'ratio_of_numeric_values_equal_0': 0.0,
            'ratio_of_numeric_values_equal_1': 0.04666666666666667,
            'ratio_of_positive_numeric_values': 1.0,
            'ratio_of_present_values': 1.0,
            'value_counts_aggregate': {
                'count': 5,
                'kurtosis': -0.6060977121954245,
                'max': 49,
                'mean': 30.0,
                'median': 29.0,
                'min': 8,
                'quartile_1': 23.0,
                'quartile_3': 41.0,
                'skewness': -0.28840734350346464,
                'std': 15.937377450509228,
            },
            'value_probabilities_aggregate': {
                'count': 5,
                'kurtosis': -0.606097712195421,
                'max': 0.32666666666666666,
                'mean': 0.2,
                'median': 0.19333333333333333,
                'min': 0.05333333333333334,
                'quartile_1': 0.15333333333333332,
                'quartile_3': 0.2733333333333333,
                'skewness': -0.2884073435034653,
                'std': 0.10624918300339484,
            },
            'values_aggregate': {
                'count': 150,
                'kurtosis': -1.3397541711393433,
                'max': 2.5,
                'mean': 1.1986666666666672,
                'median': 1.3,
                'min': 0.1,
                'quartile_1': 0.3,
                'quartile_3': 1.8,
                'skewness': -0.10499656214412734,
                'std': 0.7631607417008414,
            },
        }))
        self.assertEqual(round_numbers(test_utils.convert_through_json(dataframe.metadata.query_column(5)['data_metafeatures'])), round_numbers({
            'default_accuracy': 0.3333333333333333,
            'entropy_of_values': 1.0986122886681096,
            'equivalent_number_of_numeric_attributes': 1.7538156960944151,
            'joint_entropy_of_attributes': {
                'count': 4,
                'kurtosis': -4.468260105522818,
                'max': 0.9180949375453917,
                'mean': 0.6264126219845205,
                'median': 0.6607409495199184,
                'min': 0.26607365135285327,
                'quartile_1': 0.3993550878466134,
                'quartile_3': 0.8877984836578254,
                'skewness': -0.24309705749856694,
                'std': 0.3221913428169348,
            },
            'joint_entropy_of_numeric_attributes': {
                'count': 4,
                'kurtosis': -5.533056612798099,
                'max': 2.1801835659431514,
                'mean': 1.8888840924201158,
                'median': 1.8856077827026931,
                'min': 1.604137238331926,
                'quartile_1': 1.6476031549386407,
                'quartile_3': 2.1268887201841684,
                'skewness': 0.01639056780792744,
                'std': 0.29770030633854977,
            },
            'mutual_information_of_numeric_attributes': {
                'count': 4,
                'kurtosis': -4.468260105522818,
                'max': 0.9180949375453917,
                'mean': 0.6264126219845205,
                'median': 0.6607409495199184,
                'min': 0.26607365135285327,
                'quartile_1': 0.3993550878466134,
                'quartile_3': 0.8877984836578254,
                'skewness': -0.24309705749856694,
                'std': 0.3221913428169348,
            },
            'number_distinct_values': 3,
            'number_of_missing_values': 0,
            'number_of_present_values': 150,
            'numeric_noise_to_signal_ratio': 1.2615834611511623,
            'ratio_of_missing_values': 0.0,
            'ratio_of_present_values': 1.0,
            'value_counts_aggregate': {
                'count': 3,
                'max': 50,
                'mean': 50.0,
                'median': 50.0,
                'min': 50,
                'quartile_1': 50.0,
                'quartile_3': 50.0,
                'skewness': 0,
                'std': 0.0,
            },
            'value_probabilities_aggregate': {
                'count': 3,
                'max': 0.3333333333333333,
                'mean': 0.3333333333333333,
                'median': 0.3333333333333333,
                'min': 0.3333333333333333,
                'quartile_1': 0.3333333333333333,
                'quartile_3': 0.3333333333333333,
                'skewness': 0,
                'std': 0.0,
            },
        }))

    def test_database_with_parsed_categorical_columns(self):
        self.maxDiff = None

        dataframe = self._get_database(True)

        hyperparams_class = compute_metafeatures.ComputeMetafeaturesPrimitive.metadata.get_hyperparams()
        primitive = compute_metafeatures.ComputeMetafeaturesPrimitive(hyperparams=hyperparams_class.defaults())
        dataframe = primitive.produce(inputs=dataframe).value

        self._test_database_metafeatures(dataframe.metadata, True)

    def test_database_without_parsed_categorical_columns(self):
        self.maxDiff = None

        dataframe = self._get_database(False)

        hyperparams_class = compute_metafeatures.ComputeMetafeaturesPrimitive.metadata.get_hyperparams()
        primitive = compute_metafeatures.ComputeMetafeaturesPrimitive(hyperparams=hyperparams_class.defaults())
        dataframe = primitive.produce(inputs=dataframe).value

        self._test_database_metafeatures(dataframe.metadata, False)

    def _test_database_metafeatures(self, metadata, parse_categorical_columns):
        expected_metafeatures = {
            'attribute_counts_by_semantic_type': {
                'http://schema.org/DateTime': 1,
                'http://schema.org/Integer': 1,
                'http://schema.org/Text': 2,
                'https://metadata.datadrivendiscovery.org/types/Attribute': 6,
                'https://metadata.datadrivendiscovery.org/types/CategoricalData': 2,
            },
            'attribute_counts_by_structural_type': {
                'float': 2,
                'str': 4,
            },
            'attribute_ratios_by_semantic_type': {
                'http://schema.org/DateTime': 0.16666666666666666,
                'http://schema.org/Integer': 0.16666666666666666,
                'http://schema.org/Text': 0.3333333333333333,
                'https://metadata.datadrivendiscovery.org/types/Attribute': 1.0,
                'https://metadata.datadrivendiscovery.org/types/CategoricalData': 0.3333333333333333,
            },
            'attribute_ratios_by_structural_type': {
                'float': 0.3333333333333333,
                'str': 0.6666666666666666,
            },
            'dimensionality': 0.13333333333333333,
            'entropy_of_attributes': {
                'count': 4,
                'kurtosis': 1.5975414707531783,
                'max': 1.6094379124341005,
                'mean': 1.1249524175825663,
                'median': 1.0986122886681096,
                'min': 0.6931471805599453,
                'quartile_1': 0.9972460116410685,
                'quartile_3': 1.2263186946096072,
                'skewness': 0.4183300365459641,
                'std': 0.3753085673700856,
            },
            'entropy_of_categorical_attributes': {
                'count': 2,
                'max': 1.6094379124341005,
                'mean': 1.354025100551105,
                'median': 1.354025100551105,
                'min': 1.0986122886681096,
                'quartile_1': 1.2263186946096072,
                'quartile_3': 1.4817315064926029,
                'std': 0.3612082625687802,
            },
            'entropy_of_discrete_attributes': {
                'count': 2,
                'max': 1.0986122886681096,
                'mean': 0.8958797346140275,
                'median': 0.8958797346140275,
                'min': 0.6931471805599453,
                'quartile_1': 0.7945134575869863,
                'quartile_3': 0.9972460116410685,
                'std': 0.28670712747781957,
            },
            'entropy_of_numeric_attributes': {
                'count': 2,
                'max': 1.0986122886681096,
                'mean': 0.8958797346140275,
                'median': 0.8958797346140275,
                'min': 0.6931471805599453,
                'quartile_1': 0.7945134575869863,
                'quartile_3': 0.9972460116410685,
                'std': 0.28670712747781957,
            },
            'kurtosis_of_attributes': {
                'count': 2,
                'max': -1.5348837209302326,
                'mean': -1.8415159345391905,
                'median': -1.8415159345391905,
                'min': -2.1481481481481484,
                'quartile_1': -1.9948320413436693,
                'quartile_3': -1.6881998277347114,
                'std': 0.4336434351462721,
            },
            'mean_of_attributes': {
                'count': 2,
                'max': 946713600.0,
                'mean': 473356800.75,
                'median': 473356800.75,
                'min': 1.5,
                'quartile_1': 236678401.125,
                'quartile_3': 710035200.375,
                'std': 669427605.3408685,
            },
            'number_distinct_values_of_categorical_attributes': {
                'count': 2,
                'max': 5,
                'mean': 4.0,
                'median': 4.0,
                'min': 3,
                'quartile_1': 3.5,
                'quartile_3': 4.5,
                'std': 1.4142135623730951,
            },
            'number_distinct_values_of_discrete_attributes': {
                'count': 2,
                'max': 3,
                'mean': 2.5,
                'median': 2.5,
                'min': 2,
                'quartile_1': 2.25,
                'quartile_3': 2.75,
                'std': 0.7071067811865476,
            },
            'number_distinct_values_of_numeric_attributes': {
                'count': 2,
                'max': 3,
                'mean': 2.5,
                'median': 2.5,
                'min': 2,
                'quartile_1': 2.25,
                'quartile_3': 2.75,
                'std': 0.7071067811865476,
            },
            'number_of_attributes': 6,
            'number_of_binary_attributes': 1,
            'number_of_categorical_attributes': 2,
            'number_of_discrete_attributes': 2,
            'number_of_instances': 45,
            'number_of_instances_with_missing_values': 15,
            'number_of_instances_with_present_values': 45,
            'number_of_numeric_attributes': 2,
            'number_of_other_attributes': 0,
            'number_of_string_attributes': 2,
            'ratio_of_binary_attributes': 0.16666666666666666,
            'ratio_of_categorical_attributes': 0.3333333333333333,
            'ratio_of_discrete_attributes': 0.3333333333333333,
            'ratio_of_instances_with_missing_values': 0.3333333333333333,
            'ratio_of_instances_with_present_values': 1.0,
            'ratio_of_numeric_attributes': 0.3333333333333333,
            'ratio_of_other_attributes': 0.0,
            'ratio_of_string_attributes': 0.3333333333333333,
            'skew_of_attributes': {
                'count': 2,
                'max': 0.00017349603091112943,
                'mean': 8.674801545556472e-05,
                'median': 8.674801545556472e-05,
                'min': 0.0,
                'quartile_1': 4.337400772778236e-05,
                'quartile_3': 0.00013012202318334707,
                'std': 0.0001226802199662105,
            },
            'standard_deviation_of_attributes': {
                'count': 2,
                'max': 260578306.67149138,
                'mean': 130289153.59001951,
                'median': 130289153.59001951,
                'min': 0.5085476277156078,
                'quartile_1': 65144577.049283564,
                'quartile_3': 195433730.13075545,
                'std': 184256687.31792185,
            },
        }

        if parse_categorical_columns:
            expected_metafeatures['attribute_counts_by_structural_type'] = {
                'float': 2,
                'int': 2,
                'str': 2,
            }
            expected_metafeatures['attribute_ratios_by_structural_type'] = {
                'float': 0.3333333333333333,
                'int': 0.3333333333333333,
                'str': 0.3333333333333333,
            }

        self.assertEqual(round_numbers(test_utils.convert_through_json(metadata.query(())['data_metafeatures'])), round_numbers(expected_metafeatures))
        self.assertFalse('data_metafeatures' in metadata.query_column(0))

        expected_metafeatures = {
            'entropy_of_values': 1.0986122886681096,
            'number_distinct_values': 3,
            'number_of_missing_values': 0,
            'number_of_present_values': 45,
            'ratio_of_missing_values': 0.0,
            'ratio_of_present_values': 1.0,
            'value_counts_aggregate': {
                'count': 3,
                'max': 15,
                'mean': 15.0,
                'median': 15.0,
                'min': 15,
                'quartile_1': 15.0,
                'quartile_3': 15.0,
                'skewness': 0,
                'std': 0.0,
            },
            'value_probabilities_aggregate': {
                'count': 3,
                'max': 0.3333333333333333,
                'mean': 0.3333333333333333,
                'median': 0.3333333333333333,
                'min': 0.3333333333333333,
                'quartile_1': 0.3333333333333333,
                'quartile_3': 0.3333333333333333,
                'skewness': 0,
                'std': 0.0,
            },
        }

        if parse_categorical_columns:
            expected_metafeatures['values_aggregate'] = {
                'count': 45,
                'kurtosis': -1.5348837209302337,
                'max': 3183890296585507471,
                'mean': 1.3152606765673695e+18,
                'median': 5.866629697275507e+17,
                'min': 175228763389048878,
                'quartile_1': 1.7522876338904886e+17,
                'quartile_3': 3.1838902965855073e+18,
                'skewness': 0.679711376572956,
                'std': 1.3470047628846746e+18,
            }

        self.assertEqual(round_numbers(test_utils.convert_through_json(metadata.query_column(1)['data_metafeatures'])), round_numbers(expected_metafeatures))
        self.assertEqual(round_numbers(test_utils.convert_through_json(metadata.query_column(2)['data_metafeatures'])), round_numbers({
            'number_of_missing_values': 0,
            'number_of_present_values': 45,
            'ratio_of_missing_values': 0.0,
            'ratio_of_present_values': 1.0,
        }))
        self.assertEqual(round_numbers(test_utils.convert_through_json(metadata.query_column(3)['data_metafeatures'])), round_numbers({
            'entropy_of_values': 0.6931471805599453,
            'number_distinct_values': 2,
            'number_of_missing_values': 15,
            'number_of_negative_numeric_values': 0,
            'number_of_numeric_values': 30,
            'number_of_numeric_values_equal_-1': 0,
            'number_of_numeric_values_equal_0': 0,
            'number_of_numeric_values_equal_1': 15,
            'number_of_positive_numeric_values': 30,
            'number_of_present_values': 30,
            'ratio_of_missing_values': 0.3333333333333333,
            'ratio_of_negative_numeric_values': 0.0,
            'ratio_of_numeric_values': 0.6666666666666666,
            'ratio_of_numeric_values_equal_-1': 0.0,
            'ratio_of_numeric_values_equal_0': 0.0,
            'ratio_of_numeric_values_equal_1': 0.3333333333333333,
            'ratio_of_positive_numeric_values': 0.6666666666666666,
            'ratio_of_present_values': 0.6666666666666666,
            'value_counts_aggregate': {
                'count': 2,
                'max': 15,
                'mean': 15.0,
                'median': 15.0,
                'min': 15,
                'quartile_1': 15.0,
                'quartile_3': 15.0,
                'std': 0.0,
            },
            'value_probabilities_aggregate': {
                'count': 2,
                'max': 0.5,
                'mean': 0.5,
                'median': 0.5,
                'min': 0.5,
                'quartile_1': 0.5,
                'quartile_3': 0.5,
                'std': 0.0,
            },
            'values_aggregate': {
                'count': 30,
                'kurtosis': -2.1481481481481484,
                'max': 2.0,
                'mean': 1.5,
                'median': 1.5,
                'min': 1.0,
                'quartile_1': 1.0,
                'quartile_3': 2.0,
                'skewness': 0.0,
                'std': 0.5085476277156078,
            },
        }))
        self.assertEqual(round_numbers(test_utils.convert_through_json(metadata.query_column(4)['data_metafeatures'])), round_numbers({
            'number_of_missing_values': 0,
            'number_of_present_values': 45,
            'ratio_of_missing_values': 0.0,
            'ratio_of_present_values': 1.0,
        }))

        expected_metafeatures = {
            'entropy_of_values': 1.6094379124341005,
            'number_distinct_values': 5,
            'number_of_missing_values': 0,
            'number_of_present_values': 45,
            'ratio_of_missing_values': 0.0,
            'ratio_of_present_values': 1.0,
            'value_counts_aggregate': {
                'count': 5,
                'kurtosis': 0,
                'max': 9,
                'mean': 9.0,
                'median': 9.0,
                'min': 9,
                'quartile_1': 9.0,
                'quartile_3': 9.0,
                'skewness': 0,
                'std': 0.0,
            },
            'value_probabilities_aggregate': {
                'count': 5,
                'kurtosis': 0,
                'max': 0.2,
                'mean': 0.2,
                'median': 0.2,
                'min': 0.2,
                'quartile_1': 0.2,
                'quartile_3': 0.2,
                'skewness': 0,
                'std': 0.0,
            },
        }

        if parse_categorical_columns:
            expected_metafeatures['values_aggregate'] = {
                'count': 45,
                'kurtosis': -0.8249445297886884,
                'max': 17926897368031380755,
                'mean': 1.1617029581691474e+19,
                'median': 1.1818891258207388e+19,
                'min': 4819821729471251610,
                'quartile_1': 9.804127312560234e+18,
                'quartile_3': 1.3715410240187093e+19,
                'skewness': -0.15176089654708094,
                'std': 4.378987201456074e+18,
            }

        self.assertEqual(round_numbers(test_utils.convert_through_json(metadata.query_column(5)['data_metafeatures'])), round_numbers(expected_metafeatures))
        self.assertEqual(round_numbers(test_utils.convert_through_json(metadata.query_column(6)['data_metafeatures'])), round_numbers({
            'entropy_of_values': 1.0986122886681096,
            'number_distinct_values': 3,
            'number_of_missing_values': 0,
            'number_of_negative_numeric_values': 0,
            'number_of_numeric_values': 45,
            'number_of_numeric_values_equal_-1': 0,
            'number_of_numeric_values_equal_0': 0,
            'number_of_numeric_values_equal_1': 0,
            'number_of_positive_numeric_values': 45,
            'number_of_present_values': 45,
            'ratio_of_missing_values': 0.0,
            'ratio_of_negative_numeric_values': 0.0,
            'ratio_of_numeric_values': 1.0,
            'ratio_of_numeric_values_equal_-1': 0.0,
            'ratio_of_numeric_values_equal_0': 0.0,
            'ratio_of_numeric_values_equal_1': 0.0,
            'ratio_of_positive_numeric_values': 1.0,
            'ratio_of_present_values': 1.0,
            'value_counts_aggregate': {
                'count': 3,
                'max': 15,
                'mean': 15.0,
                'median': 15.0,
                'min': 15,
                'quartile_1': 15.0,
                'quartile_3': 15.0,
                'skewness': 0,
                'std': 0.0,
            },
            'value_probabilities_aggregate': {
                'count': 3,
                'max': 0.3333333333333333,
                'mean': 0.3333333333333333,
                'median': 0.3333333333333333,
                'min': 0.3333333333333333,
                'quartile_1': 0.3333333333333333,
                'quartile_3': 0.3333333333333333,
                'skewness': 0,
                'std': 0.0,
            },
            'values_aggregate': {
                'count': 45,
                'kurtosis': -1.5348837209302326,
                'max': 1262304000.0,
                'mean': 946713600.0,
                'median': 946684800.0,
                'min': 631152000.0,
                'quartile_1': 631152000.0,
                'quartile_3': 1262304000.0,
                'skewness': 0.00017349603091112943,
                'std': 260578306.67149138,
            },
        }))

        expected_metafeatures = {
            'categorical_noise_to_signal_ratio': 6.856024896846719,
            'discrete_noise_to_signal_ratio': 16.280596971377722,
            'entropy_of_values': 1.2922333886497557,
            'equivalent_number_of_attributes': 7.497510695804063,
            'equivalent_number_of_categorical_attributes': 7.497510695804063,
            'equivalent_number_of_discrete_attributes': 24.925850557201,
            'equivalent_number_of_numeric_attributes': 24.925850557201,
            'joint_entropy_of_attributes': {
                'count': 4,
                'kurtosis': 3.8310594212937232,
                'max': 0.27405736318703244,
                'mean': 0.11209904602421886,
                'median': 0.06401513288957879,
                'min': 0.04630855513068542,
                'quartile_1': 0.05461037397689525,
                'quartile_3': 0.12150380493690241,
                'skewness': 1.949786087429789,
                'std': 0.10842988984399864,
            },
            'joint_entropy_of_categorical_attributes': {
                'count': 2,
                'max': 2.6276139378968235,
                'mean': 2.473903498180581,
                'median': 2.473903498180581,
                'min': 2.3201930584643393,
                'quartile_1': 2.3970482783224605,
                'quartile_3': 2.5507587180387024,
                'std': 0.2173793885250416,
            },
            'joint_entropy_of_discrete_attributes': {
                'count': 2,
                'max': 2.3334680303922335,
                'mean': 2.139600733638498,
                'median': 2.139600733638498,
                'min': 1.945733436884763,
                'quartile_1': 2.0426670852616304,
                'quartile_3': 2.236534382015366,
                'std': 0.2741697603697419,
            },
            'joint_entropy_of_numeric_attributes': {
                'count': 2,
                'max': 2.3334680303922335,
                'mean': 2.139600733638498,
                'median': 2.139600733638498,
                'min': 1.945733436884763,
                'quartile_1': 2.0426670852616304,
                'quartile_3': 2.236534382015366,
                'std': 0.2741697603697419,
            },
            'mutual_information_of_attributes': {
                'count': 2,
                'max': 0.27405736318703244,
                'mean': 0.17235499102027907,
                'median': 0.17235499102027907,
                'min': 0.07065261885352572,
                'quartile_1': 0.12150380493690241,
                'quartile_3': 0.22320617710365576,
                'std': 0.1438288740437386,
            },
            'mutual_information_of_categorical_attributes': {
                'count': 2,
                'max': 0.27405736318703244,
                'mean': 0.17235499102027907,
                'median': 0.17235499102027907,
                'min': 0.07065261885352572,
                'quartile_1': 0.12150380493690241,
                'quartile_3': 0.22320617710365576,
                'std': 0.1438288740437386,
            },
            'mutual_information_of_discrete_attributes': {
                'count': 2,
                'max': 0.05737764692563185,
                'mean': 0.05184310102815864,
                'median': 0.05184310102815864,
                'min': 0.04630855513068542,
                'quartile_1': 0.049075828079422026,
                'quartile_3': 0.05461037397689525,
                'std': 0.007827029869782995,
            },
            'mutual_information_of_numeric_attributes': {
                'count': 2,
                'max': 0.05737764692563185,
                'mean': 0.05184310102815864,
                'median': 0.05184310102815864,
                'min': 0.04630855513068542,
                'quartile_1': 0.049075828079422026,
                'quartile_3': 0.05461037397689525,
                'std': 0.007827029869782995,
            },
            'noise_to_signal_ratio': 5.526950051885679,
            'number_distinct_values': 45,
            'number_of_missing_values': 0,
            'number_of_negative_numeric_values': 0,
            'number_of_numeric_values': 45,
            'number_of_numeric_values_equal_-1': 0,
            'number_of_numeric_values_equal_0': 0,
            'number_of_numeric_values_equal_1': 0,
            'number_of_positive_numeric_values': 45,
            'number_of_present_values': 45,
            'numeric_noise_to_signal_ratio': 16.280596971377722,
            'ratio_of_missing_values': 0.0,
            'ratio_of_negative_numeric_values': 0.0,
            'ratio_of_numeric_values': 1.0,
            'ratio_of_numeric_values_equal_-1': 0.0,
            'ratio_of_numeric_values_equal_0': 0.0,
            'ratio_of_numeric_values_equal_1': 0.0,
            'ratio_of_positive_numeric_values': 1.0,
            'ratio_of_present_values': 1.0,
            'value_counts_aggregate': {
                'count': 4,
                'kurtosis': 0.2795705816375573,
                'max': 19,
                'mean': 11.25,
                'median': 10.0,
                'min': 6,
                'quartile_1': 7.5,
                'quartile_3': 13.75,
                'skewness': 1.0126926768695854,
                'std': 5.737304826019502,
            },
            'value_probabilities_aggregate': {
                'count': 4,
                'kurtosis': 0.2795705816375609,
                'max': 0.4222222222222222,
                'mean': 0.25,
                'median': 0.2222222222222222,
                'min': 0.13333333333333333,
                'quartile_1': 0.16666666666666666,
                'quartile_3': 0.3055555555555556,
                'skewness': 1.0126926768695859,
                'std': 0.12749566280043337,
            },
            'values_aggregate': {
                'count': 45,
                'kurtosis': -1.376558337329924,
                'max': 70.8170731707317,
                'mean': 54.363425575007106,
                'median': 53.6699876392329,
                'min': 32.328512195122,
                'quartile_1': 45.648691933945,
                'quartile_3': 65.5693658536586,
                'skewness': -0.11742803570367141,
                'std': 11.607381033992365,
            },
        }

        if parse_categorical_columns:
            # Because the order of string values is different from the order of encoded values,
            # the numbers are slightly different between parsed and not parsed cases.
            expected_metafeatures['joint_entropy_of_categorical_attributes'] = {
                'count': 2,
                'max': 2.6276139378968226,
                'mean': 2.473903498180581,
                'median': 2.473903498180581,
                'min': 2.3201930584643393,
                'quartile_1': 2.39704827832246,
                'quartile_3': 2.550758718038702,
                'std': 0.217379388525041,
            }
            expected_metafeatures['joint_entropy_of_attributes'] = {
                'count': 4,
                'kurtosis': 3.8310594212937232,
                'max': 0.27405736318703244,
                'mean': 0.11209904602421886,
                'median': 0.06401513288957879,
                'min': 0.04630855513068542,
                'quartile_1': 0.05461037397689525,
                'quartile_3': 0.12150380493690241,
                'skewness': 1.949786087429789,
                'std': 0.10842988984399864,
            }

        self.assertEqual(round_numbers(test_utils.convert_through_json(metadata.query_column(7)['data_metafeatures'])), round_numbers(expected_metafeatures))


if __name__ == '__main__':
    unittest.main()
