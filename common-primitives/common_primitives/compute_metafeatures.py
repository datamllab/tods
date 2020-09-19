import collections
import copy
import os
import typing

import numpy  # type: ignore
import pandas  # type: ignore
from scipy import stats  # type: ignore
from sklearn import metrics  # type: ignore

from d3m import container, utils as d3m_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer

import common_primitives

__all__ = ('ComputeMetafeaturesPrimitive',)

Inputs = container.DataFrame
Outputs = container.DataFrame


class Hyperparams(hyperparams.Hyperparams):
    pass


class ComputeMetafeaturesPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    A primitive which computes meta-features and adds them to metadata.

    Primitive is meant to be used with columns already parsed.
    """

    __author__ = 'Mingjie Sun <sunmj15@gmail.com>'
    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '915832a4-8059-438d-9118-f4fb4f7b0aaf',
            'version': '0.1.0',
            'name': "Compute meta-features",
            'python_path': 'd3m.primitives.metalearning.metafeature_extractor.Common',
            'source': {
                'name': common_primitives.__author__,
                'contact': 'mailto:sunmj15@gmail.com',
                'uris': [
                    'https://gitlab.com/datadrivendiscovery/common-primitives/blob/master/common_primitives/compute_metafeatures.py',
                    'https://gitlab.com/datadrivendiscovery/common-primitives.git',
                ],
            },
            'installation': [{
               'type': metadata_base.PrimitiveInstallationType.PIP,
               'package_uri': 'git+https://gitlab.com/datadrivendiscovery/common-primitives.git@{git_commit}#egg=common_primitives'.format(
                   git_commit=d3m_utils.current_git_commit(os.path.dirname(__file__)),
               ),
            }],
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.DATA_PROFILING,
                metadata_base.PrimitiveAlgorithmType.MUTUAL_INFORMATION,
                metadata_base.PrimitiveAlgorithmType.SIGNAL_TO_NOISE_RATIO,
                metadata_base.PrimitiveAlgorithmType.INFORMATION_ENTROPY,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.METALEARNING,
        },
    )

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        attribute_columns_indices_list = inputs.metadata.list_columns_with_semantic_types(['https://metadata.datadrivendiscovery.org/types/Attribute'])
        attribute_columns_indices = set(attribute_columns_indices_list)
        target_columns_indices_list = inputs.metadata.list_columns_with_semantic_types(['https://metadata.datadrivendiscovery.org/types/TrueTarget'])
        target_columns_indices = set(target_columns_indices_list)

        string_columns_indices = set(inputs.metadata.list_columns_with_structural_types((str,)))
        numeric_columns_indices = set(inputs.metadata.list_columns_with_structural_types(d3m_utils.is_numeric))
        discrete_columns_indices = self._get_discrete_indices(inputs, numeric_columns_indices)
        binary_columns_indices = self._get_binary_indices(inputs, discrete_columns_indices)

        # Categorical columns can be represented with number or strings.
        categorical_columns_indices = set(inputs.metadata.list_columns_with_semantic_types(['https://metadata.datadrivendiscovery.org/types/CategoricalData']))

        attributes_metafeatures = [
            self._attribute_metafeatures(
                inputs,
                index,
                index in string_columns_indices,
                index in numeric_columns_indices,
                index in categorical_columns_indices,
                index in discrete_columns_indices,
            ) for index in attribute_columns_indices_list
        ]
        targets_metafeatures = [
            self._target_metafeatures(
                inputs,
                index,
                index in string_columns_indices,
                index in numeric_columns_indices,
                index in categorical_columns_indices,
                index in discrete_columns_indices,
                attribute_columns_indices_list,
                string_columns_indices,
                numeric_columns_indices,
                discrete_columns_indices,
                categorical_columns_indices,
            ) for index in target_columns_indices_list
        ]

        # Our flags are slightly different from metafeatures schema. Our columns can be both categorical and
        # discrete, for example. Or both categorical and string. So we make sure here that we are stricter.
        strict_string_columns_indices = string_columns_indices - categorical_columns_indices
        strict_numeric_columns_indices = numeric_columns_indices - categorical_columns_indices
        strict_discrete_columns_indices = discrete_columns_indices - categorical_columns_indices
        strict_binary_columns_indices = binary_columns_indices - categorical_columns_indices

        table_metafeatures = {
            'number_of_attributes': len(attribute_columns_indices),
            'number_of_instances': inputs.shape[0],
            'number_of_categorical_attributes': len(categorical_columns_indices & attribute_columns_indices),
            'number_of_string_attributes': len(strict_string_columns_indices & attribute_columns_indices),
            'number_of_numeric_attributes': len(strict_numeric_columns_indices & attribute_columns_indices),
            'number_of_discrete_attributes': len(strict_discrete_columns_indices & attribute_columns_indices),
            'number_of_binary_attributes': len(strict_binary_columns_indices & attribute_columns_indices),
        }

        if table_metafeatures['number_of_instances']:
            table_metafeatures['dimensionality'] = table_metafeatures['number_of_attributes'] / table_metafeatures['number_of_instances']
        table_metafeatures['number_of_other_attributes'] = table_metafeatures['number_of_attributes'] - table_metafeatures['number_of_categorical_attributes'] - \
            table_metafeatures['number_of_string_attributes'] - table_metafeatures['number_of_numeric_attributes']

        if table_metafeatures['number_of_attributes']:
            table_metafeatures['ratio_of_categorical_attributes'] = table_metafeatures['number_of_categorical_attributes'] / table_metafeatures['number_of_attributes']
            table_metafeatures['ratio_of_string_attributes'] = table_metafeatures['number_of_string_attributes'] / table_metafeatures['number_of_attributes']
            table_metafeatures['ratio_of_numeric_attributes'] = table_metafeatures['number_of_numeric_attributes'] / table_metafeatures['number_of_attributes']
            table_metafeatures['ratio_of_discrete_attributes'] = table_metafeatures['number_of_discrete_attributes'] / table_metafeatures['number_of_attributes']
            table_metafeatures['ratio_of_binary_attributes'] = table_metafeatures['number_of_binary_attributes'] / table_metafeatures['number_of_attributes']
            table_metafeatures['ratio_of_other_attributes'] = table_metafeatures['number_of_other_attributes'] / table_metafeatures['number_of_attributes']

        table_metafeatures['number_of_instances_with_missing_values'] = self._get_number_of_instances_with_missing_values(inputs, attribute_columns_indices, string_columns_indices)
        if table_metafeatures['number_of_instances']:
            table_metafeatures['ratio_of_instances_with_missing_values'] = table_metafeatures['number_of_instances_with_missing_values'] / table_metafeatures['number_of_instances']
        table_metafeatures['number_of_instances_with_present_values'] = self._get_number_of_instances_with_present_values(inputs, attribute_columns_indices, string_columns_indices)
        if table_metafeatures['number_of_instances']:
            table_metafeatures['ratio_of_instances_with_present_values'] = table_metafeatures['number_of_instances_with_present_values'] / table_metafeatures['number_of_instances']

        attribute_counts_by_structural_type = self._get_counts_by_structural_type(inputs, attribute_columns_indices)
        if attribute_counts_by_structural_type:
            table_metafeatures['attribute_counts_by_structural_type'] = attribute_counts_by_structural_type
            if len(attribute_columns_indices):
                table_metafeatures['attribute_ratios_by_structural_type'] = {key: value / len(attribute_columns_indices) for key, value in attribute_counts_by_structural_type.items()}

        attribute_counts_by_semantic_type = self._get_counts_by_semantic_type(inputs, attribute_columns_indices)
        if attribute_counts_by_semantic_type:
            table_metafeatures['attribute_counts_by_semantic_type'] = attribute_counts_by_semantic_type
            if len(attribute_columns_indices):
                table_metafeatures['attribute_ratios_by_semantic_type'] = {key: value / len(attribute_columns_indices) for key, value in attribute_counts_by_semantic_type.items()}

        mean_of_attributes = self._aggregate([
            attributes_metafeatures[i]['values_aggregate']['mean'] for i, index in enumerate(attribute_columns_indices_list) if index in strict_numeric_columns_indices
        ])
        if mean_of_attributes is not None:
            table_metafeatures['mean_of_attributes'] = mean_of_attributes
        standard_deviation_of_attributes = self._aggregate([
            attributes_metafeatures[i]['values_aggregate']['std'] for i, index in enumerate(attribute_columns_indices_list) if index in strict_numeric_columns_indices
        ])
        if standard_deviation_of_attributes is not None:
            table_metafeatures['standard_deviation_of_attributes'] = standard_deviation_of_attributes
        kurtosis_of_attributes = self._aggregate([
            attributes_metafeatures[i]['values_aggregate']['kurtosis'] for i, index in enumerate(attribute_columns_indices_list) if index in strict_numeric_columns_indices
        ])
        if kurtosis_of_attributes is not None:
            table_metafeatures['kurtosis_of_attributes'] = kurtosis_of_attributes
        skew_of_attributes = self._aggregate([
            attributes_metafeatures[i]['values_aggregate']['skewness'] for i, index in enumerate(attribute_columns_indices_list) if index in strict_numeric_columns_indices
        ])
        if skew_of_attributes is not None:
            table_metafeatures['skew_of_attributes'] = skew_of_attributes

        entropy_of_categorical_attributes = self._aggregate([
            attributes_metafeatures[i]['entropy_of_values'] for i, index in enumerate(attribute_columns_indices_list) if index in categorical_columns_indices
        ])
        if entropy_of_categorical_attributes is not None:
            table_metafeatures['entropy_of_categorical_attributes'] = entropy_of_categorical_attributes
        entropy_of_numeric_attributes = self._aggregate([
            attributes_metafeatures[i]['entropy_of_values'] for i, index in enumerate(attribute_columns_indices_list) if index in strict_numeric_columns_indices
        ])
        if entropy_of_numeric_attributes is not None:
            table_metafeatures['entropy_of_numeric_attributes'] = entropy_of_numeric_attributes
        entropy_of_discrete_attributes = self._aggregate([
            attributes_metafeatures[i]['entropy_of_values'] for i, index in enumerate(attribute_columns_indices_list) if index in strict_discrete_columns_indices
        ])
        if entropy_of_discrete_attributes is not None:
            table_metafeatures['entropy_of_discrete_attributes'] = entropy_of_discrete_attributes
        entropy_of_attributes = self._aggregate([
            attributes_metafeatures[i]['entropy_of_values'] for i, index in enumerate(attribute_columns_indices_list) if index in strict_numeric_columns_indices | categorical_columns_indices
        ])
        if entropy_of_attributes is not None:
            table_metafeatures['entropy_of_attributes'] = entropy_of_attributes

        number_distinct_values_of_categorical_attributes = self._aggregate([
            attributes_metafeatures[i]['number_distinct_values'] for i, index in enumerate(attribute_columns_indices_list) if index in categorical_columns_indices
        ])
        if number_distinct_values_of_categorical_attributes is not None:
            table_metafeatures['number_distinct_values_of_categorical_attributes'] = number_distinct_values_of_categorical_attributes
        number_distinct_values_of_numeric_attributes = self._aggregate([
            attributes_metafeatures[i]['number_distinct_values'] for i, index in enumerate(attribute_columns_indices_list) if index in strict_numeric_columns_indices
        ])
        if number_distinct_values_of_numeric_attributes is not None:
            table_metafeatures['number_distinct_values_of_numeric_attributes'] = number_distinct_values_of_numeric_attributes
        number_distinct_values_of_discrete_attributes = self._aggregate([
            attributes_metafeatures[i]['number_distinct_values'] for i, index in enumerate(attribute_columns_indices_list) if index in strict_discrete_columns_indices
        ])
        if number_distinct_values_of_discrete_attributes is not None:
            table_metafeatures['number_distinct_values_of_discrete_attributes'] = number_distinct_values_of_discrete_attributes

        for i, index in enumerate(target_columns_indices_list):
            if 'mutual_information_of_categorical_attributes' in targets_metafeatures[i] and 'entropy_of_categorical_attributes' in table_metafeatures:
                mutual_information_of_categorical_attributes = targets_metafeatures[i]['mutual_information_of_categorical_attributes']['mean']
                if mutual_information_of_categorical_attributes:
                    targets_metafeatures[i]['categorical_noise_to_signal_ratio'] = (table_metafeatures['entropy_of_categorical_attributes']['mean'] - mutual_information_of_categorical_attributes) / \
                        mutual_information_of_categorical_attributes
            if 'mutual_information_of_numeric_attributes' in targets_metafeatures[i] and 'entropy_of_numeric_attributes' in table_metafeatures:
                mutual_information_of_numeric_attributes = targets_metafeatures[i]['mutual_information_of_numeric_attributes']['mean']
                if mutual_information_of_numeric_attributes:
                    targets_metafeatures[i]['numeric_noise_to_signal_ratio'] = (table_metafeatures['entropy_of_numeric_attributes']['mean'] - mutual_information_of_numeric_attributes) / \
                        mutual_information_of_numeric_attributes
            if 'mutual_information_of_discrete_attributes' in targets_metafeatures[i] and 'entropy_of_discrete_attributes' in table_metafeatures:
                mutual_information_of_discrete_attributes = targets_metafeatures[i]['mutual_information_of_discrete_attributes']['mean']
                if mutual_information_of_discrete_attributes:
                    targets_metafeatures[i]['discrete_noise_to_signal_ratio'] = (table_metafeatures['entropy_of_discrete_attributes']['mean'] - mutual_information_of_discrete_attributes) / \
                        mutual_information_of_discrete_attributes
            if 'mutual_information_of_attributes' in targets_metafeatures[i] and 'entropy_of_attributes' in table_metafeatures:
                mutual_information_of_attributes = targets_metafeatures[i]['mutual_information_of_attributes']['mean']
                if mutual_information_of_attributes:
                    targets_metafeatures[i]['noise_to_signal_ratio'] = (table_metafeatures['entropy_of_attributes']['mean'] - mutual_information_of_attributes) / \
                        mutual_information_of_attributes

        outputs = copy.copy(inputs)
        outputs.metadata = inputs.metadata

        if table_metafeatures:
            outputs.metadata = outputs.metadata.update((), {'data_metafeatures': table_metafeatures})

        for i, index in enumerate(attribute_columns_indices_list):
            if attributes_metafeatures[i]:
                outputs.metadata = outputs.metadata.update_column(index, {'data_metafeatures': attributes_metafeatures[i]})

        for i, index in enumerate(target_columns_indices_list):
            if targets_metafeatures[i]:
                outputs.metadata = outputs.metadata.update_column(index, {'data_metafeatures': targets_metafeatures[i]})

        return base.CallResult(outputs)

    def _get_discrete_indices(self, columns: container.DataFrame, numeric_columns_indices: typing.Set[int]) -> typing.Set[int]:
        known_discrete_indices = columns.metadata.list_columns_with_structural_types((int, numpy.integer))

        discrete_indices = set()

        for index in numeric_columns_indices:
            if index in known_discrete_indices:
                discrete_indices.add(index)
                continue

            # Even if structural type is float, it could still be a discrete column
            # where all values are discrete, but column contains also NaN values and this
            # is why its structural type is float.
            assert d3m_utils.is_float(columns.metadata.query_column(index)['structural_type']), columns.metadata.query_column(index)['structural_type']
            assert d3m_utils.is_float(columns.dtypes[index].type), columns.dtypes[index].type

            # If all values are or integers or NaN values.
            if all(v.is_integer() for v in columns.iloc[:, index].dropna()):
                discrete_indices.add(index)

        return discrete_indices

    def _get_binary_indices(self, columns: container.DataFrame, discrete_columns_indices: typing.Set[int]) -> typing.Set[int]:
        binary_indices = set()

        for index in discrete_columns_indices:
            values: typing.Set[typing.Any] = set()

            for value in columns.iloc[:, index].dropna():
                if value in values:
                    continue
                values.add(value)

                if len(values) > 2:
                    break

            if len(values) == 2:
                binary_indices.add(index)

        return binary_indices

    @classmethod
    def _get_number_of_instances_with_missing_values(cls, columns: container.DataFrame, attribute_columns_indices: typing.Set[int], string_columns_indices: typing.Set[int]) -> int:
        number_of_instances_with_missing_values = 0

        for row in columns.itertuples(index=False, name=None):
            has_missing_values = False

            for column_index, column_value in enumerate(row):
                if column_index not in attribute_columns_indices:
                    continue

                if column_index in string_columns_indices:
                    if column_value == '':
                        has_missing_values = True
                        break
                else:
                    if pandas.isna(column_value):
                        has_missing_values = True
                        break

            if has_missing_values:
                number_of_instances_with_missing_values += 1

        return number_of_instances_with_missing_values

    def _get_number_of_instances_with_present_values(self, columns: container.DataFrame, attribute_columns_indices: typing.Set[int], string_columns_indices: typing.Set[int]) -> int:
        number_of_instances_with_present_values = 0

        for row in columns.itertuples(index=False, name=None):
            has_present_values = False

            for column_index, column_value in enumerate(row):
                if column_index not in attribute_columns_indices:
                    continue

                if column_index in string_columns_indices:
                    if column_value != '':
                        has_present_values = True
                        break
                else:
                    if not pandas.isna(column_value):
                        has_present_values = True
                        break

            if has_present_values:
                number_of_instances_with_present_values += 1

        return number_of_instances_with_present_values

    @classmethod
    def _get_counts_by_structural_type(cls, columns: container.DataFrame, columns_indices: typing.Iterable[int]) -> typing.Dict[str, int]:
        counts: typing.Dict[str, int] = collections.defaultdict(int)

        for index in columns_indices:
            counts[d3m_utils.type_to_str(columns.metadata.query_column(index)['structural_type'])] += 1

        return dict(counts)

    @classmethod
    def _get_counts_by_semantic_type(cls, columns: container.DataFrame, columns_indices: typing.Iterable[int]) -> typing.Dict[str, int]:
        counts: typing.Dict[str, int] = collections.defaultdict(int)

        for index in columns_indices:
            for semantic_type in columns.metadata.query_column(index).get('semantic_types', []):
                counts[semantic_type] += 1

        return dict(counts)

    def _columns_metafeatures(self, columns: container.DataFrame, index: int, is_string: bool, is_numeric: bool, is_categorical: bool, is_discrete: bool) -> typing.Dict[str, typing.Any]:
        column_metafeatures: typing.Dict[str, typing.Any] = {}

        values = columns.iloc[:, index]

        if is_string:
            if is_categorical:
                # Categorical string values have missing data represented as empty strings.
                values_without_na = values.replace('', numpy.nan).dropna()
            else:
                values_without_na = values
        elif is_numeric:
            values_without_na = values.dropna()
        else:
            values_without_na = values

        if is_string or is_numeric:
            column_metafeatures['number_of_missing_values'] = len(values) - len(values_without_na)
            if len(values):
                column_metafeatures['ratio_of_missing_values'] = column_metafeatures['number_of_missing_values'] / len(values)
            column_metafeatures['number_of_present_values'] = len(values_without_na)
            if len(values):
                column_metafeatures['ratio_of_present_values'] = column_metafeatures['number_of_present_values'] / len(values)

        if is_numeric or is_categorical:
            discrete_values = self._discretize(values_without_na, is_string, is_categorical, is_discrete)

            # There should be no NA anyway anymore.
            value_counts = discrete_values.value_counts(dropna=False)
            assert len(values_without_na) == value_counts.sum(), (len(values_without_na), value_counts.sum())

            if len(values_without_na):
                value_counts_normalized = value_counts / len(values_without_na)
            else:
                value_counts_normalized = None

            if is_categorical or is_discrete:
                column_metafeatures['number_distinct_values'] = value_counts.size
            else:
                column_metafeatures['number_distinct_values'] = len(values_without_na.unique())
            if value_counts_normalized is not None:
                column_metafeatures['entropy_of_values'] = stats.entropy(value_counts_normalized)
            value_counts_aggregate = self._aggregate(value_counts)
            if value_counts_aggregate is not None:
                column_metafeatures['value_counts_aggregate'] = value_counts_aggregate
            if value_counts_normalized is not None:
                value_probabilities_aggregate = self._aggregate(value_counts_normalized)
                if value_probabilities_aggregate is not None:
                    column_metafeatures['value_probabilities_aggregate'] = value_probabilities_aggregate

        if is_numeric:
            values_aggregate = self._aggregate(values_without_na)
            if values_aggregate is not None:
                column_metafeatures['values_aggregate'] = values_aggregate

        # Our flags are slightly different from metafeatures schema. Our columns can be both categorical and
        # discrete, for example. Or both categorical and string. So we make sure here that we are stricter.
        if is_numeric and not is_categorical:
            column_metafeatures['number_of_numeric_values'] = column_metafeatures['number_of_present_values']
            column_metafeatures['ratio_of_numeric_values'] = column_metafeatures['ratio_of_present_values']
            column_metafeatures['number_of_positive_numeric_values'] = int((values_without_na > 0).sum())
            if len(values):
                column_metafeatures['ratio_of_positive_numeric_values'] = column_metafeatures['number_of_positive_numeric_values'] / len(values)
            column_metafeatures['number_of_negative_numeric_values'] = int((values_without_na < 0).sum())
            if len(values):
                column_metafeatures['ratio_of_negative_numeric_values'] = column_metafeatures['number_of_negative_numeric_values'] / len(values)
            column_metafeatures['number_of_numeric_values_equal_0'] = int((values_without_na == 0).sum())
            if len(values):
                column_metafeatures['ratio_of_numeric_values_equal_0'] = column_metafeatures['number_of_numeric_values_equal_0'] / len(values)
            column_metafeatures['number_of_numeric_values_equal_1'] = int((values_without_na == 1).sum())
            if len(values):
                column_metafeatures['ratio_of_numeric_values_equal_1'] = column_metafeatures['number_of_numeric_values_equal_1'] / len(values)
            column_metafeatures['number_of_numeric_values_equal_-1'] = int((values_without_na == -1).sum())
            if len(values):
                column_metafeatures['ratio_of_numeric_values_equal_-1'] = column_metafeatures['number_of_numeric_values_equal_-1'] / len(values)

        return column_metafeatures

    def _attribute_metafeatures(self, columns: container.DataFrame, index: int, is_string: bool, is_numeric: bool, is_categorical: bool, is_discrete: bool) -> typing.Dict[str, typing.Any]:
        return self._columns_metafeatures(columns, index, is_string, is_numeric, is_categorical, is_discrete)

    def _target_metafeatures(self, columns: container.DataFrame, target_index: int, is_string: bool, is_numeric: bool, is_categorical: bool, is_discrete: bool,
                             attribute_columns_indices_list: typing.Sequence[int], string_columns_indices: typing.Set[int], numeric_columns_indices: typing.Set[int],
                             discrete_columns_indices: typing.Set[int], categorical_columns_indices: typing.Set[int]) -> typing.Dict[str, typing.Any]:
        metafeatures = self._columns_metafeatures(columns, target_index, is_string, is_numeric, is_categorical, is_discrete)

        if is_categorical:
            if 'value_probabilities_aggregate' in metafeatures:
                metafeatures['default_accuracy'] = metafeatures['value_probabilities_aggregate']['max']

        if is_numeric or is_categorical:
            categorical_joint_entropy = []
            numeric_joint_entropy = []
            discrete_joint_entropy = []
            all_joint_entropy = []

            categorical_mutual_information = []
            numeric_mutual_information = []
            discrete_mutual_information = []
            all_mutual_information = []

            discrete_target_values = self._discretize(columns.iloc[:, target_index], is_string, is_categorical, is_discrete)

            for attribute_index in attribute_columns_indices_list:
                attribute_is_string = attribute_index in string_columns_indices
                attribute_is_numeric = attribute_index in numeric_columns_indices
                attribute_is_categorical = attribute_index in categorical_columns_indices
                attribute_is_discrete = attribute_index in discrete_columns_indices

                if not (attribute_is_numeric or attribute_is_categorical):
                    continue

                discrete_attribute_values = self._discretize(columns.iloc[:, attribute_index], attribute_is_string, attribute_is_categorical, attribute_is_discrete)

                all_values_without_na = pandas.concat([discrete_attribute_values, discrete_target_values], axis=1).dropna(axis=0, how='any')

                attribute_values_without_na = all_values_without_na.iloc[:, 0]
                target_values_without_na = all_values_without_na.iloc[:, 1]

                probabilities = []
                # We sort so that we always traverse in the same order so that floating point
                # operations are always in the same order to produce exactly the same results.
                for attribute_value in sorted(set(attribute_values_without_na)):
                    for target_value in sorted(set(target_values_without_na)):
                        probabilities.append(numpy.mean(numpy.logical_and(attribute_values_without_na == attribute_value, target_values_without_na == target_value)))

                joint_entropy = stats.entropy(probabilities)
                mutual_information = metrics.mutual_info_score(attribute_values_without_na, target_values_without_na)

                # Our flags are slightly different from metafeatures schema. Our columns can be both categorical and
                # discrete, for example. Or both categorical and string. So we make sure here that we are stricter.
                if attribute_is_categorical:
                    categorical_joint_entropy.append(joint_entropy)
                    categorical_mutual_information.append(mutual_information)
                if attribute_is_numeric and not attribute_is_categorical:
                    numeric_joint_entropy.append(joint_entropy)
                    numeric_mutual_information.append(mutual_information)
                if attribute_is_discrete and not attribute_is_categorical:
                    discrete_joint_entropy.append(joint_entropy)
                    discrete_mutual_information.append(mutual_information)
                all_joint_entropy.append(joint_entropy)
                all_mutual_information.append(mutual_information)

            if categorical_joint_entropy:
                joint_entropy_of_categorical_attributes = self._aggregate(categorical_joint_entropy)
                if joint_entropy_of_categorical_attributes is not None:
                    metafeatures['joint_entropy_of_categorical_attributes'] = joint_entropy_of_categorical_attributes
            if categorical_mutual_information:
                mutual_information_of_categorical_attributes = self._aggregate(categorical_mutual_information)
                if mutual_information_of_categorical_attributes is not None:
                    metafeatures['mutual_information_of_categorical_attributes'] = mutual_information_of_categorical_attributes
                    metafeatures['equivalent_number_of_categorical_attributes'] = metafeatures['entropy_of_values'] / mutual_information_of_categorical_attributes['mean']
            if numeric_joint_entropy:
                joint_entropy_of_numeric_attributes = self._aggregate(numeric_joint_entropy)
                if joint_entropy_of_numeric_attributes is not None:
                    metafeatures['joint_entropy_of_numeric_attributes'] = joint_entropy_of_numeric_attributes
            if numeric_mutual_information:
                mutual_information_of_numeric_attributes = self._aggregate(numeric_mutual_information)
                if mutual_information_of_numeric_attributes is not None:
                    metafeatures['mutual_information_of_numeric_attributes'] = mutual_information_of_numeric_attributes
                    metafeatures['equivalent_number_of_numeric_attributes'] = metafeatures['entropy_of_values'] / mutual_information_of_numeric_attributes['mean']
            if discrete_joint_entropy:
                joint_entropy_of_discrete_attributes = self._aggregate(discrete_joint_entropy)
                if joint_entropy_of_discrete_attributes is not None:
                    metafeatures['joint_entropy_of_discrete_attributes'] = joint_entropy_of_discrete_attributes
            if discrete_mutual_information:
                mutual_information_of_discrete_attributes = self._aggregate(discrete_mutual_information)
                if mutual_information_of_discrete_attributes is not None:
                    metafeatures['mutual_information_of_discrete_attributes'] = mutual_information_of_discrete_attributes
                    metafeatures['equivalent_number_of_discrete_attributes'] = metafeatures['entropy_of_values'] / mutual_information_of_discrete_attributes['mean']
            if all_joint_entropy:
                joint_entropy_of_attributes = self._aggregate(all_mutual_information)
                if joint_entropy_of_attributes is not None:
                    metafeatures['joint_entropy_of_attributes'] = joint_entropy_of_attributes
            if all_mutual_information:
                mutual_information_of_attributes = self._aggregate(categorical_mutual_information)
                if mutual_information_of_attributes is not None:
                    metafeatures['mutual_information_of_attributes'] = mutual_information_of_attributes
                    metafeatures['equivalent_number_of_attributes'] = metafeatures['entropy_of_values'] / mutual_information_of_attributes['mean']

        return metafeatures

    def _discretize(self, values: typing.Sequence, is_string: bool, is_categorical: bool, is_discrete: bool) -> pandas.Series:
        if not isinstance(values, pandas.Series):
            values = pandas.Series(values)

        if is_discrete:
            # These can still be values with structural type float, but are in fact discrete
            # numbers which might contain NaN and this is why structural type is float.
            return values

        if is_string:
            # This means we have categorical string values.
            assert is_categorical

            # Categorical string values have missing data represented as empty strings.
            values = values.replace('', numpy.nan)

            # We leave values as strings and expect caller to specially handle this case if necessary.
            return values

        if is_categorical:
            # Categorical values should be only strings or discrete numbers, but if we got to
            # here this is not really true, but we cannot really do anything.
            return values

        # If we got to here we have true numeric values, and we have to bin them.

        values = pandas.Series(pandas.cut(values, round(len(values) ** (1/3)), include_lowest=True, labels=False))

        return values

    def _aggregate(self, values: typing.Sequence) -> typing.Optional[typing.Dict[str, typing.Any]]:
        if not isinstance(values, pandas.Series):
            values = pandas.Series(values)

        if not len(values):
            return None

        results = {
            'count': len(values),
            'min': values.min(),
            'max': values.max(),
            'mean': values.mean(),
            'median': values.median(),
            'std': values.std(),
            'quartile_1': values.quantile(0.25),
            'quartile_3': values.quantile(0.75),
            'kurtosis': values.kurtosis(),
            'skewness': values.skew(),
        }

        # We iterate over a list so that we can change dict while iterating.
        for name, value in list(results.items()):
            # If anything cannot be computed, we remove it.
            if not numpy.isfinite(value):
                del results[name]

        if not results:
            return None

        return results
