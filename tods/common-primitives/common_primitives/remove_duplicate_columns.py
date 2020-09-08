import copy
import itertools
import typing
import os

from d3m import container, exceptions, utils as d3m_utils
from d3m.base import utils as base_utils
from d3m.metadata import base as metadata_base, params, hyperparams
from d3m.primitive_interfaces import base, unsupervised_learning

import common_primitives

__all__ = ('RemoveDuplicateColumnsPrimitive',)

Inputs = container.DataFrame
Outputs = container.DataFrame


class Params(params.Params):
    equal_columns_map: typing.Optional[typing.Dict[int, typing.Set[int]]]


class Hyperparams(hyperparams.Hyperparams):
    use_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to force primitive to operate on. If any specified column is not a duplicate with any other specified, it is skipped.",
    )
    exclude_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to not operate on. Applicable only if \"use_columns\" is not provided.",
    )


# TODO: Compare columns also by determining if there exists a bijection between two columns and find such columns duplicate as well.
class RemoveDuplicateColumnsPrimitive(unsupervised_learning.UnsupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    A primitive which removes duplicate columns based on exact match in all their values.

    It adds names of removed columns into ``other_names`` metadata for columns remaining.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '130513b9-09ca-4785-b386-37ab31d0cf8b',
            'version': '0.1.0',
            'name': "Removes duplicate columns",
            'python_path': 'd3m.primitives.data_transformation.remove_duplicate_columns.Common',
            'source': {
                'name': common_primitives.__author__,
                'contact': 'mailto:mitar.commonprimitives@tnode.com',
                'uris': [
                    'https://gitlab.com/datadrivendiscovery/common-primitives/blob/master/common_primitives/remove_duplicate_columns.py',
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
                metadata_base.PrimitiveAlgorithmType.ARRAY_SLICING,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.DATA_TRANSFORMATION,
        },
    )

    def __init__(self, *, hyperparams: Hyperparams) -> None:
        super().__init__(hyperparams=hyperparams)

        self._training_inputs: Inputs = None
        self._equal_columns_map: typing.Optional[typing.Dict[int, typing.Set[int]]] = None
        self._fitted = False

    def set_training_data(self, *, inputs: Inputs) -> None:  # type: ignore
        self._training_inputs = inputs
        self._fitted = False

    def fit(self, *, timeout: float = None, iterations: int = None) -> base.CallResult[None]:
        if self._training_inputs is None:
            raise exceptions.InvalidStateError("Missing training data.")

        if self._fitted:
            return base.CallResult(None)

        columns_to_use = self._get_columns(self._training_inputs.metadata)
        columns_to_use_length = len(columns_to_use)

        equal_columns = []
        for i in range(columns_to_use_length):
            for j in range(i + 1, columns_to_use_length):
                if self._training_inputs.iloc[:, columns_to_use[i]].equals(self._training_inputs.iloc[:, columns_to_use[j]]):
                    equal_columns.append((i, j))

        # It might be that more columns are equal to each other, so we resolve those and
        # will keep only the first column and remove all others.
        equal_columns_map: typing.Dict[int, typing.Set[int]] = {}
        for i, j in equal_columns:
            for first, others in equal_columns_map.items():
                if first == i:
                    others.add(j)
                    break
                elif i in others:
                    others.add(j)
                    break
            else:
                equal_columns_map[i] = {j}

        self._equal_columns_map = equal_columns_map
        self._fitted = True

        return base.CallResult(None)

    def _get_columns(self, inputs_metadata: metadata_base.DataMetadata) -> typing.List[int]:
        def can_use_column(column_index: int) -> bool:
            return True

        columns_to_use, columns_not_to_use = base_utils.get_columns_to_use(inputs_metadata, self.hyperparams['use_columns'], self.hyperparams['exclude_columns'], can_use_column)

        return columns_to_use

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        if not self._fitted:
            raise exceptions.PrimitiveNotFittedError("Primitive not fitted.")

        outputs = copy.copy(inputs)

        # Set "other_names" metadata on columns remaining.
        for first, others in self._equal_columns_map.items():
            first_name = outputs.metadata.query_column(first).get('name', None)

            names = set()
            for other in others:
                other_metadata = outputs.metadata.query_column(other)
                # We do not care about empty strings for names either.
                if other_metadata.get('name', None):
                    if first_name != other_metadata['name']:
                        names.add(other_metadata['name'])

            first_other_names = list(outputs.metadata.query_column(first).get('other_names', []))
            first_other_names += sorted(names)
            if first_other_names:
                outputs.metadata = outputs.metadata.update_column(first, {
                    'other_names': first_other_names,
                })

        # We flatten all values of "equal_columns_map" into one list.
        outputs = outputs.remove_columns(list(itertools.chain.from_iterable(self._equal_columns_map.values())))

        return base.CallResult(outputs)

    def get_params(self) -> Params:
        if not self._fitted:
            return Params(
                equal_columns_map=None,
            )

        return Params(
            equal_columns_map=self._equal_columns_map,
        )

    def set_params(self, *, params: Params) -> None:
        self._equal_columns_map = params['equal_columns_map']
        self._fitted = params['equal_columns_map'] is not None
