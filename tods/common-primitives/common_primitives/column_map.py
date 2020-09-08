import collections
import copy
import os.path
import typing

from d3m import container, exceptions, index, utils as d3m_utils
from d3m.base import utils as base_utils
from d3m.metadata import base as metadata_base, hyperparams as hyperparams_module, params
from d3m.primitive_interfaces import base, transformer, unsupervised_learning

import common_primitives


Inputs = container.DataFrame
Outputs = container.DataFrame


class Params(params.Params):
    # For each column, for each cell in a column, we have potentially params of a primitive.
    columns_params: typing.Optional[typing.List[typing.List[params.Params]]]


class Hyperparams(hyperparams_module.Hyperparams):
    # TODO: How to specify that input type of allowed primitive has to be "DataFrame".
    #       See: https://gitlab.com/datadrivendiscovery/d3m/issues/335
    primitive = hyperparams_module.Union[typing.Union[transformer.TransformerPrimitiveBase, unsupervised_learning.UnsupervisedLearnerPrimitiveBase]](
        configuration=collections.OrderedDict(
            transformer=hyperparams_module.Primitive[transformer.TransformerPrimitiveBase](  # type: ignore
                # TODO: This default in fact gets List as input and produces List. Not DataFrame.
                #       But in fact it just passes through whatever it gets, so it works out.
                #       See: https://gitlab.com/datadrivendiscovery/d3m/issues/214
                default=index.get_primitive('d3m.primitives.operator.null.TransformerTest'),
                semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
                description="A transformer primitive.",
            ),
            unsupervised_learner=hyperparams_module.Primitive[unsupervised_learning.UnsupervisedLearnerPrimitiveBase](  # type: ignore
                # TODO: This default in fact gets List as input and produces List. Not DataFrame.
                #       But in fact it just passes through whatever it gets, so it works out.
                #       See: https://gitlab.com/datadrivendiscovery/d3m/issues/214
                default=index.get_primitive('d3m.primitives.operator.null.UnsupervisedLearnerTest'),
                semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
                description="An unsupervised learner primitive. If it is already fitted and you do not want to re-fit it, "
                            "set \"fit_primitive\" to \"no\".",
            ),
        ),
        default='transformer',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A primitive to use for mapping of each cell value. Has to take \"DataFrame\" as input.",
    )
    fit_primitive = hyperparams_module.Enumeration(
        values=['no', 'fit', 'continue_fit'],
        default='fit',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Fit an unsupervised learner primitive or not.",
    )
    produce_method = hyperparams_module.Hyperparameter[str](
        default='produce',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Name of primitive's produce method to use.",
    )
    use_columns = hyperparams_module.Set(
        elements=hyperparams_module.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to force primitive to operate on. If any specified column cannot be mapped, it is skipped.",
    )
    exclude_columns = hyperparams_module.Set(
        elements=hyperparams_module.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to not operate on. Applicable only if \"use_columns\" is not provided.",
    )
    return_result = hyperparams_module.Enumeration(
        values=['append', 'replace', 'new'],
        default='replace',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Should mapped columns be appended, should they replace original columns, or should only mapped columns be returned?",
    )
    add_index_columns = hyperparams_module.UniformBool(
        default=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Also include primary index columns if input data has them. Applicable only if \"return_result\" is set to \"new\".",
    )
    error_on_no_columns = hyperparams_module.UniformBool(
        default=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Throw an exception if no column is selected/provided. Otherwise issue a warning.",
    )


# TODO: Implement optimized "fit_multi_produce" which calls "fit_multi_produce" of underlying primitive.
class DataFrameColumnMapPrimitive(unsupervised_learning.UnsupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    A primitive which for every column with embedded ``DataFrame`` cells (by default)
    runs provided ``primitive`` on every cell's value, producing new cell's value.

    ``primitive`` can be transformer or fitted or unfitted unsupervised learner primitive.
    If it is already fitted and you do not want to re-fit it, set ``fit_primitive`` to ``no``.
    Otherwise, if ``fit_primitive`` is set to ``fit``, for each cell's value a copy of the
    primitive will be made and it will be first fitted and then produced on that value.
    If ``fit_primitive`` is set to ``continue_fit``, a copy of the primitive is made per
    column and it is continue fitted on all cell values in the column, in row order.

    Input to the ``primitive`` has to be container ``DataFrame``, but output can be any
    container type.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': 'fe58e7bb-f6c7-4d91-b897-69faf33bece5',
            'version': '0.1.0',
            'name': "Map DataFrame cell values to new values using provided primitive",
            'python_path': 'd3m.primitives.operator.column_map.Common',
            'source': {
                'name': common_primitives.__author__,
                'contact': 'mailto:mitar.commonprimitives@tnode.com',
                'uris': [
                    'https://gitlab.com/datadrivendiscovery/common-primitives/blob/master/common_primitives/column_map.py',
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
                # TODO: Change to "MAP".
                metadata_base.PrimitiveAlgorithmType.DATA_CONVERSION,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.OPERATOR,
        },
    )

    def __init__(self, *, hyperparams: Hyperparams) -> None:
        super().__init__(hyperparams=hyperparams)

        self._training_inputs: Inputs = None
        self._columns_primitives: typing.List[typing.List[base.PrimitiveBase]] = None
        self._fitted: bool = False

    def _should_fit(self) -> bool:
        if self.hyperparams['fit_primitive'] == 'no':
            return False

        if isinstance(self.hyperparams['primitive'], transformer.TransformerPrimitiveBase):
            return False

        if self.hyperparams['fit_primitive'] == 'continue_fit' and not isinstance(self.hyperparams['primitive'], base.ContinueFitMixin):
            raise exceptions.InvalidArgumentValueError("\"fit_primitive\" hyper-parameter is set to \"continue_fit\", but primitive does not inherit the \"ContinueFitMixin\" class.")

        return True

    def set_training_data(self, *, inputs: Inputs) -> None:  # type: ignore
        if not self._should_fit():
            return

        self._training_inputs = inputs
        self._fitted = False

    def fit(self, *, timeout: float = None, iterations: int = None) -> base.CallResult[None]:
        if not self._should_fit():
            return base.CallResult(None)

        if self._training_inputs is None:
            raise exceptions.InvalidStateError("Missing training data.")

        self._columns_primitives = self._fit_columns(self._training_inputs)
        self._fitted = True

        return base.CallResult(None)

    def _fit_columns(self, inputs: Inputs) -> typing.List[typing.List[base.PrimitiveBase]]:
        columns_to_use = self._get_columns(inputs.metadata)

        columns_primitives = []

        for column_index in columns_to_use:
            columns_primitives.append(self._fit_column(inputs, column_index))

        assert len(columns_primitives) == len(columns_to_use)

        return columns_primitives

    def _prepare_cell_value(self, inputs_metadata: metadata_base.DataMetadata, value: container.DataFrame, row_index: int, column_index: int) -> container.DataFrame:
        assert isinstance(value, container.DataFrame)

        value = copy.copy(value)

        value.metadata = metadata_base.DataMetadata({
            'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
        })

        value.metadata = inputs_metadata.copy_to(
            value.metadata,
            (row_index, column_index),
        )

        return value

    def _fit_column(self, inputs: Inputs, column_index: int) -> typing.List[base.PrimitiveBase]:
        column_primitives = []
        primitive = None

        for row_index, column_value in enumerate(inputs.iloc[:, column_index]):
            column_value = self._prepare_cell_value(inputs.metadata, column_value, row_index, column_index)

            # If "fit_primitive" is "continue_fit" we copy the primitive only once.
            if self.hyperparams['fit_primitive'] == 'fit' or primitive is None:
                primitive = copy.deepcopy(self.hyperparams['primitive'])
                column_primitives.append(primitive)

            primitive.set_training_data(inputs=column_value)

            if self.hyperparams['fit_primitive'] == 'fit':
                primitive.fit()
            else:
                assert self.hyperparams['fit_primitive'] == 'continue_fit'
                primitive.continue_fit()

        return column_primitives

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        if self._should_fit() and not self._fitted:
            raise exceptions.PrimitiveNotFittedError("Primitive not fitted.")

        assert self._should_fit() == self._fitted
        assert (self._columns_primitives is not None) == self._fitted

        if self.hyperparams['produce_method'] != 'produce' and not self.hyperparams['produce_method'].startswith('produce_'):
            raise exceptions.InvalidArgumentValueError(f"Invalid produce method name in \"produce_method\" hyper-parameter: {self.hyperparams['produce_method']}")

        columns_to_use, output_columns = self._produce_columns(inputs, self._columns_primitives)

        outputs = base_utils.combine_columns(inputs, columns_to_use, output_columns, return_result=self.hyperparams['return_result'], add_index_columns=self.hyperparams['add_index_columns'])

        return base.CallResult(outputs)

    def _can_use_column(self, inputs_metadata: metadata_base.DataMetadata, column_index: int) -> bool:
        structural_type = inputs_metadata.query_column_field(column_index, 'structural_type')

        return issubclass(structural_type, container.DataFrame)

    def _get_columns(self, inputs_metadata: metadata_base.DataMetadata) -> typing.List[int]:
        def can_use_column(column_index: int) -> bool:
            return self._can_use_column(inputs_metadata, column_index)

        columns_to_use, columns_not_to_use = base_utils.get_columns_to_use(inputs_metadata, self.hyperparams['use_columns'], self.hyperparams['exclude_columns'], can_use_column)

        if not columns_to_use:
            if self.hyperparams['error_on_no_columns']:
                raise ValueError("No inputs columns.")
            else:
                self.logger.warning("No inputs columns.")

        if self.hyperparams['use_columns'] and columns_not_to_use:
            self.logger.warning("Not all specified inputs columns can be used. Skipping columns: %(columns)s", {
                'columns': columns_not_to_use,
            })

        return columns_to_use

    def _produce_columns(
        self, inputs: Inputs, columns_primitives: typing.Optional[typing.List[typing.List[base.PrimitiveBase]]],
    ) -> typing.Tuple[typing.List[int], typing.List[Outputs]]:
        columns_to_use = self._get_columns(inputs.metadata)

        output_columns = []

        for column_index in columns_to_use:
            output_columns.append(self._produce_column(inputs, column_index, columns_primitives))

        assert len(columns_to_use) == len(output_columns)

        return columns_to_use, output_columns

    # TODO: Instead of copying metadata to a cell value and then back, we could maybe just hack it by setting a correct reference.
    #       So cell value metadata would point directly into dataframe column's (we would select input column
    #       first and just modify metadata directly there) metadata object for element corresponding to the cell value.
    #       How would that work if there is any metadata on dataframe's ALL_ELEMENTS? For updating it does not matter
    #       because cell value metadata has precedence anyway? But we would still first have to copy metadata from ALL_ELEMENTS
    #       to cell value metadata so that it is available there for querying.
    def _produce_column(self, inputs: Inputs, column_index: int, columns_primitives: typing.Optional[typing.List[typing.List[base.PrimitiveBase]]]) -> Outputs:
        output_column_values = []

        if columns_primitives is not None:
            if self.hyperparams['fit_primitive'] == 'fit':
                # We will set it later for every row.
                primitive = None
            else:
                assert self.hyperparams['fit_primitive'] == 'continue_fit'
                # When "fit_primitive" is "continue_fit", we have only
                # one primitive instance for the whole column.
                primitive = columns_primitives[column_index][0]
        else:
            # It could be that "fit_primitive" is "no" or that we have a transformer primitive.
            primitive = self.hyperparams['primitive']

        for row_index, column_value in enumerate(inputs.iloc[:, column_index]):
            column_value = self._prepare_cell_value(inputs.metadata, column_value, row_index, column_index)

            if columns_primitives is not None and self.hyperparams['fit_primitive'] == 'fit':
                primitive = columns_primitives[column_index][row_index]

            output_value = getattr(primitive, self.hyperparams['produce_method'])(inputs=column_value).value

            output_column_values.append(output_value)

        output_column = container.DataFrame({inputs.columns[column_index]: output_column_values}, generate_metadata=False)

        output_column.metadata = metadata_base.DataMetadata(inputs.metadata.query(()))
        output_column.metadata = output_column.metadata.update((metadata_base.ALL_ELEMENTS, 0), inputs.metadata.query((metadata_base.ALL_ELEMENTS, column_index)))
        output_column.metadata = output_column.metadata.generate(output_column)

        # TODO: Because metadata generation does not reuse existing metadata, we have to copy it ourselves.
        #       See: https://gitlab.com/datadrivendiscovery/d3m/issues/119
        for row_index, column_value in enumerate(output_column_values):
            output_column.metadata = column_value.metadata.copy_to(
                output_column.metadata,
                (),
                (row_index, 0),
            )

        # TODO: Should we compact metadata? It could make it nicer.
        #       But it could be slow, especially with nested DataFrames.

        return output_column

    def get_params(self) -> Params:
        if not self._fitted:
            return Params(
                columns_params=None,
            )

        return Params(
            columns_params=[
                [primitive.get_params() for primitive in column]
                for column in self._columns_primitives
            ],
        )

    def set_params(self, *, params: Params) -> None:
        if params['columns_primitives'] is None:
            self._columns_primitives = None
            self._fitted = False
            return

        columns_primitives = []
        for column in params['columns_primitives']:
            column_primitives = []

            for params in column:
                primitive = copy.deepcopy(self.hyperparams['primitive'])
                primitive.set_params(params)
                column_primitives.append(primitive)

            columns_primitives.append(column_primitives)

        self._columns_primitives = columns_primitives
        self._fitted = True
