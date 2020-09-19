import os.path
import pickle
import typing

from d3m import container, exceptions, utils as d3m_utils
from d3m.base import utils as base_utils
from d3m.container import dataset
from d3m.metadata import base as metadata_base, hyperparams as hyperparams_module, params as params_module
from d3m.primitive_interfaces import base, unsupervised_learning

import common_primitives

from .slacker import feature_extraction

__all__ = ('AnnotatedTabularExtractorPrimitive',)

Inputs = container.DataFrame
Outputs = container.DataFrame


class Params(params_module.Params):
    column_types: typing.Optional[typing.Dict[str, str]]
    numeric_columns: typing.Optional[typing.List[str]]
    categorical_columns: typing.Optional[typing.List[str]]
    text_columns: typing.Optional[typing.List[str]]
    numeric_imputer: typing.Optional[bytes]
    numeric_scaler: typing.Optional[bytes]
    categorical_encoder: typing.Optional[bytes]
    categorical_imputer: typing.Optional[bytes]
    one_hot_encoder: typing.Optional[bytes]


class Hyperparams(hyperparams_module.Hyperparams):
    normalize_text = hyperparams_module.UniformBool(
        default=False,
        description="Convert text to lowercase and strip whitespace.",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    )
    numeric_strategy = hyperparams_module.Enumeration[str](
        values=['mean', 'median'],
        default='mean',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    )
    add_missing_indicator = hyperparams_module.UniformBool(
        default=True,
        description="Add columns to indicate missing values.",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )


class AnnotatedTabularExtractorPrimitive(unsupervised_learning.UnsupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):  # pylint: disable=inherit-non-class
    """
    A primitive wrapping for MIT-LL slacker's ``AnnotatedTabularExtractor``.
    """

    __author__ = 'Tianrui, Jian and Julia'
    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '6c425897-6ffe-45b8-949f-002d872ccf12',
            'version': '0.1.0',
            'name': 'Annotated tabular extractor',
            'python_path': 'd3m.primitives.data_cleaning.tabular_extractor.Common',
            'source': {
                'name': common_primitives.__author__,
                'contact': 'mailto:y.cao@berkeley.edu',
                'uris': [
                    'https://gitlab.com/datadrivendiscovery/common-primitives/blob/master/common_primitives/tabular_extractor.py',
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
                metadata_base.PrimitiveAlgorithmType.IMPUTATION,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.DATA_CLEANING,
        },
    )

    def __init__(self, *, hyperparams: Hyperparams) -> None:
        super().__init__(hyperparams=hyperparams)

        self._training_inputs: Inputs = None
        self._extractor: feature_extraction.AnnotatedTabularExtractor = None

    def set_training_data(self, *, inputs: Inputs) -> None:  # type: ignore
        self._training_inputs = inputs
        self._extractor = None

    def fit(self, *, timeout: float = None, iterations: int = None) -> base.CallResult[None]:
        # If already fitted with current training data, this call is a no-op.
        if self._extractor:
            return base.CallResult(None)

        if self._training_inputs is None:
            raise exceptions.InvalidStateError("Missing training data.")

        extractor = self._create_extractor()

        attribute_columns = self._get_attribute_columns(self._training_inputs.metadata)

        extractor.set_cols_info(self._metadata_to_cols_info(self._training_inputs.metadata, attribute_columns))
        extractor.fit_transform(self._training_inputs.iloc[:, attribute_columns], [])

        # Fitted.
        self._extractor = extractor

        return base.CallResult(None)

    def _create_extractor(self) -> feature_extraction.AnnotatedTabularExtractor:
        return feature_extraction.AnnotatedTabularExtractor(
            normalize_text=self.hyperparams['normalize_text'],
            numeric_strategy=self.hyperparams['numeric_strategy'],
            add_missing_indicator=self.hyperparams['add_missing_indicator'],
        )

    def _get_attribute_columns(self, inputs_metadata: metadata_base.DataMetadata) -> typing.Sequence[int]:
        return inputs_metadata.list_columns_with_semantic_types(['https://metadata.datadrivendiscovery.org/types/Attribute'])

    def _metadata_to_cols_info(self, inputs_metadata: metadata_base.DataMetadata, attribute_columns: typing.Sequence[int]) -> typing.Sequence[typing.Dict]:
        cols_info = []
        for i, column_index in enumerate(attribute_columns):
            column_metadata = inputs_metadata.query_column(column_index)

            column_type = None
            column_roles = []
            for semantic_type in column_metadata['semantic_types']:
                if semantic_type in dataset.SEMANTIC_TYPES_TO_D3M_COLUMN_TYPES:
                    if column_type is not None:
                        raise exceptions.InvalidStateError(
                            "Duplicate semantic types for column types: '{first_type}' and '{second_type}'".format(
                                first_type=column_type,
                                second_type=dataset.SEMANTIC_TYPES_TO_D3M_COLUMN_TYPES[semantic_type],
                            ),
                        )
                    column_type = dataset.SEMANTIC_TYPES_TO_D3M_COLUMN_TYPES[semantic_type]
                elif semantic_type in dataset.SEMANTIC_TYPES_TO_D3M_ROLES:
                    column_roles.append(dataset.SEMANTIC_TYPES_TO_D3M_ROLES[semantic_type])

            if column_type is None:
                raise exceptions.InvalidStateError("Could not find a column type among semantic types.")

            cols_info.append(
                {
                    'colIndex': i,
                    'colName': column_metadata['name'],
                    'colType': column_type,
                    'role': column_roles,
                },
            )

        return cols_info

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        if not self._extractor:
            raise exceptions.PrimitiveNotFittedError("Primitive not fitted.")

        attribute_columns = self._get_attribute_columns(inputs.metadata)

        # This is a sparse scipy CSR matrix.
        transformed_inputs = self._extractor.transform(inputs.iloc[:, attribute_columns])

        output_columns = container.DataFrame(transformed_inputs.toarray(), generate_metadata=True)

        # All transformed inputs are attributes.
        output_columns.metadata = output_columns.metadata.add_semantic_type(
            (metadata_base.ALL_ELEMENTS, metadata_base.ALL_ELEMENTS),
            'https://metadata.datadrivendiscovery.org/types/Attribute',
        )

        # This replaces attribute columns with output columns, while keeping other columns (like "d3mIndex" and target columns).
        outputs = base_utils.combine_columns(inputs, list(attribute_columns), [output_columns], return_result='replace', add_index_columns=True)

        return base.CallResult(outputs)

    def get_params(self) -> Params:
        if not self._extractor:
            return Params(
                column_types=None,
                numeric_columns=None,
                categorical_columns=None,
                text_columns=None,
                numeric_imputer=None,
                numeric_scaler=None,
                categorical_encoder=None,
                categorical_imputer=None,
                one_hot_encoder=None,
            )

        return Params(
            # In Python 3.6 all dicts preserve order, so we can do this.
            # We have to do it as a workaround for a pytypes bug.
            # See: https://github.com/Stewori/pytypes/issues/52
            column_types=dict(self._extractor.column_types),
            numeric_columns=self._extractor.numeric_columns,
            categorical_columns=self._extractor.categorical_columns,
            text_columns=self._extractor.text_columns,
            # Generally, one should not just pickle child instances, but extract underlying params.
            numeric_imputer=pickle.dumps(self._extractor.numeric_imputer) if hasattr(self._extractor, 'numeric_imputer') else None,
            numeric_scaler=pickle.dumps(self._extractor.numeric_scaler) if hasattr(self._extractor, 'numeric_scaler') else None,
            categorical_encoder=pickle.dumps(self._extractor.categorical_encoder) if hasattr(self._extractor, 'categorical_encoder') else None,
            categorical_imputer=pickle.dumps(self._extractor.categorical_imputer) if hasattr(self._extractor, 'categorical_imputer') else None,
            one_hot_encoder=pickle.dumps(self._extractor.one_hot_encoder) if hasattr(self._extractor, 'one_hot_encoder') else None,
        )

    def set_params(self, *, params: Params) -> None:
        if not all(params[param] is not None for param in ['column_types', 'numeric_columns', 'categorical_columns', 'text_columns']):
            self._extractor = None
        else:
            extractor = self._create_extractor()

            extractor.column_types = params['column_types']
            extractor.numeric_columns = params['numeric_columns']
            extractor.categorical_columns = params['categorical_columns']
            extractor.text_columns = params['text_columns']

            if params['numeric_imputer'] is not None:
                extractor.numeric_imputer = pickle.loads(params['numeric_imputer'])
            if params['numeric_scaler'] is not None:
                extractor.numeric_scaler = pickle.loads(params['numeric_scaler'])
            if params['categorical_encoder'] is not None:
                extractor.categorical_encoder = pickle.loads(params['categorical_encoder'])
            if params['categorical_imputer'] is not None:
                extractor.categorical_imputer = pickle.loads(params['categorical_imputer'])
            if params['one_hot_encoder'] is not None:
                extractor.one_hot_encoder = pickle.loads(params['one_hot_encoder'])

            self._extractor = extractor
