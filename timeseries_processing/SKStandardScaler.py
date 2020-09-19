from typing import Any, Callable, List, Dict, Union, Optional, Sequence, Tuple
from numpy import ndarray
from collections import OrderedDict
from scipy import sparse
import os
import sklearn
import numpy
import typing

# Custom import commands if any
from sklearn.preprocessing import StandardScaler

from d3m.container.numpy import ndarray as d3m_ndarray
from d3m.container import DataFrame as d3m_dataframe
from d3m.metadata import hyperparams, params, base as metadata_base
from d3m import utils
from d3m.base import utils as base_utils
from d3m.exceptions import PrimitiveNotFittedError
from d3m.primitive_interfaces.base import CallResult, DockerContainer

# from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase

from d3m.primitive_interfaces.base import ProbabilisticCompositionalityMixin, ContinueFitMixin
from d3m import exceptions
import pandas

from d3m import container, utils as d3m_utils
import uuid

Inputs = d3m_dataframe
# Inputs = container.Dataset
Outputs = d3m_dataframe

__all__ = ('SKStandardScaler',)


class Params(params.Params):

    scale_: Optional[ndarray]
    mean_: Optional[ndarray]
    var_: Optional[ndarray]
    n_samples_seen_: Optional[numpy.int64]

    # Keep previous
    input_column_names: Optional[Any]
    target_names_: Optional[Sequence[Any]]
    training_indices_: Optional[Sequence[int]]
    target_column_indices_: Optional[Sequence[int]]
    target_columns_metadata_: Optional[List[OrderedDict]]



class Hyperparams(hyperparams.Hyperparams):
    # Added by Guanchu
    with_mean = hyperparams.UniformBool(
        default=True,
        description='If True, center the data before scaling. This does not work (and will raise an exception) when attempted on sparse matrices, because centering them entails building a dense matrix which in common use cases is likely to be too large to fit in memory.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    with_std = hyperparams.UniformBool(
        default=True,
        description='If True, scale the data to unit variance (or equivalently, unit standard deviation).',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    # copy = hyperparams.UniformBool(
    #     default=True,
    #     description='If False, try to avoid a copy and do inplace scaling instead. This is not guaranteed to always work inplace; e.g. if the data is not a NumPy array or scipy.sparse CSR matrix, a copy may still be returned.',
    #     semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    # )

    # Keep previous
    use_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to force primitive to operate on. If any specified column cannot be parsed, it is skipped.",
    )
    exclude_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to not operate on. Applicable only if \"use_columns\" is not provided.",
    )
    return_result = hyperparams.Enumeration(
        values=['append', 'replace', 'new'],
        default='new',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Should parsed columns be appended, should they replace original columns, or should only parsed columns be returned? This hyperparam is ignored if use_semantic_types is set to false.",
    )
    use_semantic_types = hyperparams.UniformBool(
        default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Controls whether semantic_types metadata will be used for filtering columns in input dataframe. Setting this to false makes the code ignore return_result and will produce only the output dataframe"
    )
    add_index_columns = hyperparams.UniformBool(
        default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Also include primary index columns if input data has them. Applicable only if \"return_result\" is set to \"new\".",
    )
    error_on_no_input = hyperparams.UniformBool(
        default=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Throw an exception if no input column is selected/provided. Defaults to true to behave like sklearn. To prevent pipelines from breaking set this to False.",
    )

    return_semantic_type = hyperparams.Enumeration[str](
        values=['https://metadata.datadrivendiscovery.org/types/Attribute',
                'https://metadata.datadrivendiscovery.org/types/ConstructedAttribute'],
        default='https://metadata.datadrivendiscovery.org/types/Attribute',
        description='Decides what semantic type to attach to generated attributes',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )


class SKStandardScaler(UnsupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    Standardize features by removing the mean and scaling to unit variance.
    See `sklearn documentation <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html?highlight=standardscaler#sklearn.preprocessing.StandardScaler>`_ for more details.
    
    Parameters
    ----------
    with_mean : bool
        If True, center the data before scaling. This does not work (and will raise an exception) when attempted on sparse matrices, because centering them entails building a dense matrix which in common use cases is likely to be too large to fit in memory.

    with_std : bool
        If True, scale the data to unit variance (or equivalently, unit standard deviation).

    Attributes
    ----------
    scale_: ndarray or None, shape (n_features,)
        Per feature relative scaling of the data. This is calculated using np.sqrt(var_). Equal to None when with_std=False.

    mean_: ndarray or None, shape (n_features,)
        The mean value for each feature in the training set. Equal to None when with_mean=False.

    var_: ndarray or None, shape (n_features,)
        The variance for each feature in the training set. Used to compute scale_. Equal to None when with_std=False.

    n_samples_seen_: int or array, shape (n_features,)
        The number of samples processed by the estimator for each feature. If there are not missing samples, the n_samples_seen will be an integer, otherwise it will be an array. Will be reset on new calls to fit, but increments across partial_fit calls.
    """
    
    __author__ = "DATALAB @Taxes A&M University"
    metadata = metadata_base.PrimitiveMetadata({
        "algorithm_types": [metadata_base.PrimitiveAlgorithmType.DATA_CONVERSION, ],
        "name": "Standard_scaler",
        "primitive_family": metadata_base.PrimitiveFamily.DATA_TRANSFORMATION,
        "python_path": "d3m.primitives.tods.timeseries_processing.transformation.standard_scaler",
        "hyperparams_to_tune": ['with_mean', 'with_std'],
        "source": {'name': "DATALAB @Taxes A&M University", 'contact': 'mailto:khlai037@tamu.edu',
                    'uris': ['https://gitlab.com/lhenry15/tods.git']},
        "version": "0.0.1",
        "id": str(uuid.uuid3(uuid.NAMESPACE_DNS, 'SKStandardScaler')),
    })

    def __init__(self, *,
                 hyperparams: Hyperparams,
                 random_seed: int = 0,
                 docker_containers: Dict[str, DockerContainer] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

        # False
        self._clf = StandardScaler(with_mean=self.hyperparams['with_mean'],
                                   with_std=self.hyperparams['with_std'],
                                   # copy=self.hyperparams['copy'],
                                  )

        self._inputs = None
        self._outputs = None
        self._training_inputs = None
        self._training_outputs = None
        self._target_names = None
        self._training_indices = None
        self._target_column_indices = None
        self._target_columns_metadata: List[OrderedDict] = None
        self._input_column_names = None
        self._fitted = False

        # print(self._clf.get_params(deep=True))
        # print(getattr(self._clf, 'lambdas_'))
        # print(dir(self._clf))


    def set_training_data(self, *, inputs: Inputs) -> None:

        """
        Set training data for Standardizer.
        Args:
            inputs: Container DataFrame

        Returns:
            None
        """

        self._inputs = inputs
        self._fitted = False


    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:

        """
        Fit model with training data.
        Args:
            *: Container DataFrame. Time series data up to fit.

        Returns:
            None
        """

        if self._fitted:
            return CallResult(None)

        self._training_inputs, self._training_indices = self._get_columns_to_fit(self._inputs, self.hyperparams)
        self._input_column_names = self._training_inputs.columns

        if self._training_inputs is None:
            return CallResult(None)

        if len(self._training_indices) > 0:
            self._clf.fit_transform(self._training_inputs)
            self._fitted = True
        else:
            if self.hyperparams['error_on_no_input']:
                raise RuntimeError("No input columns were selected")
            self.logger.warn("No input columns were selected")

        # print(self._training_inputs.std())

        return CallResult(None)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:

        """
        Process the testing data.
        Args:
            inputs: Container DataFrame. Time series data up to standardlize.

        Returns:
            Container DataFrame after standardlization.
        """

        if not self._fitted:
            raise PrimitiveNotFittedError("Primitive not fitted.")
        sk_inputs = inputs
        if self.hyperparams['use_semantic_types']:
            sk_inputs = inputs.iloc[:, self._training_indices]
        output_columns = []
        if len(self._training_indices) > 0:
            sk_output = self._clf.transform(sk_inputs)
            if sparse.issparse(sk_output):
                sk_output = sk_output.toarray()
            outputs = self._wrap_predictions(inputs, sk_output)
            if len(outputs.columns) == len(self._input_column_names):
                outputs.columns = self._input_column_names
            output_columns = [outputs]
        else:
            if self.hyperparams['error_on_no_input']:
                raise RuntimeError("No input columns were selected")
            self.logger.warn("No input columns were selected")

        # print(outputs.metadata.to_internal_simple_structure())

        outputs = base_utils.combine_columns(return_result=self.hyperparams['return_result'],
                                             add_index_columns=self.hyperparams['add_index_columns'],
                                             inputs=inputs, column_indices=self._training_indices,
                                             columns_list=output_columns)
        # print(inputs)
        # print(outputs)
        # print(inputs.metadata.to_internal_simple_structure())
        # print(outputs.metadata.to_internal_simple_structure())

        return CallResult(outputs)


    def get_params(self) -> Params:

        """
        Return parameters.
        Args:
            None

        Returns:
            class Params
        """

        if not self._fitted:
            return Params(
                scale_=None,
                mean_=None,
                var_=None,
                n_sample_seen_=None,

                # Keep previous
                input_column_names=self._input_column_names,
                training_indices_=self._training_indices,
                target_names_=self._target_names,
                target_column_indices_=self._target_column_indices,
                target_columns_metadata_=self._target_columns_metadata
            )
        # print(self._clf.n_samples_seen_.shape)
        # print(type(self._clf.n_samples_seen_))
        # print(type(self._clf.mean_))
        return Params(
            scale_=getattr(self._clf, 'scale_', None),
            mean_=getattr(self._clf, 'mean_', None),
            var_=getattr(self._clf, 'var_', None),
            n_samples_seen_=getattr(self._clf, 'n_samples_seen_', None),
            # Keep previous
            input_column_names=self._input_column_names,
            training_indices_=self._training_indices,
            target_names_=self._target_names,
            target_column_indices_=self._target_column_indices,
            target_columns_metadata_=self._target_columns_metadata,
        )


    def set_params(self, *, params: Params) -> None:

        """
        Set parameters for Standardizer.
        Args:
            params: class Params

        Returns:
            None
        """

        self._clf.scale_ = params['scale_']
        self._clf.mean_ = params['mean_']
        self._clf.var_ = params['var_']
        self._clf.n_samples_seen_ = params['n_samples_seen_']
        # Keep previous
        self._input_column_names = params['input_column_names']
        self._training_indices = params['training_indices_']
        self._target_names = params['target_names_']
        self._target_column_indices = params['target_column_indices_']
        self._target_columns_metadata = params['target_columns_metadata_']

        if params['scale_'] is not None:
            self._fitted = True
        if params['mean_'] is not None:
            self._fitted = True
        if params['var_'] is not None:
            self._fitted = True
        if params['n_samples_seen_'] is not None:
            self._fitted = True


    @classmethod
    def _get_columns_to_fit(cls, inputs: Inputs, hyperparams: Hyperparams):

        """
        Select columns to fit.
        Args:
            inputs: Container DataFrame
            hyperparams: d3m.metadata.hyperparams.Hyperparams

        Returns:
            list
        """


        if not hyperparams['use_semantic_types']:
            return inputs, list(range(len(inputs.columns)))

        inputs_metadata = inputs.metadata

        def can_produce_column(column_index: int) -> bool:
            return cls._can_produce_column(inputs_metadata, column_index, hyperparams)

        columns_to_produce, columns_not_to_produce = base_utils.get_columns_to_use(inputs_metadata,
                                                                                   use_columns=hyperparams['use_columns'],
                                                                                   exclude_columns=hyperparams[
                                                                                       'exclude_columns'],
                                                                                   can_use_column=can_produce_column)
        return inputs.iloc[:, columns_to_produce], columns_to_produce
        # return columns_to_produce


    @classmethod
    def _can_produce_column(cls, inputs_metadata: metadata_base.DataMetadata, column_index: int,
                            hyperparams: Hyperparams) -> bool:
        """
        Output whether a column can be processed.
        Args:
            inputs_metadata: d3m.metadata.base.DataMetadata
            column_index: int

        Returns:
            bool
        """

        column_metadata = inputs_metadata.query((metadata_base.ALL_ELEMENTS, column_index))

        accepted_structural_types = (int, float, numpy.integer, numpy.float64)
        accepted_semantic_types = set()
        accepted_semantic_types.add("https://metadata.datadrivendiscovery.org/types/Attribute")
        if not issubclass(column_metadata['structural_type'], accepted_structural_types):
            return False

        semantic_types = set(column_metadata.get('semantic_types', []))

        # print(semantic_types)

        if len(semantic_types) == 0:
            cls.logger.warning("No semantic types found in column metadata")
            return False

        # Making sure all accepted_semantic_types are available in semantic_types
        if len(accepted_semantic_types - semantic_types) == 0:
            return True

        return False


    @classmethod
    def _get_target_columns_metadata(cls, outputs_metadata: metadata_base.DataMetadata, hyperparams) -> List[OrderedDict]:

        """
        Output metadata of selected columns.
        Args:
            outputs_metadata: metadata_base.DataMetadata
            hyperparams: d3m.metadata.hyperparams.Hyperparams

        Returns:
            d3m.metadata.base.DataMetadata
        """

        outputs_length = outputs_metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length']

        target_columns_metadata: List[OrderedDict] = []
        for column_index in range(outputs_length):
            column_metadata = OrderedDict(outputs_metadata.query_column(column_index))

            # Update semantic types and prepare it for predicted targets.
            semantic_types = set(column_metadata.get('semantic_types', []))
            semantic_types_to_remove = set([])
            add_semantic_types = []
            add_semantic_types.add(hyperparams["return_semantic_type"])
            semantic_types = semantic_types - semantic_types_to_remove
            semantic_types = semantic_types.union(add_semantic_types)
            column_metadata['semantic_types'] = list(semantic_types)

            target_columns_metadata.append(column_metadata)

        return target_columns_metadata


    @classmethod
    def _update_predictions_metadata(cls, inputs_metadata: metadata_base.DataMetadata, outputs: Optional[Outputs],
                                     target_columns_metadata: List[OrderedDict]) -> metadata_base.DataMetadata:

        """
        Updata metadata for selected columns.
        Args:
            inputs_metadata: metadata_base.DataMetadata
            outputs: Container Dataframe
            target_columns_metadata: list

        Returns:
            d3m.metadata.base.DataMetadata
        """

        outputs_metadata = metadata_base.DataMetadata().generate(value=outputs)

        for column_index, column_metadata in enumerate(target_columns_metadata):
            column_metadata.pop("structural_type", None)
            outputs_metadata = outputs_metadata.update_column(column_index, column_metadata)

        return outputs_metadata


    def _wrap_predictions(self, inputs: Inputs, predictions: ndarray) -> Outputs:

        """
        Wrap predictions into dataframe
        Args:
            inputs: Container Dataframe
            predictions: array-like data (n_samples, n_features)

        Returns:
            Dataframe
        """

        outputs = d3m_dataframe(predictions, generate_metadata=True)
        target_columns_metadata = self._copy_inputs_metadata(inputs.metadata, self._training_indices, outputs.metadata,
                                                             self.hyperparams)
        outputs.metadata = self._update_predictions_metadata(inputs.metadata, outputs, target_columns_metadata)
        # print(outputs.metadata.to_internal_simple_structure())

        return outputs


    @classmethod
    def _copy_inputs_metadata(cls, inputs_metadata: metadata_base.DataMetadata, input_indices: List[int],
                              outputs_metadata: metadata_base.DataMetadata, hyperparams):

        """
        Updata metadata for selected columns.
        Args:
            inputs_metadata: metadata.base.DataMetadata
            input_indices: list
            outputs_metadata: metadata.base.DataMetadata
            hyperparams: d3m.metadata.hyperparams.Hyperparams

        Returns:
            d3m.metadata.base.DataMetadata
        """

        outputs_length = outputs_metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length']
        target_columns_metadata: List[OrderedDict] = []
        for column_index in input_indices:
            column_name = inputs_metadata.query((metadata_base.ALL_ELEMENTS, column_index)).get("name")
            if column_name is None:
                column_name = "output_{}".format(column_index)

            column_metadata = OrderedDict(inputs_metadata.query_column(column_index))
            semantic_types = set(column_metadata.get('semantic_types', []))
            semantic_types_to_remove = set([])
            add_semantic_types = set()
            add_semantic_types.add(hyperparams["return_semantic_type"])
            semantic_types = semantic_types - semantic_types_to_remove
            semantic_types = semantic_types.union(add_semantic_types)
            column_metadata['semantic_types'] = list(semantic_types)

            column_metadata["name"] = str(column_name)
            target_columns_metadata.append(column_metadata)

        #  If outputs has more columns than index, add Attribute Type to all remaining
        if outputs_length > len(input_indices):
            for column_index in range(len(input_indices), outputs_length):
                column_metadata = OrderedDict()
                semantic_types = set()
                semantic_types.add(hyperparams["return_semantic_type"])
                column_name = "output_{}".format(column_index)
                column_metadata["semantic_types"] = list(semantic_types)
                column_metadata["name"] = str(column_name)
                target_columns_metadata.append(column_metadata)

        return target_columns_metadata


SKStandardScaler.__doc__ = SKStandardScaler.__doc__
