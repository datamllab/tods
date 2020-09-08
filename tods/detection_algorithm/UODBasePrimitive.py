from typing import Any, Callable, List, Dict, Union, Optional, Sequence, Tuple
from numpy import ndarray
from collections import OrderedDict
from scipy import sparse
import os
import sklearn
import numpy
# import typing
import abc
import typing
#
# # Custom import commands if any
import warnings
import numpy as np
from sklearn.utils import check_array
from sklearn.exceptions import NotFittedError
# from numba import njit
# from pyod.utils.utility import argmaxn
from pyod.models.base import BaseDetector

import copy

from d3m.container.numpy import ndarray as d3m_ndarray
from d3m.container import DataFrame as d3m_dataframe
from d3m.metadata import hyperparams, params, base as metadata_base
from d3m import utils
from d3m.base import utils as base_utils
from d3m.exceptions import PrimitiveNotFittedError
from d3m.primitive_interfaces.base import CallResult, DockerContainer, PrimitiveBase, MultiCallResult, Params, Hyperparams

# # from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase

from d3m.primitive_interfaces.base import *

from d3m.primitive_interfaces.base import ProbabilisticCompositionalityMixin, ContinueFitMixin
from d3m import exceptions
import pandas

from d3m import container, utils as d3m_utils

Inputs = d3m_dataframe
# Inputs = container.Dataset
Outputs = d3m_dataframe

# import abc
# import typing

from d3m.primitive_interfaces.base import *

__all__ = ('UnsupervisedOutlierDetectorBase',)


class Params_ODBase(params.Params):

    # decision_scores_: Optional[ndarray]
    # threshold_: Optional[float]
    # labels_: Optional[ndarray]
    left_inds_: Optional[ndarray]
    right_inds_: Optional[ndarray]
    # clf_: Optional[BaseDetector]
    clf_: Optional[Any]

    # Keep previous
    input_column_names: Optional[Any]
    target_names_: Optional[Sequence[Any]]
    training_indices_: Optional[Sequence[int]]
    target_column_indices_: Optional[Sequence[int]]
    target_columns_metadata_: Optional[List[OrderedDict]]



class Hyperparams_ODBase(hyperparams.Hyperparams):

    contamination = hyperparams.Uniform( # Hyperparameter[float](
        lower=0.,
        upper=0.5,
        default=0.1,
        description='If False, try to avoid a copy and do inplace scaling instead. This is not guaranteed to always work inplace; e.g. if the data is not a NumPy array or scipy.sparse CSR matrix, a copy may still be returned.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    window_size = hyperparams.Hyperparameter[int](
        default=1,
        description='The moving window size.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    step_size = hyperparams.Hyperparameter[int](
        default=1,
        description='The displacement for moving window.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )

    # Keep previous
    return_subseq_inds = hyperparams.UniformBool(
        default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="If true, return value includes subsequence index."
    )
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

class UnsupervisedOutlierDetectorBase(UnsupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    Parameters
    ----------
    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set, i.e.
        the proportion of outliers in the data set. When fitting this is used
        to define the threshold on the decision function.

    Attributes
    ----------
    clf_.decision_scores_ : numpy array of shape (n_samples,)
        The outlier scores of the training data.
        The higher, the more abnormal. Outliers tend to have higher
        scores. This value is available once the detector is
        fitted.

    clf_.threshold_: float within (0, 1)
        For outlier, decision_scores_ more than threshold_.
        For inlier, decision_scores_ less than threshold_.

    clf_.labels_ : int, either 0 or 1
        The binary labels of the training data. 0 stands for inliers.
        and 1 for outliers/anomalies. It is generated by applying.
        ``threshold_`` on ``decision_scores_``.

    left_inds_ : ndarray,
        One of the mapping from decision_score to data.
        For point outlier detection, left_inds_ exactly equals the index of each data point.
        For Collective outlier detection, left_inds_ equals the start index of each subsequence.

    left_inds_ : ndarray,
        One of the mapping from decision_score to data.
        For point outlier detection, left_inds_ exactly equals the index of each data point plus 1.
        For Collective outlier detection, left_inds_ equals the ending index of each subsequence.
    """
    # probability_score:
    # window_size: int
    # The moving window size.

    __author__ = "DATALAB @Taxes A&M University"
    metadata: metadata_base.PrimitiveMetadata = None

    def __init__(self, *,
                 hyperparams: Hyperparams,
                 random_seed: int = 0,
                 docker_containers: Dict[str, DockerContainer] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

        self._clf = None
        self._clf_fit_parameter = {}
        self.primitiveNo = 0

        self.window_size = hyperparams['window_size']
        self.step_size = hyperparams['step_size']
        self.left_inds_ = None
        self.right_inds_ = None

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
#
    @abc.abstractmethod
    def set_training_data(self, *, inputs: Inputs) -> None:
        """
        Set training data for outlier detection.
        Args:
            inputs: Container DataFrame

        Returns:
            None
        """
        self._inputs = inputs
        self._fitted = False

    def _set_subseq_inds(self):

        self.left_inds_ = getattr(self._clf, 'left_inds_', None)
        self.right_inds_ = getattr(self._clf, 'right_inds_', None)

        if self.left_inds_ is None or self.right_inds_ is None:
            self.left_inds_ = numpy.arange(0, len(self._inputs), self.step_size)
            self.right_inds_ = self.left_inds_ + self.window_size
            self.right_inds_[self.right_inds_ > len(self._inputs)] = len(self._inputs)
            # print(self.left_inds_, self.right_inds_)

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        """
        Fit model with training data.
        Args:
            *: Container DataFrame. Time series data up to fit.

        Returns:
            None
        """
        # print('Fit:', self._clf)

        if self._fitted:
            return CallResult(None)

        self._training_inputs, self._training_indices = self._get_columns_to_fit(self._inputs, self.hyperparams)
        self._input_column_names = self._training_inputs.columns

        if self._training_inputs is None:
            return CallResult(None)

        if len(self._training_indices) > 0:

            # print('Fit: ', self._clf)
            # print('Fit: ', self._training_inputs.values.shape)
            # print('Fit: ', self._clf.fit(self._training_inputs.values))

            self._clf.fit(X=self._training_inputs.values, **self._clf_fit_parameter)
            self._fitted = True
            self._set_subseq_inds()

        else:
            if self.hyperparams['error_on_no_input']:
                raise RuntimeError("No input columns were selected")
            self.logger.warn("No input columns were selected")

        return CallResult(None)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Process the testing data.
        Args:
            inputs: Container DataFrame. Time series data up to outlier detection.

        Returns:
            Container DataFrame
            1 marks Outliers, 0 marks normal.
        """

        if not self._fitted:
            raise PrimitiveNotFittedError("Primitive not fitted.")
        sk_inputs = inputs
        if self.hyperparams['use_semantic_types']:
            sk_inputs = inputs.iloc[:, self._training_indices]
        output_columns = []
        if len(self._training_indices) > 0:

            if self.hyperparams['return_subseq_inds']:

                if getattr(self._clf, 'left_inds_', None) is None or getattr(self._clf, 'right_inds_', None) is None: # point OD
                    pred_label = self._clf.predict(sk_inputs.values)
                    left_inds_ = numpy.arange(0, len(pred_label), self.step_size)
                    right_inds_ = left_inds_ + self.window_size
                    right_inds_[right_inds_ > len(pred_label)] = len(pred_label)
                else:
                    pred_label, left_inds_, right_inds_ = self._clf.predict(sk_inputs.values)

                # print(pred_label.shape, left_inds_.shape, right_inds_.shape)
                # print(pred_label, left_inds_, right_inds_)

                sk_output = numpy.concatenate((numpy.expand_dims(pred_label, axis=1),
                                               numpy.expand_dims(left_inds_, axis=1),
                                               numpy.expand_dims(right_inds_, axis=1)), axis=1)


            else:
                if getattr(self._clf, 'left_inds_', None) is None or getattr(self._clf, 'right_inds_', None) is None: # point OD
                    sk_output = self._clf.predict(sk_inputs.values)

                else:
                    sk_output, _, _ = self._clf.predict(sk_inputs.values)

            # print(sk_output)
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

        outputs = base_utils.combine_columns(return_result=self.hyperparams['return_result'],
                                             add_index_columns=self.hyperparams['add_index_columns'],
                                             inputs=inputs, column_indices=self._training_indices,
                                             columns_list=output_columns)

        return CallResult(outputs)

    def produce_score(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Process the testing data.
        Args:
            inputs: Container DataFrame. Time series data up to outlier detection.

        Returns:
            Container DataFrame
            1 marks Outliers, 0 marks normal.
        """

        if not self._fitted:
            raise PrimitiveNotFittedError("Primitive not fitted.")
        sk_inputs = inputs
        if self.hyperparams['use_semantic_types']:
            sk_inputs = inputs.iloc[:, self._training_indices]
        output_columns = []
        if len(self._training_indices) > 0:

            if self.hyperparams['return_subseq_inds']:

                if getattr(self._clf, 'left_inds_', None) is None or getattr(self._clf, 'right_inds_', None) is None: # point OD
                    pred_score = self._clf.decision_function(sk_inputs.values).ravel()
                    left_inds_ = numpy.arange(0, len(pred_score), self.step_size)
                    right_inds_ = left_inds_ + self.window_size
                    right_inds_[right_inds_ > len(pred_score)] = len(pred_score)

                else:
                    pred_score, left_inds_, right_inds_ = self._clf.decision_function(sk_inputs.values)

                # print(pred_score.shape, left_inds_.shape, right_inds_.shape)

                sk_output = numpy.concatenate((numpy.expand_dims(pred_score, axis=1),
                                               numpy.expand_dims(left_inds_, axis=1),
                                               numpy.expand_dims(right_inds_, axis=1)), axis=1)

            else:
                if getattr(self._clf, 'left_inds_', None) is None or getattr(self._clf, 'right_inds_', None) is None: # point OD
                    sk_output = self._clf.decision_function(sk_inputs.values)

                else:
                    sk_output, _, _ = self._clf.decision_function(sk_inputs.values)

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

        outputs = base_utils.combine_columns(return_result=self.hyperparams['return_result'],
                                             add_index_columns=self.hyperparams['add_index_columns'],
                                             inputs=inputs, column_indices=self._training_indices,
                                             columns_list=output_columns)
        return CallResult(outputs)


    def get_params(self) -> Params_ODBase:
        """
        Return parameters.
        Args:
            None

        Returns:
            class Params_ODBase
        """

        if not self._fitted:
            return Params_ODBase(
                # decision_scores_=None,
                # threshold_=None,
                # labels_=None,
                left_inds_=None,
                right_inds_=None,
                clf_=copy.copy(self._clf),

                # Keep previous
                input_column_names=self._input_column_names,
                training_indices_=self._training_indices,
                target_names_=self._target_names,
                target_column_indices_=self._target_column_indices,
                target_columns_metadata_=self._target_columns_metadata
            )

        return Params_ODBase(
            # decision_scores_=getattr(self._clf, 'decision_scores_', None),
            # threshold_=getattr(self._clf, 'threshold_', None),
            # labels_=getattr(self._clf, 'labels_', None),
            left_inds_=self.left_inds_, # numpy.array(self.left_inds_)
            right_inds_=self.right_inds_, # numpy.array(self.right_inds_)
            clf_=copy.copy(self._clf),

            # Keep previous
            input_column_names=self._input_column_names,
            training_indices_=self._training_indices,
            target_names_=self._target_names,
            target_column_indices_=self._target_column_indices,
            target_columns_metadata_=self._target_columns_metadata
        )
        # pass


    def set_params(self, *, params: Params_ODBase) -> None:
        """
        Set parameters for outlier detection.
        Args:
            params: class Params_ODBase

        Returns:
            None
        """

        # self._clf.decision_scores_ = params['decision_scores_']
        # self._clf.threshold_ = params['threshold_']
        # self._clf.labels_ = params['labels_']
        self.left_inds_ = params['left_inds_']
        self.right_inds_ = params['right_inds_']
        self._clf = copy.copy(params['clf_'])

        # Keep previous
        self._input_column_names = params['input_column_names']
        self._training_indices = params['training_indices_']
        self._target_names = params['target_names_']
        self._target_column_indices = params['target_column_indices_']
        self._target_columns_metadata = params['target_columns_metadata_']


        # if params['decision_scores_'] is not None:
        #     self._fitted = True
        # if params['threshold_'] is not None:
        #     self._fitted = True
        # if params['labels_'] is not None:
        #     self._fitted = True
        if params['left_inds_'] is not None:
            self._fitted = True
        if params['right_inds_'] is not None:
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
                                                                                   exclude_columns=hyperparams['exclude_columns'],
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
        # target_columns_metadata = self._copy_inputs_metadata(inputs.metadata, self._training_indices, outputs.metadata,
        #                                                      self.hyperparams)
        target_columns_metadata = self._add_target_columns_metadata(outputs.metadata, self.hyperparams, self.primitiveNo)
        outputs.metadata = self._update_predictions_metadata(inputs.metadata, outputs, target_columns_metadata)
        # print(outputs.metadata.to_internal_simple_structure())

        return outputs

    @classmethod
    def _add_target_columns_metadata(cls, outputs_metadata: metadata_base.DataMetadata, hyperparams, primitiveNo):
        """
        Add target columns metadata
        Args:
            outputs_metadata: metadata.base.DataMetadata
            hyperparams: d3m.metadata.hyperparams.Hyperparams

        Returns:
            List[OrderedDict]
        """
        outputs_length = outputs_metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length']
        target_columns_metadata: List[OrderedDict] = []
        for column_index in range(outputs_length):
            column_name = "{0}{1}_{2}".format(cls.metadata.query()['name'], primitiveNo, column_index)
            column_metadata = OrderedDict()
            semantic_types = set()
            semantic_types.add(hyperparams["return_semantic_type"])
            column_metadata['semantic_types'] = list(semantic_types)

            column_metadata["name"] = str(column_name)
            target_columns_metadata.append(column_metadata)

        return target_columns_metadata

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


# OutlierDetectorBase.__doc__ = OutlierDetectorBase.__doc__
