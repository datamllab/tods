from typing import Any, Callable, List, Dict, Union, Optional, Sequence, Tuple
from numpy import ndarray
from collections import OrderedDict
from scipy import sparse
from sklearn.utils import check_array
import numpy as np
import typing
import time
import pandas as pd
import uuid

from d3m import container
from d3m.primitive_interfaces import base, transformer
from d3m.metadata import base as metadata_base, hyperparams

from d3m.container.numpy import ndarray as d3m_ndarray
from d3m.container import DataFrame as d3m_dataframe
from d3m.metadata import hyperparams, params, base as metadata_base
from d3m import utils
from d3m.base import utils as base_utils
from d3m.exceptions import PrimitiveNotFittedError
from d3m.primitive_interfaces.base import CallResult, DockerContainer


__all__ = ('SubsequenceSegmentationPrimitive',)

Inputs = container.DataFrame
Outputs = container.DataFrame
from tods.utils import construct_primitive_metadata

class Hyperparams(hyperparams.Hyperparams):
    # Tuning
    window_size = hyperparams.Hyperparameter[int](
        default=10,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="The moving window size.",
    )
    step = hyperparams.Hyperparameter[int](
        default=1,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="The displacement for moving window.",
    )
    # return_numpy = hyperparams.UniformBool(
    #     default=True,
    #     semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    #     description="If True, return the data format in 3d numpy array."
    # )
    # flatten = hyperparams.UniformBool(
    #     default=True,
    #     semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    #     description="If True, flatten the returned array in 2d."
    # )
    flatten_order= hyperparams.Enumeration(
        values=['C', 'F', 'A'],
        default='F',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Decide the order of the flatten for multivarite sequences."
    )


    # Control
    columns_using_method= hyperparams.Enumeration(
        values=['name', 'index'],
        default='index',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Choose to use columns by names or indecies. If 'name', \"use_columns\" or \"exclude_columns\" is used. If 'index', \"use_columns_name\" or \"exclude_columns_name\" is used."
    )
    use_columns_name = hyperparams.Set(
        elements=hyperparams.Hyperparameter[str](''),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column names to force primitive to operate on. If any specified column cannot be parsed, it is skipped.",
    )
    exclude_columns_name = hyperparams.Set(
        elements=hyperparams.Hyperparameter[str](''),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column names to not operate on. Applicable only if \"use_columns_name\" is not provided.",
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
        default='replace',
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
        values=['https://metadata.datadrivendiscovery.org/types/Attribute', 'https://metadata.datadrivendiscovery.org/types/ConstructedAttribute'],
        default='https://metadata.datadrivendiscovery.org/types/Attribute',
        description='Decides what semantic type to attach to generated attributes',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )

    
class SubsequenceSegmentationPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    Subsequence Time Seires Segmentation.

Parameters
----------
    window_size : int
        The moving window size.
    step : int, optional (default=1)
        The displacement for moving window.
    return_numpy : bool, optional (default=True)
        If True, return the data format in 3d numpy array.
    flatten : bool, optional (default=True)
        If True, flatten the returned array in 2d.
    flatten_order : str, optional (default='F')
        Decide the order of the flatten for multivarite sequences.
        ‘C’ means to flatten in row-major (C-style) order.
        ‘F’ means to flatten in column-major (Fortran- style) order.
        ‘A’ means to flatten in column-major order if a is Fortran contiguous in memory,
        row-major order otherwise. ‘K’ means to flatten a in the order the elements occur in memory.
        The default is ‘F’. 
    use_columns: Set
        A set of column indices to force primitive to operate on. If any specified column cannot be parsed, it is skipped.   
    exclude_columns: Set
        A set of column indices to not operate on. Applicable only if \"use_columns\" is not provided. 
    return_result: Enumeration
        Should parsed columns be appended, should they replace original columns, or should only parsed columns be returned? This hyperparam is ignored if use_semantic_types is set to false.  
    use_semantic_types: Bool
        Controls whether semantic_types metadata will be used for filtering columns in input dataframe. Setting this to false makes the code ignore return_result and will produce only the output dataframe.
    add_index_columns: Bool
        Also include primary index columns if input data has them. Applicable only if \"return_result\" is set to \"new\".
    error_on_no_input: Bool
        Throw an exception if no input column is selected/provided. Defaults to true to behave like sklearn. To prevent pipelines from breaking set this to False.
    return_semantic_type: Enumeration[str](
        Decides what semantic type to attach to generated attributes'
    """

    metadata = construct_primitive_metadata(module='timeseries_processing', name='subsequence_segmentation', id='SubsequenceSegmentationPrimitive', primitive_family='data_preprocessing', hyperparams=['window_size', 'step', 'flatten_order'], description='Subsequence Segmentation Primitive')
    
    


    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Process the testing data.
        Args:
            inputs: Container DataFrame.

        Returns:
            Container DataFrame after BKFilter.
        """
        # Get cols to fit.
        self._fitted = False
        self._training_inputs, self._training_indices = self._get_columns_to_fit(inputs, self.hyperparams)
        self._input_column_names = self._training_inputs.columns


        if len(self._training_indices) > 0:
            # self._clf.fit(self._training_inputs)
            self._fitted = True
        else: # pragma: no cover
            if self.hyperparams['error_on_no_input']:
                raise RuntimeError("No input columns were selected")
            self.logger.warn("No input columns were selected")



        if not self._fitted: # pragma: no cover
            raise PrimitiveNotFittedError("Primitive not fitted.")
        sk_inputs = inputs

        if self.hyperparams['use_semantic_types']: # pragma: no cover
            sk_inputs = inputs.iloc[:, self._training_indices]
        output_columns = []
        if len(self._training_indices) > 0:
            sk_output = self._get_sub_matrices(sk_inputs, 
                                                window_size=self.hyperparams['window_size'],
                                                step=self.hyperparams['step'],
                                                flatten_order=self.hyperparams['flatten_order'])
            if sparse.issparse(sk_output):
                sk_output = sk_output.toarray()

            outputs = self._wrap_predictions(inputs, sk_output)

            if len(outputs.columns) == len(self._input_column_names):
                outputs.columns = self._input_column_names
            output_columns = [outputs]         
            
        else: # pragma: no cover
            if self.hyperparams['error_on_no_input']:
                raise RuntimeError("No input columns were selected")
            self.logger.warn("No input columns were selected")

        # outputs = base_utils.combine_columns(return_result=self.hyperparams['return_result'],
        #                                        add_index_columns=self.hyperparams['add_index_columns'],
        #                                        inputs=inputs, column_indices=self._training_indices,
        #                                        columns_list=output_columns)
        
        # print(outputs.shape)
        # self._write(outputs)
        return CallResult(outputs)
        
    
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

        def can_produce_column(column_index: int) -> bool: # pragma: no cover
            return cls._can_produce_column(inputs_metadata, column_index, hyperparams)

        use_columns = []
        exclude_columns = []

        # if hyperparams['columns_using_method'] == 'name':
        #     inputs_cols = inputs.columns.values.tolist()
        #     for i in range(len(inputs_cols)):
        #         if inputs_cols[i] in hyperparams['use_columns_name']:
        #             use_columns.append(i)
        #         elif inputs_cols[i] in hyperparams['exclude_columns_name']:
        #             exclude_columns.append(i)      
        # else: 
        use_columns=hyperparams['use_columns']
        exclude_columns=hyperparams['exclude_columns']           
        
        columns_to_produce, columns_not_to_produce = base_utils.get_columns_to_use(inputs_metadata, use_columns=use_columns, exclude_columns=exclude_columns, can_use_column=can_produce_column)
        return inputs.iloc[:, columns_to_produce], columns_to_produce
        # return columns_to_produce

    @classmethod
    def _can_produce_column(cls, inputs_metadata: metadata_base.DataMetadata, column_index: int, hyperparams: Hyperparams) -> bool: # pragma: no cover
        """
        Output whether a column can be processed.
        Args:
            inputs_metadata: d3m.metadata.base.DataMetadata
            column_index: int

        Returns:
            boolnp
        """
        column_metadata = inputs_metadata.query((metadata_base.ALL_ELEMENTS, column_index))

        accepted_structural_types = (int, float, np.integer, np.float64)
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
    def _update_predictions_metadata(cls, inputs_metadata: metadata_base.DataMetadata, outputs: Optional[Outputs],
                                     target_columns_metadata: List[OrderedDict]) -> metadata_base.DataMetadata: # pragma: no cover
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

    def _wrap_predictions(self, inputs: Inputs, predictions: ndarray) -> Outputs: # pragma: no cover
        """
        Wrap predictions into dataframe
        Args:
            inputs: Container Dataframe
            predictions: array-like data (n_samples, n_features)

        Returns:
            Dataframe
        """
        outputs = d3m_dataframe(predictions, generate_metadata=True)
        target_columns_metadata = self._add_target_columns_metadata(outputs.metadata, self.hyperparams)
        outputs.metadata = self._update_predictions_metadata(inputs.metadata, outputs, target_columns_metadata)
        return outputs


    @classmethod
    def _add_target_columns_metadata(cls, outputs_metadata: metadata_base.DataMetadata, hyperparams): # pragma: no cover
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
            column_name = "output_{}".format(column_index)
            column_metadata = OrderedDict()
            semantic_types = set()
            semantic_types.add(hyperparams["return_semantic_type"])
            column_metadata['semantic_types'] = list(semantic_types)

            column_metadata["name"] = str(column_name)
            target_columns_metadata.append(column_metadata)

        return target_columns_metadata

    def _write(self, inputs:Inputs): # pragma: no cover
        inputs.to_csv(str(time.time())+'.csv')

    def _get_sub_sequences_length(self, n_samples, window_size, step): # pragma: no cover
        """Pseudo chop a univariate time series into sub sequences. Return valid
        length only.
        Parameters
        ----------
        X : numpy array of shape (n_samples,)
            The input samples.
        window_size : int
            The moving window size.
        step_size : int, optional (default=1)
            The displacement for moving window.
        Returns
        -------
        valid_len : int
            The number of subsequences.

        """
        # valid_len = int(np.floor((n_samples - window_size) / step)) + 1
        valid_len = int(np.ceil(n_samples / step))
        return valid_len
        # 


    def _get_sub_matrices(self, X, window_size, step=1, flatten_order='F'): # pragma: no cover
        """
        Chop a multivariate time series into sub sequences (matrices).
        Parameters
        ----------
        X : numpy array of shape (n_samples,)
            The input samples.
        window_size : int
            The moving window size.
        step : int, optional (default=1)
            The displacement for moving window.

        return_numpy : bool, optional (default=True)
            If True, return the data format in 3d numpy array.
        flatten : bool, optional (default=True)
            If True, flatten the returned array in 2d.

        flatten_order : str, optional (default='F')
            Decide the order of the flatten for multivarite sequences.
            ‘C’ means to flatten in row-major (C-style) order.
            ‘F’ means to flatten in column-major (Fortran- style) order.
            ‘A’ means to flatten in column-major order if a is Fortran contiguous in memory,
            row-major order otherwise. ‘K’ means to flatten a in the order the elements occur in memory.
            The default is ‘F’.
        Returns
        -------
        X_sub : numpy array of shape (valid_len, window_size*n_sequences)
            The numpy matrix with each row stands for a flattend submatrix.
        """
        X = check_array(X).astype(np.float)
        n_samples, n_sequences = X.shape[0], X.shape[1]

        # get the valid length
        valid_len = self._get_sub_sequences_length(n_samples, window_size, step)

        X_sub = []
        X_left_inds = []
        X_right_inds = []

        # Added by JJ
        X = np.append(X, np.zeros([window_size, n_sequences]), axis=0)

        # exclude the edge
        steps = list(range(0, n_samples, step))
        steps = steps[:valid_len]

        # print(n_samples, n_sequences)
        for idx, i in enumerate(steps):
            X_sub.append(X[i: i + window_size, :])
            X_left_inds.append(i)
            X_right_inds.append(i + window_size)

        X_sub = np.asarray(X_sub)


        temp_array = np.zeros([valid_len, window_size * n_sequences])
        if flatten_order == 'C':
            for i in range(valid_len):
                temp_array[i, :] = X_sub[i, :, :].flatten(order='C')

        else:
            for i in range(valid_len):
                temp_array[i, :] = X_sub[i, :, :].flatten(order='F')
        
        return temp_array #, np.asarray(X_left_inds), np.asarray(X_right_inds)

            # else:
            #     return np.asarray(X_sub), np.asarray(X_left_inds), np.asarray(X_right_inds)
        # else:
        #     return X_sub, np.asarray(X_left_inds), np.asarray(X_right_inds)


    


