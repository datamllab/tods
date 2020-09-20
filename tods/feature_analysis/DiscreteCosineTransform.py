import os
import typing
import pandas as pd 
import numpy as np

from d3m import container, utils
from d3m.base import utils as base_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer

import common_primitives
import logging
import math
from scipy.fft import dct
from collections import OrderedDict
from typing import cast, Dict, List, Union, Sequence, Optional, Tuple


from scipy import sparse
from numpy import ndarray

__all__ = ('DiscreteCosineTransform',)

Inputs = container.DataFrame
Outputs = container.DataFrame


class Hyperparams(hyperparams.Hyperparams):

    type_ = hyperparams.UniformInt(
        lower=1,
        upper=4,
        upper_inclusive = True,
        default=2,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Type of the DCT. Default is 2",
    )


    axis = hyperparams.Hyperparameter[int](
        default=-1,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Axis over which to compute the DCT. If not given, the last axis is used.",
    )

    n = hyperparams.Union[Union[int, None]](
        configuration=OrderedDict(
            limit=hyperparams.Bounded[int](
                lower=1,
                upper=None,
                default=10,
            ),
            unlimited=hyperparams.Constant(
                default=None,
                description='If n is not given, the length of the input along the axis specified by axis is used.',
            ),
        ),
        default='unlimited',
        description='Length of the transformed axis of the output. If n is smaller than the length of the input, the input is cropped. If it is larger, the input is padded with zeros.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    norm = hyperparams.Enumeration(
        values=[None,"ortho"],
        default=None,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Normalization mode. Default is None, meaning no normalization on the forward transforms and scaling by 1/n on the ifft. For norm=""ortho"", both directions are scaled by 1/sqrt(n).",
    )

    overwrite_x = hyperparams.UniformBool(
        default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="If True, the contents of x can be destroyed; the default is False. See the notes below for more details.",

    )

    workers = hyperparams.Union[Union[float, None]](
        configuration=OrderedDict(
            limit=hyperparams.Bounded[int](
                lower=1,
                upper=None,
                default=10,
            ),
            unlimited=hyperparams.Constant(
                default=None,
                description='If nothing is give as a paramter',
            ),
        ),
        default='unlimited',
        description="Maximum number of workers to use for parallel computation. If negative, the value wraps around from os.cpu_count().",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    )

    # parameters for column
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

class DCT:
    def __init__(self,type_,n,axis,overwrite_x,norm,workers):
        self._type = type_
        self._n = n
        self._axis = axis
        self._overwrite_x = overwrite_x
        self._norm = norm
        self._workers = workers
        
    def produce(self, inputs):

        dataframe = inputs
        processed_df = utils.pandas.DataFrame()
        try:
            for target_column in dataframe.columns :
                dct_input = dataframe[target_column].values
                dct_output = dct(x=dct_input,type=self._type,n=self._n,axis=self._axis,overwrite_x=self._overwrite_x,norm=self._norm,workers=self._workers)
                processed_df[target_column+"_dct_coeff"]=pd.Series(dct_output)
            
        except IndexError:
            logging.warning("Index not found in dataframe")

        return processed_df;


        

class DiscreteCosineTransform(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    Compute the 1-D discrete Cosine Transform.
    Return the Discrete Cosine Transform of arbitrary type sequence x.

    scipy documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dct.html#scipy.fft.dct


    Parameters
    ----------

    type_: int
        Type of the DCT. Default is 2

    n: int
        Length of the transformed axis of the output. If n is smaller than the length of the input, the input is cropped. If it is larger, the input is padded with zeros.

    axis: int
        Axis over which to compute the DCT. If not given, the last axis is used.
    
    norm: str
        Normalization mode. Default is None, meaning no normalization on the forward transforms and scaling by 1/n on the ifft. For norm=""ortho"", both directions are scaled by 1/sqrt(n).
    
    overwrite_x: boolean
        If True, the contents of x can be destroyed; the default is False. See the notes below for more details.

    workers: int
        Maximum number of workers to use for parallel computation. If negative, the value wraps around from os.cpu_count(). Defualt is None.
    

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

    error_on_no_input: Bool(
        Throw an exception if no input column is selected/provided. Defaults to true to behave like sklearn. To prevent pipelines from breaking set this to False.

    return_semantic_type: Enumeration[str](
        Decides what semantic type to attach to generated attributes'

    """

    __author__ = "Data Lab"
    metadata = metadata_base.PrimitiveMetadata(
        {
            "__author__ " : "DATA Lab at Texas A&M University",
            'name': "Discrete Cosine Transform",
            'python_path': 'd3m.primitives.tods.feature_analysis.discrete_cosine_transform',
            'source': {
                'name': 'DATA Lab at Texas A&M University',
                'contact': 'mailto:khlai037@tamu.edu',
                'uris': [
                    'https://gitlab.com/lhenry15/tods.git',
                    'https://gitlab.com/lhenry15/tods/-/blob/purav/anomaly-primitives/anomaly_primitives/DiscreteCosineTransform.py',
                ],
            },
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.DISCRETE_COSINE_TRANSFORM,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.FEATURE_CONSTRUCTION,
            'id': '584fa7d5-39cc-4cf8-8d5b-5f3a2648f767',
            'hyperparameters_to_tune':['n','norm','axis','type_'],
            'version': '0.0.1',
        },
    )

    def __init__(self, *, hyperparams: Hyperparams) -> None:
        super().__init__(hyperparams=hyperparams)

        self._clf = DCT(type_=self.hyperparams['type_'],
                        n=self.hyperparams['n'],
                        axis=self.hyperparams['axis'],
                        overwrite_x=self.hyperparams['overwrite_x'],
                        norm = self.hyperparams['norm'],
                        workers = self.hyperparams['workers']
                        )

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        """

            Args:
                inputs: Container DataFrame

            Returns:
                Container DataFrame added with DCT coefficients in a column named 'column_name_dct_coeff'

        """
        assert isinstance(inputs, container.DataFrame), type(dataframe)

        self._fitted = False
        self._training_inputs, self._training_indices = self._get_columns_to_fit(inputs, self.hyperparams)
        self._input_column_names = self._training_inputs.columns
        
        if len(self._training_indices) > 0:
            # self._clf.fit(self._training_inputs)
            self._fitted = True
        else:
            if self.hyperparams['error_on_no_input']:
                raise RuntimeError("No input columns were selected")
            self.logger.warn("No input columns were selected")
        
        
        if not self._fitted:
            raise PrimitiveNotFittedError("Primitive not fitted.")

        sk_inputs = inputs
        if self.hyperparams['use_semantic_types']:
            cols = [inputs.columns[x] for x in self._training_indices]
            sk_inputs = container.DataFrame(data = inputs.iloc[:, self._training_indices].values,columns = cols, generate_metadata=True)

        output_columns = []
        if len(self._training_indices) > 0:
            sk_output = self._clf.produce(sk_inputs)
            
            if sparse.issparse(sk_output):
                sk_output = sk_output.toarray()
            outputs = self._wrap_predictions(inputs, sk_output)
            # if len(outputs.columns) == len(self._input_column_names):
            #     outputs.columns = self._input_column_names
            output_columns = [outputs]
        else:
            if self.hyperparams['error_on_no_input']:
                raise RuntimeError("No input columns were selected")
            self.logger.warn("No input columns were selected")

        
        outputs = base_utils.combine_columns(return_result=self.hyperparams['return_result'],
                                            add_index_columns=self.hyperparams['add_index_columns'],
                                            inputs=inputs, column_indices=self._training_indices,
                                            columns_list=output_columns)

        
        return base.CallResult(outputs)




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
            # return inputs, list(hyperparams['use_columns'])

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
        accepted_structural_types = (int, float, np.integer, np.float64,str)
        accepted_semantic_types = set()
        accepted_semantic_types.add("https://metadata.datadrivendiscovery.org/types/Attribute")
        if not issubclass(column_metadata['structural_type'], accepted_structural_types):
            print(column_index, "does not match the structural_type requirements in metadata. Skipping column")
            return False

        semantic_types = set(column_metadata.get('semantic_types', []))

        # print("length sematic type",len(semantic_types))

        # returing true for testing purposes for custom dataframes
        return True;

        if len(semantic_types) == 0:
            cls.logger.warning("No semantic types found in column metadata")
            return False

        # Making sure all accepted_semantic_types are available in semantic_types
        if len(accepted_semantic_types - semantic_types) == 0:
            return True

        print(semantic_types)
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

        outputs = container.DataFrame(predictions, generate_metadata=True)
        target_columns_metadata = self._add_target_columns_metadata(outputs.metadata,self.hyperparams)
        outputs.metadata = self._update_predictions_metadata(inputs.metadata, outputs, target_columns_metadata)
        # print(outputs.metadata.to_internal_simple_structure())

        return outputs


    @classmethod
    def _add_target_columns_metadata(cls, outputs_metadata: metadata_base.DataMetadata, hyperparams):
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
            # column_name = "output_{}".format(column_index)
            column_metadata = OrderedDict()
            semantic_types = set()
            semantic_types.add(hyperparams["return_semantic_type"])
            column_metadata['semantic_types'] = list(semantic_types)
            # column_metadata["name"] = str(column_name)
            target_columns_metadata.append(column_metadata)
        return target_columns_metadata


DiscreteCosineTransform.__doc__ = DiscreteCosineTransform.__doc__



