import os
import uuid
import typing
import collections

import numpy as np
import pandas as pd


from datetime import datetime, timezone
from d3m.primitive_interfaces import base, transformer
from d3m import container
from d3m.metadata import base as metadata_base, hyperparams


__all__ = ('TimeIntervalTransformPrimitive',)

Inputs = container.DataFrame
Outputs = container.DataFrame

from tods.utils import construct_primitive_metadata

"""
TODO: Implementation for up-sampling the data (when time_interval is less than current time series interval)
"""

class Hyperparams(hyperparams.Hyperparams):
    time_interval = hyperparams.Hyperparameter[typing.Union[str, None]](
        default='5T',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='timestamp to transform.'
    )
    
    # Keep previous
    dataframe_resource = hyperparams.Hyperparameter[typing.Union[str, None]](
        default=None,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Resource ID of a DataFrame to extract if there are multiple tabular resources inside a Dataset and none is a dataset entry point.",
    )
    use_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(2,),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to force primitive to operate on. If any specified column cannot be parsed, it is skipped.",
    )
    exclude_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(0,1,3,),
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
    


class TimeIntervalTransformPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    
    """
    A primitive which configures the time interval of the dataframe.
    Resample the timestamps based on the time_interval passed as hyperparameter

    Parameters
    -----------
    time_interval :Union[str, None]
        timestamp to transform.   
    dataframe_resource :Union[str, None]
        Resource ID of a DataFrame to extract if there are multiple tabular resources inside a Dataset and none is a dataset entry point.
    use_columns :Set
        A set of column indices to force primitive to operate on. If any specified column cannot be parsed, it is skipped.
    exclude_columns :Set
        A set of column indices to not operate on. Applicable only if \"use_columns\" is not provided.
    return_result :Enumeration
        Should parsed columns be appended, should they replace original columns, or should only parsed columns be returned? This hyperparam is ignored if use_semantic_types is set to false.
    use_semantic_types :Bool
        Controls whether semantic_types metadata will be used for filtering columns in input dataframe. Setting this to false makes the code ignore return_result and will produce only the output dataframe
    add_index_columns :Bool
        Also include primary index columns if input data has them. Applicable only if \"return_result\" is set to \"new\".
    error_on_no_input :Bool
        Throw an exception if no input column is selected/provided. Defaults to true to behave like sklearn. To prevent pipelines from breaking set this to False.
    return_semantic_type :Enumeration[str]
        Decides what semantic type to attach to generated attributes
    """
    
    metadata = construct_primitive_metadata(module='data_processing', name='time_interval_transform', id='TimeIntervalTransformPrimitive', primitive_family='data_preprocessing', description='Time Interval Transform')


    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        resource = inputs.reset_index(drop=True)
        
        """
        Args:
            inputs: Container DataFrame
        Returns:
            Container DataFrame with resampled time intervals
        """

        if self.hyperparams['time_interval'] is None: # pragma: no cover
            time_interval = '5T'
        else:
            time_interval = self.hyperparams['time_interval']

        try:
            outputs = self._time_interval_transform(inputs, hyperparams)
            
        except Exception as e: # pragma: no cover
            self.logger.error("Error in Performing Time Interval Transform",e)

        self._update_metadata(outputs)

        return base.CallResult(outputs)

    def _time_interval_transform(self, inputs: Inputs, hyperparams: Hyperparams):

        """
        Args:
            inputs: Container DataFrame
        Returns:
            Container DataFrame with resampled time intervals
        """
        
        #configure dataframe for resampling
        inputs['timestamp'] = pd.to_datetime(inputs['timestamp'], unit='s')
        inputs['timestamp'] = inputs['timestamp'].dt.tz_localize('US/Pacific')
        inputs = inputs.set_index('timestamp')

        #resample dataframe
        inputs = inputs.resample(self.hyperparams['time_interval']).mean()

        #configure dataframe to original format
        inputs = inputs.reset_index()
        value_columns = list(set(inputs.columns) - set(['d3mIndex', 'timestamp', 'ground_truth']))
        inputs = inputs.reindex(columns=['d3mIndex','timestamp'] + value_columns + ['ground_truth'])
        inputs['timestamp'] = inputs['timestamp'].astype(np.int64) // 10 ** 9
        inputs['d3mIndex'] = range(0, len(inputs))

        """
        Since the mean of the ground_truth was taken for a set interval, 
        we should set those values that are greater than 0 to 1 so they are consistent with original data
        """
        for i in range(len(inputs['ground_truth'])):
            if(inputs['ground_truth'][i] > 0):
                inputs.loc[i, 'ground_truth'] = 1

        inputs = container.DataFrame(inputs)    #convert pandas DataFrame back to d3m comtainer(Important)

        return inputs


    def _update_metadata(self, outputs):
        outputs.metadata = outputs.metadata.generate(outputs)

