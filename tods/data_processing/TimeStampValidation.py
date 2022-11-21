
import os
import typing
import numpy
import uuid

from d3m import container, utils as d3m_utils
from d3m.metadata import base as metadata_base
from d3m.metadata import hyperparams
from d3m.primitive_interfaces import base, transformer


__all__ = ('TimeStampValidationPrimitive',)

Inputs = container.DataFrame
Outputs = container.DataFrame

from tods.utils import construct_primitive_metadata

class Hyperparams(hyperparams.Hyperparams):
    pass

class TimeStampValidationPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    A primitive to check time series is sorted by time stamp , if not then return sorted time series.
    """
    
    metadata = construct_primitive_metadata(module='data_processing', name='timestamp_validation', id='TimeStampValidationPrimitive', primitive_family='data_validate', description='Time Stamp Validation')

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        """

        Args:
            inputs: Container DataFrame
            timeout: Default
            iterations: Default

        Returns:
            Container DataFrame sorted by Time Stamp

        """
        self.logger.info('Time Stamp order validation called')
        outputs = inputs
        try:
            if (self._is_time_stamp_sorted(inputs, 'timestamp')):
                outputs = inputs
            else:
                outputs = inputs.sort_values(by=["timestamp"])


            self._update_metadata(outputs)

            outputs.reset_index(drop=True, inplace=True)
            self.logger.info('Type of data : %s',type(outputs))

        except Exception as e :
            self.logger.error('Time Stamp order validation error  %s :',e)
        print(self.logger.info(base.CallResult(outputs).value))
        return base.CallResult(outputs)

    def _is_time_stamp_sorted(self,input:Inputs,column:str = 'timestamp') -> bool :
        """

        Args:
            input: Container Dataframe
            column: Column Name

        Returns:
            Boolean : True  if timestamp column is sorted  False if not

        """
        return all(input[column][i] <= input[column][i+1] for i in range(len(input[column])-1))

    def _update_metadata(self, outputs):
        outputs.metadata = outputs.metadata.generate(outputs)
