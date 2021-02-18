
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


class Hyperparams(hyperparams.Hyperparams):
    pass

class TimeStampValidationPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    A primitive to check time series is sorted by time stamp , if not then return sorted time series
    """
    metadata = metadata_base.PrimitiveMetadata({
            '__author__': "DATA Lab at Texas A&M University",
            'version': '0.1.0',
            'name': 'Time Stamp Validation',
            'python_path': 'd3m.primitives.tods.data_processing.timestamp_validation',
            'keywords': ['Time Stamp', 'Sort Order'],
            'source': {
                'name': 'DATA Lab at Texas A&M University',
                'contact': 'mailto:khlai037@tamu.edu'
            },
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.TODS_PRIMITIVE,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.DATA_VALIDATION,
	    'id': str(uuid.uuid3(uuid.NAMESPACE_DNS, 'TimeStampValidationPrimitive')),

        })

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
