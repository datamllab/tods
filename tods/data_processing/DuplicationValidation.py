from d3m import container
from d3m.primitive_interfaces import base, transformer
from d3m.metadata import base as metadata_base, hyperparams
import uuid

import os.path
from d3m import utils

import time

__all__ = ('DuplicationValidationPrimitive',)

Inputs = container.DataFrame
Outputs = container.DataFrame

from tods.utils import construct_primitive_metadata

class Hyperparams(hyperparams.Hyperparams):
    """
    
    """
    keep_option = hyperparams.Enumeration(
        values=['first', 'average'],
        default='first',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="When dropping rows, choose to keep the first one of duplicated data or calculate their average",
    )


class DuplicationValidationPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    Check whether the seires data involves duplicate data in one timestamp, and provide processing if the duplication exists.

    Parameters
    ----------
    keep_option :enumeration
        When dropping rows, choose to keep the first one or calculate the average
    """


    __author__: "DATA Lab at Texas A&M University"
    
    
    metadata = construct_primitive_metadata(module='data_processing', name='duplication_validation', id='DuplicationValidationPrimitive', primitive_family='data_preprocessing', description='duplication validation primitive')

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        """
        Args:
            inputs: Container DataFrame
            timeout: Default
            iterations: Default

        Returns:
            Container DataFrame after drop the duplication
        """
        # self.logger.warning('Hi, DuplicationValidation.produce was called!')

        if self.hyperparams['keep_option'] == 'first':
            outputs = self._timestamp_keep_first(inputs)

        if self.hyperparams['keep_option'] == 'average':
            outputs = self._timestamp_keep_average(inputs)

        self._update_metadata(outputs)
            
        # self._write(outputs)
        return base.CallResult(outputs)
    
    def _update_metadata(self, outputs):
        outputs.metadata = outputs.metadata.generate(outputs)

    def _timestamp_keep_first(self, inputs: Inputs):
        return inputs.drop_duplicates(subset=['timestamp'],keep='first')

    def _timestamp_keep_average(self, inputs: Inputs):
        inputs_copy = inputs.copy()
        inputs = inputs.drop_duplicates(subset=['timestamp'],keep='first')
        
        inputs_copy = inputs_copy.groupby('timestamp').mean().reset_index()

        for col in list(inputs.columns.values):
            if not col in ['d3mIndex', 'timestamp', 'ground_truth']:
                
                inputs[col] = inputs_copy[col].values
                

        return inputs

