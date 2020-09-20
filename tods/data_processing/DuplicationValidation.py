from d3m import container
from d3m.primitive_interfaces import base, transformer
from d3m.metadata import base as metadata_base, hyperparams

import os.path
from d3m import utils

import time

__all__ = ('DuplicationValidation',)

Inputs = container.DataFrame
Outputs = container.DataFrame


class Hyperparams(hyperparams.Hyperparams):
    """
    
    """
    keep_option = hyperparams.Enumeration(
        values=['first', 'average'],
        default='first',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="When dropping rows, choose to keep the first one of duplicated data or calculate their average",
    )


class DuplicationValidation(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    Check whether the seires data involves duplicate data in one timestamp, and provide processing if the duplication exists.

    Parameters
    ----------
    keep_option: enumeration
        When dropping rows, choose to keep the first one or calculate the average
    """
 
    __author__: "DATA Lab at Texas A&M University"
    metadata = metadata_base.PrimitiveMetadata({
         "name": "duplication validation primitive",
         "python_path": "d3m.primitives.tods.data_processing.duplication_validation",
         "source": {'name': 'DATA Lab at Texas A&M University', 'contact': 'mailto:khlai037@tamu.edu', 
         'uris': ['https://gitlab.com/lhenry15/tods.git', 'https://gitlab.com/lhenry15/tods/-/blob/Junjie/anomaly-primitives/anomaly_primitives/DuplicationValidation.py']},
         "algorithm_types": [metadata_base.PrimitiveAlgorithmType.DUPLICATION_VALIDATION,],
         "primitive_family": metadata_base.PrimitiveFamily.DATA_PREPROCESSING,
         "id": "cf6d8137-73d8-496e-a2e3-49f941ee716d",
         "hyperparams_to_tune": ['keep_option'],
         "version": "0.0.1",
    })

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

    def _write(self, inputs:Inputs):
        """
        write inputs to current directory, only for test
        """
        inputs.to_csv(str(time.time())+'.csv')
