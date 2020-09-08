from d3m import container, exceptions
from d3m.primitive_interfaces import base, transformer
from d3m.metadata import base as metadata_base, hyperparams

import os.path
from d3m import utils

import time

__all__ = ('ContinuityValidation',)

Inputs = container.DataFrame
Outputs = container.DataFrame


class Hyperparams(hyperparams.Hyperparams):
    continuity_option = hyperparams.Enumeration(
        values=['ablation', 'imputation'],
        default='imputation',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Choose ablation or imputation the original data",
    )

    interval = hyperparams.Uniform(
        default = 1,
        lower = 0.000000001,
        upper = 10000000000,
        description='Only used in imputation, give the timestamp interval.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )


class ContinuityValidation(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    Check whether the seires data is consitent in time interval and provide processing if not consistent.

    Parameters
    ----------
    continuity_option: enumeration
        Choose ablation or imputation.
            ablation: delete some rows and increase timestamp interval to keep the timestamp consistent
            imputation: linearly imputate the absent timestamps to keep the timestamp consistent
    interval: float
        Only used in imputation, give the timestamp interval. ‘interval’ should be an integral multiple of 'timestamp' or 'timestamp' should be an integral multiple of ‘interval’
    """

    __author__: "DATA Lab at Texas A&M University"
    metadata = metadata_base.PrimitiveMetadata({
         "name": "continuity validation primitive",
         "python_path": "d3m.primitives.tods.data_processing.continuity_validation",
         "source": {'name': 'DATA Lab at Texas A&M University', 'contact': 'mailto:khlai037@tamu.edu', 
         'uris': ['https://gitlab.com/lhenry15/tods.git', 'https://gitlab.com/lhenry15/tods/-/blob/Junjie/anomaly-primitives/anomaly_primitives/ContinuityValidation.py']},
         "algorithm_types": [metadata_base.PrimitiveAlgorithmType.CONTINUITY_VALIDATION, ],
         "primitive_family": metadata_base.PrimitiveFamily.DATA_PREPROCESSING,
         "id": "ef8fb025-d157-476c-8e2e-f8fe56162195",
         "hyperparams_to_tune": ['continuity_option', 'interval'],
         "version": "0.0.1",
    })

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        """
        Args:
            inputs: Container DataFrame
            timeout: Default
            iterations: Default

        Returns:
            Container DataFrame with consistent timestamp

        """
        # self.logger.warning('Hi, ContinuityValidation.produce was called!')
        if self.hyperparams['continuity_option'] == 'ablation':
            outputs = self._continuity_ablation(inputs)

        if self.hyperparams['continuity_option'] == 'imputation':
            outputs = self._continuity_imputation(inputs)

        
        outputs.reset_index(drop=True, inplace=True)
        self._update_metadata(outputs)

        # self._write(outputs)
        return base.CallResult(outputs)


    def _update_metadata(self, outputs):
        outputs.metadata = outputs.metadata.generate(outputs)


    def _continuity_ablation(self, inputs: Inputs):

        ablation_set = self._find_ablation_set(inputs)
        inputs = inputs.loc[inputs['timestamp'].isin(ablation_set)].copy()

        inputs.sort_values("timestamp",inplace=True)
        inputs['d3mIndex'] = list(range(inputs.shape[0]))

        return inputs


    def _find_ablation_set(self, inputs):
        """
        Find the longest series with minimum timestamp interval of inputs
        """
        # find the min inteval and max interval
        min_interval = inputs.iloc[1]['timestamp'] - inputs.iloc[0]['timestamp']
        for i in range(2, inputs.shape[0]):
            curr_interval = inputs.iloc[i]['timestamp'] - inputs.iloc[i - 1]['timestamp']
            if min_interval > curr_interval:
                min_interval = curr_interval

        max_interval = ((inputs.iloc[-1]['timestamp'] - inputs.iloc[0]['timestamp']) + min_interval * (2 - inputs.shape[0]))

        print((inputs.iloc[-1]['timestamp'] - inputs.iloc[0]['timestamp']), inputs.shape[0])

        interval = min_interval
        ablation_set = list()
        origin_set = set(inputs['timestamp'])

        print(min_interval, max_interval)

        while interval <= max_interval:
            start = 0
            while  (inputs.iloc[start]['timestamp'] <= inputs.iloc[0]['timestamp'] + max_interval) and (inputs.iloc[start]['timestamp'] <= inputs.iloc[-1]['timestamp']):
                tmp_list = list()
                tmp = utils.numpy.arange(start=inputs.iloc[start]['timestamp'], step=interval,stop=inputs.iloc[-1]['timestamp'])

                for i in tmp:
                    if i in origin_set:
                        tmp_list.append(i)
                    else: break

                ablation_set.append(tmp_list)
                start += 1

            interval += min_interval

        max_size_index = 0
        for i in range(1, len(ablation_set)):
            if len(ablation_set[i]) > len(ablation_set[max_size_index]):
                max_size_index = i
        return ablation_set[max_size_index]


    def _continuity_imputation(self, inputs: Inputs):
        """
        Linearly imputate the missing timestmap and value of inputs
        """
        interval = self.hyperparams['interval']
        time1 = inputs.iloc[0]['timestamp']
      
        for i in range(1, inputs.shape[0]):
            time2 = inputs.iloc[i]['timestamp']
            if time2 - time1 != interval:
            
                blank_number = int((time2 - time1) / interval) # how many imputation should there be between two timestamps in original data
                for j in range(1, blank_number):

                    dict = {'timestamp':[time1 + interval * j], 'ground_truth':[int(inputs.iloc[i]['ground_truth'])]}

                    for col in list(inputs.columns.values):
                        if not col in ['d3mIndex', 'timestamp', 'ground_truth']:
                            dict[col] = [inputs.iloc[i-1][col] + (inputs.iloc[i][col] - inputs.iloc[i-1][col]) / blank_number * j]
                    
                    inputs = inputs.append(utils.pandas.DataFrame(dict), ignore_index=True, sort=False)
                                
            time1 = time2

        inputs.sort_values("timestamp",inplace=True)
        inputs['d3mIndex'] = list(range(inputs.shape[0]))
        return inputs


    def _write(self, inputs:Inputs):
        """
        write inputs to current directory, only for test
        """
        inputs.to_csv(str(time.time())+'.csv')
