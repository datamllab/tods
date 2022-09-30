import os
from typing import Any,Optional,List
import statsmodels.api as sm
import numpy as np
from d3m import container, utils as d3m_utils
from d3m import utils

from numpy import ndarray
from collections import OrderedDict
from scipy import sparse
import os

import numpy
import typing
import time

from d3m import container
from d3m.primitive_interfaces import base, transformer

from d3m.container import DataFrame as d3m_dataframe
from d3m.metadata import hyperparams, params, base as metadata_base

from d3m.base import utils as base_utils
import uuid
from d3m.exceptions import PrimitiveNotFittedError

__all__ = ('SystemWiseDetectionPrimitive',)

Inputs = container.DataFrame
Outputs = container.DataFrame
from tods.utils import construct_primitive_metadata
class Params(params.Params): # pragma: no cover
       #to-do : how to make params dynamic
       use_column_names: Optional[Any]



class Hyperparams(hyperparams.Hyperparams): # pragma: no cover

       #Tuning Parameter
       #default -1 considers entire time series is considered
       window_size = hyperparams.Hyperparameter(default=10, semantic_types=[
           'https://metadata.datadrivendiscovery.org/types/TuningParameter',
       ], description="Window Size for decomposition")

       method_type = hyperparams.Enumeration(
           values=['max', 'avg', 'sliding_window_sum','majority_voting_sliding_window_sum','majority_voting_sliding_window_max'],
           default='majority_voting_sliding_window_max',
           semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
           description="The type of method used to find anomalous system",
       )
       contamination = hyperparams.Uniform(
           lower=0.,
           upper=0.5,
           default=0.1,
           description='The amount of contamination of the data set, i.e. the proportion of outliers in the data set. ',
           semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
       )

       #control parameter
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



class SystemWiseDetectionPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]): # pragma: no cover
    """
    Primitive to find abs_energy of time series.

    Parameters
    ----------
    window_size :int(default=10)
        Window Size for decomposition

    method_type :str ('max', 'avg', 'sliding_window_sum','majority_voting_sliding_window_sum','majority_voting_sliding_window_max')
        The type of method used to find anomalous system

    contamination : float in (0., 0.5), optional (default=0.1)
           The amount of contamination of the data set, i.e. the proportion of outliers in the data set. 
    """

    metadata = construct_primitive_metadata(module='detection_algorithm', name='system_wise_detection', id='Sytem_Wise_Anomaly_Detection_Primitive', primitive_family='anomaly_detect', hyperparams=['window_size','method_type','contamination'], description='Sytem_Wise_Anomaly_Detection_Primitive')

    def __init__(self, *, hyperparams: Hyperparams) -> None:
        super().__init__(hyperparams=hyperparams)
        self.primitiveNo = 0

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        """
        Args:
            inputs: Container DataFrame
            timeout: Default
            iterations: Default
        Returns:
            Container DataFrame containing abs_energy of  time series
        """

        self.logger.info('System wise Detection Input  Primitive called')
        
        # Get cols to fit.
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
        system_wise_detection_input = inputs
        if self.hyperparams['use_semantic_types']:
            system_wise_detection_input = inputs.iloc[:, self._training_indices]
        output_columns = []
        if len(self._training_indices) > 0:
            system_wise_detection_output = self._system_wise_detection(system_wise_detection_input,self.hyperparams["method_type"],self.hyperparams["window_size"],self.hyperparams["contamination"])
            outputs = system_wise_detection_output


            if sparse.issparse(system_wise_detection_output):
                system_wise_detection_output = system_wise_detection_output.toarray()
            outputs = self._wrap_predictions(inputs, system_wise_detection_output)

            #if len(outputs.columns) == len(self._input_column_names):
               # outputs.columns = self._input_column_names

            output_columns = [outputs]


        else:
            if self.hyperparams['error_on_no_input']:
                raise RuntimeError("No input columns were selected")
            self.logger.warn("No input columns were selected")


        self.logger.info('System wise Detection  Primitive returned')
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

        inputs_metadata = inputs.metadata

        def can_produce_column(column_index: int) -> bool:
            return cls._can_produce_column(inputs_metadata, column_index, hyperparams)

        use_columns = hyperparams['use_columns']
        exclude_columns = hyperparams['exclude_columns']

        columns_to_produce, columns_not_to_produce = base_utils.get_columns_to_use(inputs_metadata,
                                                                                   use_columns=use_columns,
                                                                                   exclude_columns=exclude_columns,
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
        return True
        if len(semantic_types) == 0:
            cls.logger.warning("No semantic types found in column metadata")
            return False

        # Making sure all accepted_semantic_types are available in semantic_types
        if len(accepted_semantic_types - semantic_types) == 0:
            return True

        return False

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
        target_columns_metadata = self._add_target_columns_metadata(outputs.metadata, self.hyperparams,self.primitiveNo)
        outputs.metadata = self._update_predictions_metadata(inputs.metadata, outputs, target_columns_metadata)

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

    def _write(self, inputs: Inputs):
        inputs.to_csv(str(time.time()) + '.csv')

    def _system_wise_detection(self,X,method_type,window_size,contamination):
        #systemIds = X.system_id.unique()
        systemIds = [int(idx) for idx in X.index]
        #groupedX = X.groupby(X.system_id)
        print(systemIds)
        print(X.iloc[0])
        systemDf = X.iloc(systemIds[0])['system']
        print(systemDf)
        exit()

        transformed_X = []
        if(method_type=="max"):
            """
            Sytems are sorted based on maximum of reconstruction errors"
            """
            maxOutlierScorePerSystemList = []
            for systemId in systemIds:
                systemDf = groupedX.get_group(systemId)
                #systemDf = X[systemId]['system']
                maxOutlierScorePerSystemList.append(np.max(np.abs(systemDf["value_0"].values)))

            ranking = np.sort(maxOutlierScorePerSystemList)
            threshold = ranking[int((1 - contamination) * len(ranking))]
            self.threshold = threshold
            mask = (maxOutlierScorePerSystemList >= threshold)
            ranking[mask] = 1
            ranking[np.logical_not(mask)] = 0
            for iter in range(len(systemIds)):
                transformed_X.append([systemIds[iter],ranking[iter]])

        if (method_type == "avg"):
            """
            Sytems are sorted based on average of reconstruction errors"
            """
            avgOutlierScorePerSystemList = []
            for systemId in systemIds:
                systemDf = groupedX.get_group(systemId)
                avgOutlierScorePerSystemList.append(np.mean(np.abs(systemDf["value_0"].values)))

            ranking = np.sort(avgOutlierScorePerSystemList)
            threshold = ranking[int((1 - contamination) * len(ranking))]
            self.threshold = threshold
            mask = (avgOutlierScorePerSystemList >= threshold)
            ranking[mask] = 1
            ranking[np.logical_not(mask)] = 0
            for iter in range(len(systemIds)):
                transformed_X.append([systemIds[iter], ranking[iter]])

        if (method_type == "sliding_window_sum"):
            """
            Sytems are sorted based on max of max of reconstruction errors in each window"
            """
            OutlierScorePerSystemList = []
            for systemId in systemIds:
                systemDf = groupedX.get_group(systemId)
                column_value = systemDf["value_0"].values
                column_score = np.zeros(len(column_value))
                for iter in range(window_size - 1, len(column_value)):
                    sequence = column_value[iter - window_size + 1:iter + 1]
                    column_score[iter] = np.sum(np.abs(sequence))
                column_score[:window_size - 1] = column_score[window_size - 1]
                OutlierScorePerSystemList.append(column_score.tolist())
            OutlierScorePerSystemList = np.asarray(OutlierScorePerSystemList)

            maxOutlierScorePerSystemList = OutlierScorePerSystemList.max(axis=1).tolist()

            ranking = np.sort(maxOutlierScorePerSystemList)
            threshold = ranking[int((1 - contamination) * len(ranking))]
            self.threshold = threshold
            mask = (maxOutlierScorePerSystemList >= threshold)
            ranking[mask] = 1
            ranking[np.logical_not(mask)] = 0
            for iter in range(len(systemIds)):
                transformed_X.append([systemIds[iter], ranking[iter]])

        if (method_type == "majority_voting_sliding_window_sum"):
            """
            Sytem with most vote based on max of sum of reconstruction errors in each window
            """
            OutlierScorePerSystemList = []
            for systemId in systemIds:
                systemDf = groupedX.get_group(systemId)
                column_value = systemDf["value_0"].values
                column_score = np.zeros(len(column_value))
                for iter in range(window_size - 1, len(column_value)):
                    sequence = column_value[iter - window_size + 1:iter + 1]
                    column_score[iter] = np.sum(np.abs(sequence))
                column_score[:window_size - 1] = column_score[window_size - 1]
                OutlierScorePerSystemList.append(column_score.tolist())
            OutlierScorePerSystemList = np.asarray(OutlierScorePerSystemList)
            OutlierScorePerSystemList = (
                    OutlierScorePerSystemList == OutlierScorePerSystemList.max(axis=0)[None, :]).astype(int)

            maxOutlierScorePerSystemList = OutlierScorePerSystemList.sum(axis=1).tolist()

            ranking = np.sort(maxOutlierScorePerSystemList)
            threshold = ranking[int((1 - contamination) * len(ranking))]
            self.threshold = threshold
            mask = (maxOutlierScorePerSystemList >= threshold)
            ranking[mask] = 1
            ranking[np.logical_not(mask)] = 0
            for iter in range(len(systemIds)):
                transformed_X.append([systemIds[iter], ranking[iter]])

        if (method_type == "majority_voting_sliding_window_max"):
            """
            Sytem with most vote based on max of max of reconstruction errors in each window
            """
            OutlierScorePerSystemList = []
            for systemId in systemIds:
                systemDf = groupedX.get_group(systemId)
                column_value = systemDf["value_0"].values
                column_score = np.zeros(len(column_value))
                for iter in range(window_size - 1, len(column_value)):
                    sequence = column_value[iter - window_size + 1:iter + 1]
                    column_score[iter] = np.max(np.abs(sequence))
                column_score[:window_size - 1] = column_score[window_size - 1]
                OutlierScorePerSystemList.append(column_score.tolist())
            OutlierScorePerSystemList = np.asarray(OutlierScorePerSystemList)
            OutlierScorePerSystemList = (
                    OutlierScorePerSystemList == OutlierScorePerSystemList.max(axis=0)[None, :]).astype(int)

            maxOutlierScorePerSystemList = OutlierScorePerSystemList.sum(axis=1).tolist()

            ranking = np.sort(maxOutlierScorePerSystemList)
            threshold = ranking[int((1 - contamination) * len(ranking))]
            self.threshold = threshold
            mask = (maxOutlierScorePerSystemList >= threshold)
            ranking[mask] = 1
            ranking[np.logical_not(mask)] = 0
            for iter in range(len(systemIds)):
                transformed_X.append([systemIds[iter], ranking[iter]])

        return transformed_X


