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
from d3m.exceptions import PrimitiveNotFittedError

__all__ = ('SpectralResidualTransformPrimitive',)

Inputs = container.DataFrame
Outputs = container.DataFrame

class Params(params.Params):
       #to-do : how to make params dynamic
       use_column_names: Optional[Any]



class Hyperparams(hyperparams.Hyperparams):

       #Tuning Parameter

       avg_filter_dimension = hyperparams.Hyperparameter(default=3, semantic_types=[
           'https://metadata.datadrivendiscovery.org/types/TuningParameter',
       ], description="Spectral Residual average filter dimension")
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
           default='append',
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



class SpectralResidualTransformPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    Primitive to find Spectral Residual Transform of time series
    """
    __author__ = "DATA Lab at Texas A&M University",
    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '88dda04b-090b-49a5-8035-279eb3be9cd9',
            'version': '0.1.0',
            'name': 'Time Series Spectral Residual',
            'python_path': 'd3m.primitives.tods.feature_analysis.spectral_residual_transform',
            'keywords': ['Time Series','FFT'],
            "hyperparams_to_tune": ['avg_filter_dimension'],
            'source': {
                'name': 'DATA Lab at Texas A&M University',
                'uris': ['https://gitlab.com/lhenry15/tods.git','https://gitlab.com/lhenry15/tods/-/blob/devesh/tods/feature_analysis/SpectralResidualTransform.py'],
                'contact': 'mailto:khlai037@tamu.edu'

            },
            'installation': [
                {'type': metadata_base.PrimitiveInstallationType.PIP,
                 'package_uri': 'git+https://gitlab.com/lhenry15/tods.git@{git_commit}#egg=TODS'.format(
                     git_commit=d3m_utils.current_git_commit(os.path.dirname(__file__)),
                 ),
                 }

            ],
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.DATA_PROFILING,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.FEATURE_CONSTRUCTION,

        }
    )

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        """

        Args:
            inputs: Container DataFrame
            timeout: Default
            iterations: Default

        Returns:
            Container DataFrame containing Spectral Residual Transform of  time series
        """
        self.logger.info('Spectral Residual  Primitive called')

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
        spectral_residual_input = inputs
        if self.hyperparams['use_semantic_types']:
            spectral_residual_input = inputs.iloc[:, self._training_indices]
        output_columns = []
        if len(self._training_indices) > 0:
            spectral_residual_output = self._spectral_residual_transform(spectral_residual_input,self.hyperparams["avg_filter_dimension"])

            if sparse.issparse(spectral_residual_output):
                spectral_residual_output = spectral_residual_output.toarray()
            outputs = self._wrap_predictions(inputs, spectral_residual_output)

            #if len(outputs.columns) == len(self._input_column_names):
               # outputs.columns = self._input_column_names

            output_columns = [outputs]


        else:
            if self.hyperparams['error_on_no_input']:
                raise RuntimeError("No input columns were selected")
            self.logger.warn("No input columns were selected")
        outputs = base_utils.combine_columns(return_result=self.hyperparams['return_result'],
                                             add_index_columns=self.hyperparams['add_index_columns'],
                                             inputs=inputs, column_indices=self._training_indices,
                                             columns_list=output_columns)

        self.logger.info('Spectral Residual  Primitive returned')

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
        target_columns_metadata = self._add_target_columns_metadata(outputs.metadata, self.hyperparams)
        outputs.metadata = self._update_predictions_metadata(inputs.metadata, outputs, target_columns_metadata)

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

    def _spectral_residual_transform(self, X,avg_filter_dimension):
        """
        This method transform a time series into spectral residual series
        :param values: list.
            a list of float values.
        :return: mag: list.
            a list of float values as the spectral residual values
        """
        EPS = 1e-8
        transformed_X = utils.pandas.DataFrame()
        for column in X.columns:
            values = X[column].values
            trans = np.fft.fft(values)
            mag = np.sqrt(trans.real ** 2 + trans.imag ** 2)
            eps_index = np.where(mag <= EPS)[0]
            mag[eps_index] = EPS

            mag_log = np.log(mag)
            mag_log[eps_index] = 0
            spectral = np.exp(mag_log - self._average_filter(mag_log, n=avg_filter_dimension))

            trans.real = trans.real * spectral / mag
            trans.imag = trans.imag * spectral / mag
            trans.real[eps_index] = 0
            trans.imag[eps_index] = 0

            wave_r = np.fft.ifft(trans)
            mag = np.round(np.sqrt(wave_r.real ** 2 + wave_r.imag ** 2),4)
            transformed_X[column + "_spectral_residual"] = mag

        return transformed_X

    def _average_filter(self,values, n=3):
        """
        Calculate the sliding window average for the give time series.
        Mathematically, res[i] = sum_{j=i-t+1}^{i} values[j] / t, where t = min(n, i+1)
        :param values: list.
            a list of float numbers
        :param n: int, default 3.
            window size.
        :return res: list.
            a list of value after the average_filter process.
        """

        if n >= len(values):
            n = len(values)

        res = np.cumsum(values, dtype=float)
        res[n:] = res[n:] - res[:-n]
        res[n:] = res[n:] / n

        for i in range(1, n):
            res[i] /= (i + 1)

        return res

    def _write(self, inputs: Inputs):
        inputs.to_csv(str(time.time()) + '.csv')


