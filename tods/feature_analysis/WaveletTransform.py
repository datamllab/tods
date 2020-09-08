import os
import typing
from numpy import ndarray
import numpy as np

from d3m import container, utils as d3m_utils
from d3m.base import utils as base_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer
from typing import Union

import pywt
import pandas
import math

import common_primitives
import numpy
from typing import Optional, List
from collections import OrderedDict
from scipy import sparse
import logging
import uuid

__all__ = ('WaveletTransformer',)

Inputs = container.DataFrame
Outputs = container.DataFrame


class Hyperparams(hyperparams.Hyperparams):
    wavelet = hyperparams.Enumeration(
        values=['bior1.1', 'bior1.3', 'bior1.5', 'bior2.2', 'bior2.4', 'bior2.6', 'bior2.8',
                'bior3.1', 'bior3.3', 'bior3.5', 'bior3.7', 'bior3.9', 'bior4.4', 'bior5.5',
                'bior6.8', 'cgau1', 'cgau2', 'cgau3', 'cgau4', 'cgau5', 'cgau6', 'cgau7', 'cgau8',
                'cmor', 'coif1', 'coif2', 'coif3', 'coif4', 'coif5', 'coif6', 'coif7', 'coif8',
                'coif9', 'coif10', 'coif11', 'coif12', 'coif13', 'coif14', 'coif15', 'coif16',
                'coif17', 'db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8', 'db9', 'db10',
                'db11', 'db12', 'db13', 'db14', 'db15', 'db16', 'db17', 'db18', 'db19', 'db20',
                'db21', 'db22', 'db23', 'db24', 'db25', 'db26', 'db27', 'db28', 'db29', 'db30',
                'db31', 'db32', 'db33', 'db34', 'db35', 'db36', 'db37', 'db38', 'dmey', 'fbsp',
                'gaus1', 'gaus2', 'gaus3', 'gaus4', 'gaus5', 'gaus6', 'gaus7', 'gaus8', 'haar',
                'mexh', 'morl', 'rbio1.1', 'rbio1.3', 'rbio1.5', 'rbio2.2', 'rbio2.4', 'rbio2.6',
                'rbio2.8', 'rbio3.1', 'rbio3.3', 'rbio3.5', 'rbio3.7', 'rbio3.9', 'rbio4.4',
                'rbio5.5', 'rbio6.8', 'shan', 'sym2', 'sym3', 'sym4', 'sym5', 'sym6', 'sym7',
                'sym8', 'sym9', 'sym10', 'sym11', 'sym12', 'sym13', 'sym14', 'sym15', 'sym16',
                'sym17', 'sym18', 'sym19', 'sym20'],

        default='db8',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="Wavelet to use.",
    )
    mode = hyperparams.Enumeration(
        values=['zero', 'constant', 'symmetric', 'periodic', 'smooth', 'periodization', 'reflect',
                'antisymmetric', 'antireflect'],

        default='symmetric',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="Signal extension mode.",
    )
    axis = hyperparams.UniformInt(
        lower=0,
        upper=2,
        default=0,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="Axis over which to compute the DWT. If 0, independently DWT each feature, otherwise (if 1) DWT each sample.",
    )

    level = hyperparams.Union[Union[int, None]](
        configuration=OrderedDict(
            init=hyperparams.Hyperparameter[int](
                default=0,
            ),
            ninit=hyperparams.Hyperparameter[None](
                default=None,
            ),
        ),
        default='ninit',
        description="Decomposition level (must be >= 0). If level is None (default) then it will be calculated using the dwt_max_level function.",
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    )

    # level = hyperparams.Hyperparameter[None](
    #     default=None,
    #     semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    #     description="Decomposition level (must be >= 0). If level is None (default) then it will be calculated using the dwt_max_level function.",
    # )

    inverse = hyperparams.UniformInt(
        lower=0,
        upper=2,
        default=0,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Inverse wavelet transformation if inverse=1.",
    )
    id = hyperparams.Hyperparameter[str](
        default='0000',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="identification number.",
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


class WaveletTransformer(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    A primitive of Multilevel 1D Discrete Wavelet Transform of data.
    See `PyWavelet documentation <https://pywavelets.readthedocs.io/en/latest/ref/>`_ for details.
    Parameters
        ----------
        wavelet: str
            Wavelet to use

        mode: str
            Signal extension mode, see https://pywavelets.readthedocs.io/en/latest/ref/signal-extension-modes.html#ref-modes for details.

        axis: int
            Axis over which to compute the DWT. If not given, transforming along columns.

        window_size : int
            The moving window size.

        level: int
            Decomposition level (must be > 0). If level is 0 (default) then it will be calculated using the maximum level.

        Attributes
        ----------
        None
    """

    __author__ = "DATALAB @Taxes A&M University"
    metadata = metadata_base.PrimitiveMetadata(
        {
            "name": "Wavelet_transformation",
            "python_path": "d3m.primitives.tods.feature_analysis.wavelet_transform",
            "source": {'name': "DATALAB @Taxes A&M University", 'contact': 'mailto:khlai037@tamu.edu',
                       'uris': ['https://gitlab.com/lhenry15/tods.git']},
            "algorithm_types": [metadata_base.PrimitiveAlgorithmType.FREQUENCY_TRANSFORM, ],
            "primitive_family": metadata_base.PrimitiveFamily.FEATURE_EXTRACTION,
            "version": "0.0.1",
            "hyperparams_to_tune": ['wavelet', 'mode', 'axis', 'level'],
            "id": str(uuid.uuid3(uuid.NAMESPACE_DNS, 'WaveletTransformer')),
        },
    )

    def __init__(self, *, hyperparams: Hyperparams) -> None:
        super().__init__(hyperparams=hyperparams) # , random_seed=random_seed, docker_containers=docker_containers)

        # False
        self._clf = Wavelet(wavelet=self.hyperparams['wavelet'],
                            mode=self.hyperparams['mode'],
                            axis=self.hyperparams['axis'],
                            level=self.hyperparams['level'],
                            # id=self.hyperparams['id'],
                          )


    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        """
        Process the testing data.
        Args:
            inputs: Container DataFrame. Time series data up to Wavelet transform.

        Returns:
            [cA_n, cD_n, cD_n-1, …, cD2, cD1]: Container DataFrame after Wavelet Transformation.
            Ordered frame of coefficients arrays where n denotes the level of decomposition. The first element (cA_n) of the result is approximation coefficients array and the following elements (cD_n - cD_1) are details coefficients arrays.
        """
        assert isinstance(inputs, container.DataFrame), type(container.DataFrame)

        _, self._columns_to_produce = self._get_columns_to_fit(inputs, self.hyperparams)
        self._input_column_names = inputs.columns

        # print('columns_to_produce=', self._columns_to_produce)


        sk_inputs = inputs
        if self.hyperparams['use_semantic_types']:
            sk_inputs = inputs.iloc[:, self._columns_to_produce]
        output_columns = []
        if len(self._columns_to_produce) > 0:
            sk_output = self._clf.produce(sk_inputs, self.hyperparams['inverse'])
            if sparse.issparse(sk_output):
                sk_output = sk_output.toarray()
            outputs = self._wrap_predictions(inputs, sk_output)
            if len(outputs.columns) == len(self._input_column_names):
                outputs.columns = self._input_column_names
            output_columns = [outputs]
        else:
            if self.hyperparams['error_on_no_input']:
                raise RuntimeError("No input columns were selected")
            self.logger.warn("No input columns were selected")
        outputs = base_utils.combine_columns(return_result=self.hyperparams['return_result'],
                                             add_index_columns=self.hyperparams['add_index_columns'],
                                             inputs=inputs, column_indices=self._columns_to_produce,
                                             columns_list=output_columns)

        # print(inputs)
        # print(outputs)
        # if self.hyperparams['inverse'] == 1:
        #     print(outputs)
        # print(outputs.metadata.to_internal_simple_structure())

        # outputs = inputs
        return base.CallResult(outputs)

        # return base.CallResult(dataframe)

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
        # print('======_get_columns_to_fit======')

        if not hyperparams['use_semantic_types']:
            return inputs, list(range(len(inputs.columns)))

        inputs_metadata = inputs.metadata

        def can_produce_column(column_index: int) -> bool:
            return cls._can_produce_column(inputs_metadata, column_index, hyperparams)

        columns_to_produce, columns_not_to_produce = base_utils.get_columns_to_use(inputs_metadata,
                                                                                   use_columns=hyperparams[
                                                                                       'use_columns'],
                                                                                   exclude_columns=hyperparams[
                                                                                       'exclude_columns'],
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

        # print(column_metadata)
        # print(column_metadata['structural_type'], accepted_structural_types)

        if not issubclass(column_metadata['structural_type'], accepted_structural_types):
            return False

        semantic_types = set(column_metadata.get('semantic_types', []))

        # print(column_metadata)
        # print(semantic_types, accepted_semantic_types)

        if len(semantic_types) == 0:
            cls.logger.warning("No semantic types found in column metadata")
            return False

        # Making sure all accepted_semantic_types are available in semantic_types
        if len(accepted_semantic_types - semantic_types) == 0:
            return True

        return False

    @classmethod
    def _get_target_columns_metadata(cls, outputs_metadata: metadata_base.DataMetadata, hyperparams) -> List[
        OrderedDict]:
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
        target_columns_metadata = self._copy_inputs_metadata(inputs.metadata, self._columns_to_produce, outputs.metadata,
                                                             self.hyperparams)
        outputs.metadata = self._update_predictions_metadata(inputs.metadata, outputs, target_columns_metadata)
        return outputs

    @classmethod
    def _copy_inputs_metadata(cls, inputs_metadata: metadata_base.DataMetadata, input_indices: List[int],
                              outputs_metadata: metadata_base.DataMetadata, hyperparams):
        """
        Updata metadata for selected columns.
        Args:
            inputs_metadata: metadata.base.DataMetadata
            input_indices: list
            outputs_metadata: metadata.base.DataMetadata
            hyperparams: d3m.metadata.hyperparams.Hyperparams

        Returns:
            d3m.metadata.base.DataMetadata
        """
        outputs_length = outputs_metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length']
        target_columns_metadata: List[OrderedDict] = []
        for column_index in input_indices:
            column_name = inputs_metadata.query((metadata_base.ALL_ELEMENTS, column_index)).get("name")
            if column_name is None:
                column_name = "output_{}".format(column_index)

            column_metadata = OrderedDict(inputs_metadata.query_column(column_index))
            semantic_types = set(column_metadata.get('semantic_types', []))
            semantic_types_to_remove = set([])
            add_semantic_types = set()
            add_semantic_types.add(hyperparams["return_semantic_type"])
            semantic_types = semantic_types - semantic_types_to_remove
            semantic_types = semantic_types.union(add_semantic_types)
            column_metadata['semantic_types'] = list(semantic_types)
            # print(column_metadata['semantic_types'])

            column_metadata["name"] = str(column_name)
            target_columns_metadata.append(column_metadata)

        #  If outputs has more columns than index, add Attribute Type to all remaining
        if outputs_length > len(input_indices):
            for column_index in range(len(input_indices), outputs_length):
                column_metadata = OrderedDict()
                semantic_types = set()
                semantic_types.add(hyperparams["return_semantic_type"])
                column_name = "output_{}".format(column_index)
                column_metadata["semantic_types"] = list(semantic_types)
                column_metadata["name"] = str(column_name)
                target_columns_metadata.append(column_metadata)

        # print(target_columns_metadata)
        return target_columns_metadata

WaveletTransformer.__doc__ = WaveletTransformer.__doc__

class Wavelet:

    wt_info = dict()

    def __init__(self, wavelet='db1', mode='symmetric', axis=-1, level=1, id=0):
        self._wavelet = wavelet
        self._mode = mode
        self._axis = axis
        self._level = level
        self._id = id
        return

    def produce(self, data, inverse):

        if inverse == 1:
            output = self.inverse_transform_to_dataframe(coeffs=data)

        else:
            output = self.transform_to_dataframe(data)

        return output


    def transform_to_dataframe(self, data):

        # print(data)
        coeffs_buf = pandas.DataFrame(columns=[])


        for index, data_to_transform in data.iteritems():
            # data_to_transform = data.squeeze(1)
            # print(data_to_transform)
            if self._level == None:
                wavelet_dec_len = pywt.Wavelet(self._wavelet).dec_len
                self._level = pywt.dwt_max_level(len(data_to_transform), wavelet_dec_len)

            coeffs = pywt.wavedec(data=data_to_transform, wavelet=self._wavelet, level=self._level)
            coeffs_T = pandas.DataFrame(coeffs).T
            coeffs_buf = pandas.concat([coeffs_buf, coeffs_T], axis=1)
            # coeffs_T = ndarray(coeffs).T
            # print(coeffs_T)

        # print(coeffs_buf)

        return coeffs_buf # coeffs_T

    def transform_to_single_dataframe(self, data):

        # print(data)
        data_to_transform = data.squeeze(1)
        wavelet_dec_len = pywt.Wavelet(self._wavelet).dec_len
        self._level = pywt.dwt_max_level(len(data_to_transform), wavelet_dec_len)

        coeffs = pywt.wavedec(data=data_to_transform, wavelet=self._wavelet, level=self._level)

        cAD_size = [len(cAD) for cAD in coeffs]
        Wavelet.wt_info[self._id] = {'wavelet': self._wavelet,
                                     'cAD_size': cAD_size,
                                     }

        # print(len(data_to_transform))
        #
        coeffs_list = [] # ndarray([0])
        for cAD in coeffs:
            # print(cAD.shape)
            # print(cAD[0:10])
            coeffs_list += list(cAD)

        # print(len(coeffs_list))

        coeffs_T = pandas.DataFrame(coeffs_list)
        # print(coeffs_T)

        return coeffs_T

    def inverse_transform_to_dataframe(self, coeffs):
        # print('=======inverse_transform======')
        # print('level: ', self._level)
        # print(coeffs)

        coeffs_list = [numpy.array(col[~pandas.isnull(col)]) for index, col in coeffs.iteritems()]
        # print(coeffs_list)
        data = pywt.waverec(coeffs=coeffs_list, wavelet=self._wavelet)

        # print(data)
        return data # [0:-1]

    def inverse_transform_to_single_dataframe(self, coeffs):
        # print('=======inverse_transform======')
        # print('level: ', self._level)
        # print(coeffs)
        # print(Wavelet.wt_info[self._id])
        wt_info = Wavelet.wt_info[self._id]
        # print(wt_info)
        # print(wt_info['cAD_size'])
        # print(wt_info['wavelet'])
        cAD_size = wt_info['cAD_size']
        self._wavelet = wt_info['wavelet']

        coeffs_format = []
        coeff = coeffs
        for cAD_len in cAD_size:
            coeffs_format.append(np.array(coeff[0:cAD_len]).squeeze(axis=1))
            coeff = coeff[cAD_len:]

        # for cAD in coeffs_format:
        #     print(cAD.shape)
        #     print(cAD[0:10])

        # print(coeffs_format)
        data = pywt.waverec(coeffs=coeffs_format, wavelet=self._wavelet)

        # print(data.shape)
        # print(data)
        return data # [0:-1]
