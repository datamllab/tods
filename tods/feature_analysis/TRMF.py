from typing import Any, Callable, List, Dict, Union, Optional, Sequence, Tuple
from numpy import ndarray
from collections import OrderedDict
from scipy import sparse
import os
import sklearn
import numpy as np
import typing
import time

# Custom import commands if any
from sklearn.decomposition.truncated_svd import TruncatedSVD


from d3m.container.numpy import ndarray as d3m_ndarray
from d3m.container import DataFrame as d3m_dataframe
from d3m.metadata import hyperparams, params, base as metadata_base
from d3m import utils
from d3m.base import utils as base_utils
from d3m.exceptions import PrimitiveNotFittedError
from d3m.primitive_interfaces.base import CallResult, DockerContainer
from d3m.primitive_interfaces import base, transformer
# from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase


Inputs = d3m_dataframe
Outputs = d3m_dataframe

__all__ = ('TRMF',)

# class Params(params.Params):
#     components_: Optional[ndarray]
#     explained_variance_ratio_: Optional[ndarray]
#     explained_variance_: Optional[ndarray]
#     singular_values_: Optional[ndarray]
#     input_column_names: Optional[Any]
#     target_names_: Optional[Sequence[Any]]
#     training_indices_: Optional[Sequence[int]]
#     target_column_indices_: Optional[Sequence[int]]
#     target_columns_metadata_: Optional[List[OrderedDict]]


class Hyperparams(hyperparams.Hyperparams):
    # Tuning
    lags = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(1,),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Set of lag indices to use in model.",
    )
    K = hyperparams.UniformInt(
        lower=0,
        upper=100000000,
        default=2,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="Length of latent embedding dimension.",
    )
    lambda_f = hyperparams.Uniform(
        lower=0,
        upper=100000000,
        default=1.0,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="Regularization parameter used for matrix F.",
    )
    lambda_x = hyperparams.Uniform(
        lower=0,
        upper=100000000,
        default=1.0,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="Regularization parameter used for matrix X.",
    )
    lambda_w = hyperparams.Uniform(
        lower=0,
        upper=100000000,
        default=1.0,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="Regularization parameter used for matrix W.",
    )
    alpha = hyperparams.Uniform(
        lower=0,
        upper=100000000,
        default=1000.0,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="Regularization parameter used for make the sum of lag coefficient close to 1. That helps to avoid big deviations when forecasting.",
    )
    eta = hyperparams.Uniform(
        lower=0,
        upper=100000000,
        default=1.0,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="Regularization parameter used for X when undercovering autoregressive dependencies.",
    )
    max_iter = hyperparams.UniformInt(
        lower=0,
        upper=100000000,
        default=1000,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="Number of iterations of updating matrices F, X and W.",
    )
    F_step = hyperparams.Uniform(
        lower=0,
        upper=100000000,
        default=0.0001,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="Step of gradient descent when updating matrix F.",
    )
    X_step = hyperparams.Uniform(
        lower=0,
        upper=100000000,
        default=0.0001,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="Step of gradient descent when updating matrix X.",
    )
    W_step = hyperparams.Uniform(
        lower=0,
        upper=100000000,
        default=0.0001,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="Step of gradient descent when updating matrix W.",
    )

    # Control
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
        values=['https://metadata.datadrivendiscovery.org/types/Attribute', 'https://metadata.datadrivendiscovery.org/types/ConstructedAttribute'],
        default='https://metadata.datadrivendiscovery.org/types/Attribute',
        description='Decides what semantic type to attach to generated attributes',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )

class TRMF(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """Temporal Regularized Matrix Factorization.

    Parameters
    ---------- 

    lags : array-like, shape (n_lags,)
        Set of lag indices to use in model.
    
    K : int
        Length of latent embedding dimension
    
    lambda_f : float
        Regularization parameter used for matrix F.
    
    lambda_x : float
        Regularization parameter used for matrix X.
    
    lambda_w : float
        Regularization parameter used for matrix W.

    alpha : float
        Regularization parameter used for make the sum of lag coefficient close to 1.
        That helps to avoid big deviations when forecasting.
    
    eta : float
        Regularization parameter used for X when undercovering autoregressive dependencies.

    max_iter : int
        Number of iterations of updating matrices F, X and W.

    F_step : float
        Step of gradient descent when updating matrix F.

    X_step : float
        Step of gradient descent when updating matrix X.

    W_step : float
        Step of gradient descent when updating matrix W.


    Attributes
    ----------

    F : ndarray, shape (n_timeseries, K)
        Latent embedding of timeseries.

    X : ndarray, shape (K, n_timepoints)
        Latent embedding of timepoints.

    W : ndarray, shape (K, n_lags)
        Matrix of autoregressive coefficients.

    Reference
    ----------
    "https://github.com/SemenovAlex/trmf"

    Yu, H. F., Rao, N., & Dhillon, I. S. (2016). Temporal regularized matrix factorization for high-dimensional time series prediction.
    In Advances in neural information processing systems (pp. 847-855).
    Which can be found there: http://www.cs.utexas.edu/~rofuyu/papers/tr-mf-nips.pdf
    """

    __author__: "DATA Lab at Texas A&M University"
    metadata = metadata_base.PrimitiveMetadata({
         "name": "Temporal Regularized Matrix Factorization Primitive",
         "python_path": "d3m.primitives.tods.feature_analysis.trmf",
         "source": {'name': 'DATA Lab at Texas A&M University', 'contact': 'mailto:khlai037@tamu.edu', 
         'uris': ['https://gitlab.com/lhenry15/tods.git', 'https://gitlab.com/lhenry15/tods/-/blob/Junjie/anomaly-primitives/anomaly_primitives/TRMF.py']},
         "algorithm_types": [metadata_base.PrimitiveAlgorithmType.TEMPORAL_REGULARIZED_MATRIX_FACTORIZATION, ],
         "primitive_family": metadata_base.PrimitiveFamily.FEATURE_CONSTRUCTION,
         "id": "d6be6941-61d0-4cbd-85ef-a10c86aa40b1",
         "hyperparams_to_tune": ['lags', 'K', 'lambda_f', 'lambda_x', 'lambda_w', 'alpha', 'eta', 'max_iter', 'F_step', 'X_step', 'W_step'],
         "version": "0.0.1",
    })

        
    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Process the testing data.
        Args:
            inputs: Container DataFrame.

        Returns:
            Container DataFrame after Truncated SVD.
        """
        self._clf = trmf(
            lags=self.hyperparams['lags'],
            K=self.hyperparams['K'],
            lambda_f=self.hyperparams['lambda_f'],
            lambda_x=self.hyperparams['lambda_x'],
            lambda_w=self.hyperparams['lambda_w'],
            alpha=self.hyperparams['alpha'],
            eta=self.hyperparams['eta'],
            max_iter=self.hyperparams['max_iter'],
            F_step=self.hyperparams['F_step'],
            X_step=self.hyperparams['X_step'],
            W_step=self.hyperparams['W_step'],            
        ) 


        tmp = inputs.copy()
        for col in inputs.columns:
            tmp[col] = inputs[col]/inputs[col].max()

        self._inputs = tmp
        self._fitted = False


        # Get cols to fit.
        self._training_inputs, self._training_indices = self._get_columns_to_fit(self._inputs, self.hyperparams)
        self._input_column_names = self._training_inputs.columns


        if len(self._training_indices) > 0:
            self._clf.fit(self._training_inputs)
            self._fitted = True
        else:
            if self.hyperparams['error_on_no_input']:
                raise RuntimeError("No input columns were selected")
            self.logger.warn("No input columns were selected")



        if not self._fitted:
            raise PrimitiveNotFittedError("Primitive not fitted.")

        sk_inputs = inputs
        if self.hyperparams['use_semantic_types']:
            sk_inputs = inputs.iloc[:, self._training_indices]
        output_columns = []
        if len(self._training_indices) > 0:

            sk_output = self._clf.get_X()


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
                                               inputs=inputs, column_indices=self._training_indices,
                                               columns_list=output_columns)

        # self._write(outputs)
        return CallResult(outputs)


    
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

        columns_to_produce, columns_not_to_produce = base_utils.get_columns_to_use(inputs_metadata,
                                                                             use_columns=hyperparams['use_columns'],
                                                                             exclude_columns=hyperparams['exclude_columns'],
                                                                             can_use_column=can_produce_column)
        return inputs.iloc[:, columns_to_produce], columns_to_produce
        # return columns_to_produce

    @classmethod
    def _can_produce_column(cls, inputs_metadata: metadata_base.DataMetadata, column_index: int, hyperparams: Hyperparams) -> bool:
        """
        Output whether a column can be processed.
        Args:
            inputs_metadata: d3m.metadata.base.DataMetadata
            column_index: int

        Returns:
            bool
        """
        column_metadata = inputs_metadata.query((metadata_base.ALL_ELEMENTS, column_index))

        accepted_structural_types = (int, float, np.integer, np.float64)
        accepted_semantic_types = set()
        accepted_semantic_types.add("https://metadata.datadrivendiscovery.org/types/Attribute")
        if not issubclass(column_metadata['structural_type'], accepted_structural_types):
            return False

        semantic_types = set(column_metadata.get('semantic_types', []))

        if len(semantic_types) == 0:
            cls.logger.warning("No semantic types found in column metadata")
            return False
        
        # Making sure all accepted_semantic_types are available in semantic_types
        if len(accepted_semantic_types - semantic_types) == 0:
            return True

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
            column_name = "output_{}".format(column_index)
            column_metadata = OrderedDict()
            semantic_types = set()
            semantic_types.add(hyperparams["return_semantic_type"])
            column_metadata['semantic_types'] = list(semantic_types)

            column_metadata["name"] = str(column_name)
            target_columns_metadata.append(column_metadata)

        return target_columns_metadata

    def _write(self, inputs:Inputs):
        """
        write inputs to current directory, only for test
        """
        inputs.to_csv(str(time.time())+'.csv')


"""
Temporal Regularized Matrix Factorization
"""
class trmf:

    # Added by JJ
    def get_X(self):
        return self.X.T


    # Original
    def __init__(self, lags, K, lambda_f, lambda_x, lambda_w, alpha, eta, max_iter=1000, 
                 F_step=0.0001, X_step=0.0001, W_step=0.0001):
        self.lags = lags
        self.L = len(lags)
        self.K = K
        self.lambda_f = lambda_f
        self.lambda_x = lambda_x
        self.lambda_w = lambda_w
        self.alpha = alpha
        self.eta = eta
        self.max_iter = max_iter
        self.F_step = F_step
        self.X_step = X_step
        self.W_step = W_step
        
        self.W = None
        self.F = None
        self.X = None


    def fit(self, train, resume=False):
        """Fit the TRMF model according to the given training data.

        Model fits through sequential updating three matrices:
            -   matrix self.F;
            -   matrix self.X;
            -   matrix self.W.
            
        Each matrix updated with gradient descent.

        Parameters
        ----------
        train : ndarray, shape (n_timeseries, n_timepoints)
            Training data.

        resume : bool
            Used to continue fitting.

        Returns
        -------
        self : object
            Returns self.
        """

        if not resume:
            self.Y = train.T
            mask = np.array((~np.isnan(self.Y)).astype(int))
            self.mask = mask
            self.Y[self.mask == 0] = 0.
            self.N, self.T = self.Y.shape
            self.W = np.random.randn(self.K, self.L) / self.L
            self.F = np.random.randn(self.N, self.K)
            self.X = np.random.randn(self.K, self.T)

        for _ in range(self.max_iter):
            self._update_F(step=self.F_step)
            self._update_X(step=self.X_step)
            self._update_W(step=self.W_step)


    def predict(self, h):
        """Predict each of timeseries h timepoints ahead.

        Model evaluates matrix X with the help of matrix W,
        then it evaluates prediction by multiplying it by F.

        Parameters
        ----------
        h : int
            Number of timepoints to forecast.

        Returns
        -------
        preds : ndarray, shape (n_timeseries, T)
            Predictions.
        """

        X_preds = self._predict_X(h)
        return np.dot(self.F, X_preds)


    def _predict_X(self, h):
        """Predict X h timepoints ahead.

        Evaluates matrix X with the help of matrix W.

        Parameters
        ----------
        h : int
            Number of timepoints to forecast.

        Returns
        -------
        X_preds : ndarray, shape (self.K, h)
            Predictions of timepoints latent embeddings.
        """

        X_preds = np.zeros((self.K, h))
        X_adjusted = np.hstack([self.X, X_preds])
        for t in range(self.T, self.T + h):
            for l in range(self.L):
                lag = self.lags[l]
                X_adjusted[:, t] += X_adjusted[:, t - lag] * self.W[:, l]
        return X_adjusted[:, self.T:]

    def impute_missings(self):
        """Impute each missing element in timeseries.

        Model uses matrix X and F to get all missing elements.

        Parameters
        ----------

        Returns
        -------
        data : ndarray, shape (n_timeseries, T)
            Predictions.
        """
        data = self.Y
        data[self.mask == 0] = np.dot(self.F, self.X)[self.mask == 0]
        return data


    def _update_F(self, step, n_iter=1):
        """Gradient descent of matrix F.

        n_iter steps of gradient descent of matrix F.

        Parameters
        ----------
        step : float
            Step of gradient descent when updating matrix.

        n_iter : int
            Number of gradient steps to be made.

        Returns
        -------
        self : objects
            Returns self.
        """

        for _ in range(n_iter):
            self.F -= step * self._grad_F()


    def _update_X(self, step, n_iter=1):
        """Gradient descent of matrix X.

        n_iter steps of gradient descent of matrix X.

        Parameters
        ----------
        step : float
            Step of gradient descent when updating matrix.

        n_iter : int
            Number of gradient steps to be made.

        Returns
        -------
        self : objects
            Returns self.
        """

        for _ in range(n_iter):
            self.X -= step * self._grad_X()


    def _update_W(self, step, n_iter=1):
        """Gradient descent of matrix W.

        n_iter steps of gradient descent of matrix W.

        Parameters
        ----------
        step : float
            Step of gradient descent when updating matrix.

        n_iter : int
            Number of gradient steps to be made.

        Returns
        -------
        self : objects
            Returns self.
        """

        for _ in range(n_iter):
            self.W -= step * self._grad_W()


    def _grad_F(self):
        """Gradient of matrix F.

        Evaluating gradient of matrix F.

        Parameters
        ----------

        Returns
        -------
        self : objects
            Returns self.
        """

        return - 2 * np.dot((self.Y - np.dot(self.F, self.X)) * self.mask, self.X.T) + 2 * self.lambda_f * self.F


    def _grad_X(self):
        """Gradient of matrix X.

        Evaluating gradient of matrix X.

        Parameters
        ----------

        Returns
        -------
        self : objects
            Returns self.
        """

        for l in range(self.L):
            lag = self.lags[l]
            W_l = self.W[:, l].repeat(self.T, axis=0).reshape(self.K, self.T)
            X_l = self.X * W_l
            z_1 = self.X - np.roll(X_l, lag, axis=1)
            z_1[:, :max(self.lags)] = 0.
            z_2 = - (np.roll(self.X, -lag, axis=1) - X_l) * W_l
            z_2[:, -lag:] = 0.

        grad_T_x = z_1 + z_2
        return - 2 * np.dot(self.F.T, self.mask * (self.Y - np.dot(self.F, self.X))) + self.lambda_x * grad_T_x + self.eta * self.X


    def _grad_W(self):
        """Gradient of matrix W.

        Evaluating gradient of matrix W.

        Parameters
        ----------

        Returns
        -------
        self : objects
            Returns self.
        """

        grad = np.zeros((self.K, self.L))
        for l in range(self.L):
            lag = self.lags[l]
            W_l = self.W[:, l].repeat(self.T, axis=0).reshape(self.K, self.T)
            X_l = self.X * W_l
            z_1 = self.X - np.roll(X_l, lag, axis=1)
            z_1[:, :max(self.lags)] = 0.
            z_2 = - (z_1 * np.roll(self.X, lag, axis=1)).sum(axis=1)
            grad[:, l] = z_2
        return grad + self.W * 2 * self.lambda_w / self.lambda_x -\
               self.alpha * 2 * (1 - self.W.sum(axis=1)).repeat(self.L).reshape(self.W.shape)
