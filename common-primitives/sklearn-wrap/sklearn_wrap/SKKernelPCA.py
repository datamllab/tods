from typing import Any, Callable, List, Dict, Union, Optional, Sequence, Tuple
from numpy import ndarray
from collections import OrderedDict
from scipy import sparse
import os
import sklearn
import numpy
import typing

# Custom import commands if any
from sklearn.decomposition.kernel_pca import KernelPCA


from d3m.container.numpy import ndarray as d3m_ndarray
from d3m.container import DataFrame as d3m_dataframe
from d3m.metadata import hyperparams, params, base as metadata_base
from d3m import utils
from d3m.base import utils as base_utils
from d3m.exceptions import PrimitiveNotFittedError
from d3m.primitive_interfaces.base import CallResult, DockerContainer
from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase


Inputs = d3m_dataframe
Outputs = d3m_dataframe


class Params(params.Params):
    lambdas_: Optional[ndarray]
    alphas_: Optional[ndarray]
    dual_coef_: Optional[ndarray]
    X_fit_: Optional[ndarray]
    _centerer: Optional[sklearn.base.BaseEstimator]
    X_transformed_fit_: Optional[ndarray]
    input_column_names: Optional[Any]
    target_names_: Optional[Sequence[Any]]
    training_indices_: Optional[Sequence[int]]
    target_column_indices_: Optional[Sequence[int]]
    target_columns_metadata_: Optional[List[OrderedDict]]



class Hyperparams(hyperparams.Hyperparams):
    n_components = hyperparams.Union(
        configuration=OrderedDict({
            'int': hyperparams.Bounded[int](
                lower=0,
                upper=None,
                default=100,
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            ),
            'none': hyperparams.Constant(
                default=None,
                description='All non-zero components are kept.',
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            )
        }),
        default='none',
        description='Number of components. If None, all non-zero components are kept.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    kernel = hyperparams.Choice(
        choices={
            'linear': hyperparams.Hyperparams.define(
                configuration=OrderedDict({})
            ),
            'poly': hyperparams.Hyperparams.define(
                configuration=OrderedDict({
                    'degree': hyperparams.Bounded[int](
                        default=3,
                        lower=0,
                        upper=None,
                        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
                    ),
                    'gamma': hyperparams.Union(
                        configuration=OrderedDict({
                            'float': hyperparams.Constant(
                                default=1.0,
                                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
                            ),
                            'none': hyperparams.Constant(
                                default=None,
                                description='Equals 1/n_features.',
                                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
                            )
                        }),
                        default='none',
                        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
                    ),
                    'coef0': hyperparams.Constant(
                        default=1,
                        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
                    )
                })
            ),
            'rbf': hyperparams.Hyperparams.define(
                configuration=OrderedDict({
                    'gamma': hyperparams.Union(
                        configuration=OrderedDict({
                            'float': hyperparams.Constant(
                                default=1.0,
                                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
                            ),
                            'none': hyperparams.Constant(
                                default=None,
                                description='Equals 1/n_features.',
                                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
                            )
                        }),
                        default='none',
                        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
                    )
                })
            ),
            'sigmoid': hyperparams.Hyperparams.define(
                configuration=OrderedDict({
                    'gamma': hyperparams.Union(
                        configuration=OrderedDict({
                            'float': hyperparams.Constant(
                                default=1.0,
                                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
                            ),
                            'none': hyperparams.Constant(
                                default=None,
                                description='Equals 1/n_features.',
                                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
                            )
                        }),
                        default='none',
                        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
                    ),
                    'coef0': hyperparams.Constant(
                        default=1,
                        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
                    )
                })
            ),
            'precomputed': hyperparams.Hyperparams.define(
                configuration=OrderedDict({})
            )
        },
        default='rbf',
        description='Kernel. Default="linear".',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    fit_inverse_transform = hyperparams.UniformBool(
        default=False,
        description='Learn the inverse transform for non-precomputed kernels. (i.e. learn to find the pre-image of a point)',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    alpha = hyperparams.Constant(
        default=1,
        description='Hyperparameter of the ridge regression that learns the inverse transform (when fit_inverse_transform=True).',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    eigen_solver = hyperparams.Enumeration[str](
        default='auto',
        values=['auto', 'dense', 'arpack'],
        description='Select eigensolver to use. If n_components is much less than the number of training samples, arpack may be more efficient than the dense eigensolver.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    tol = hyperparams.Bounded[float](
        default=0,
        lower=0,
        upper=None,
        description='Convergence tolerance for arpack. If 0, optimal value will be chosen by arpack.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    max_iter = hyperparams.Union(
        configuration=OrderedDict({
            'int': hyperparams.Bounded[int](
                lower=0,
                upper=None,
                default=4,
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            ),
            'none': hyperparams.Constant(
                default=None,
                description='Optimal value is chosen by arpack.',
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            )
        }),
        default='none',
        description='Maximum number of iterations for arpack. If None, optimal value will be chosen by arpack.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    remove_zero_eig = hyperparams.UniformBool(
        default=False,
        description='If True, then all components with zero eigenvalues are removed, so that the number of components in the output may be < n_components (and sometimes even zero due to numerical instability). When n_components is None, this parameter is ignored and components with zero eigenvalues are removed regardless.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    n_jobs = hyperparams.Union(
        configuration=OrderedDict({
            'limit': hyperparams.Bounded[int](
                default=1,
                lower=1,
                upper=None,
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            ),
            'all_cores': hyperparams.Constant(
                default=-1,
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            )
        }),
        default='limit',
        description='The number of parallel jobs to run. If `-1`, then the number of jobs is set to the number of CPU cores.  .. versionadded:: 0.18  copy_X : boolean, default=True If True, input X is copied and stored by the model in the `X_fit_` attribute. If no further changes will be done to X, setting `copy_X=False` saves memory by storing a reference.  .. versionadded:: 0.18',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ResourcesUseParameter']
    )
    
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
        values=['https://metadata.datadrivendiscovery.org/types/Attribute', 'https://metadata.datadrivendiscovery.org/types/ConstructedAttribute'],
        default='https://metadata.datadrivendiscovery.org/types/Attribute',
        description='Decides what semantic type to attach to generated attributes',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )

class SKKernelPCA(UnsupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    Primitive wrapping for sklearn KernelPCA
    `sklearn documentation <https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html>`_
    
    """
    
    __author__ = "JPL MARVIN"
    metadata = metadata_base.PrimitiveMetadata({ 
         "algorithm_types": [metadata_base.PrimitiveAlgorithmType.PRINCIPAL_COMPONENT_ANALYSIS, ],
         "name": "sklearn.decomposition.kernel_pca.KernelPCA",
         "primitive_family": metadata_base.PrimitiveFamily.FEATURE_EXTRACTION,
         "python_path": "d3m.primitives.feature_extraction.kernel_pca.SKlearn",
         "source": {'name': 'JPL', 'contact': 'mailto:shah@jpl.nasa.gov', 'uris': ['https://gitlab.com/datadrivendiscovery/sklearn-wrap/issues', 'https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html']},
         "version": "2019.11.13",
         "id": "fec6eba2-4a1b-3ea9-a31f-1da371941ede",
         "hyperparams_to_tune": ['n_components', 'kernel', 'alpha'],
         'installation': [
                        {'type': metadata_base.PrimitiveInstallationType.PIP,
                           'package_uri': 'git+https://gitlab.com/datadrivendiscovery/sklearn-wrap.git@{git_commit}#egg=sklearn_wrap'.format(
                               git_commit=utils.current_git_commit(os.path.dirname(__file__)),
                            ),
                           }]
    })

    def __init__(self, *,
                 hyperparams: Hyperparams,
                 random_seed: int = 0,
                 docker_containers: Dict[str, DockerContainer] = None) -> None:

        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)
        
        # False
        self._clf = KernelPCA(
              n_components=self.hyperparams['n_components'],
              kernel=self.hyperparams['kernel']['choice'],
              degree=self.hyperparams['kernel'].get('degree', 3),
              gamma=self.hyperparams['kernel'].get('gamma', 'none'),
              coef0=self.hyperparams['kernel'].get('coef0', 1),
              fit_inverse_transform=self.hyperparams['fit_inverse_transform'],
              alpha=self.hyperparams['alpha'],
              eigen_solver=self.hyperparams['eigen_solver'],
              tol=self.hyperparams['tol'],
              max_iter=self.hyperparams['max_iter'],
              remove_zero_eig=self.hyperparams['remove_zero_eig'],
              n_jobs=self.hyperparams['n_jobs'],
              random_state=self.random_seed,
        )
        
        self._inputs = None
        self._outputs = None
        self._training_inputs = None
        self._training_outputs = None
        self._target_names = None
        self._training_indices = None
        self._target_column_indices = None
        self._target_columns_metadata: List[OrderedDict] = None
        self._input_column_names = None
        self._fitted = False
        
        
    def set_training_data(self, *, inputs: Inputs) -> None:
        self._inputs = inputs
        self._fitted = False
        
    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        if self._fitted:
            return CallResult(None)

        self._training_inputs, self._training_indices = self._get_columns_to_fit(self._inputs, self.hyperparams)
        self._input_column_names = self._training_inputs.columns

        if self._training_inputs is None:
            return CallResult(None)

        if len(self._training_indices) > 0:
            self._clf.fit(self._training_inputs)
            self._fitted = True
        else:
            if self.hyperparams['error_on_no_input']:
                raise RuntimeError("No input columns were selected")
            self.logger.warn("No input columns were selected")
        return CallResult(None)
        
    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        if not self._fitted:
            raise PrimitiveNotFittedError("Primitive not fitted.")
        sk_inputs = inputs
        if self.hyperparams['use_semantic_types']:
            sk_inputs = inputs.iloc[:, self._training_indices]
        output_columns = []
        if len(self._training_indices) > 0:
            sk_output = self._clf.transform(sk_inputs)
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
        return CallResult(outputs)
        

    def get_params(self) -> Params:
        if not self._fitted:
            return Params(
                lambdas_=None,
                alphas_=None,
                dual_coef_=None,
                X_fit_=None,
                _centerer=None,
                X_transformed_fit_=None,
                input_column_names=self._input_column_names,
                training_indices_=self._training_indices,
                target_names_=self._target_names,
                target_column_indices_=self._target_column_indices,
                target_columns_metadata_=self._target_columns_metadata
            )

        return Params(
            lambdas_=getattr(self._clf, 'lambdas_', None),
            alphas_=getattr(self._clf, 'alphas_', None),
            dual_coef_=getattr(self._clf, 'dual_coef_', None),
            X_fit_=getattr(self._clf, 'X_fit_', None),
            _centerer=getattr(self._clf, '_centerer', None),
            X_transformed_fit_=getattr(self._clf, 'X_transformed_fit_', None),
            input_column_names=self._input_column_names,
            training_indices_=self._training_indices,
            target_names_=self._target_names,
            target_column_indices_=self._target_column_indices,
            target_columns_metadata_=self._target_columns_metadata
        )

    def set_params(self, *, params: Params) -> None:
        self._clf.lambdas_ = params['lambdas_']
        self._clf.alphas_ = params['alphas_']
        self._clf.dual_coef_ = params['dual_coef_']
        self._clf.X_fit_ = params['X_fit_']
        self._clf._centerer = params['_centerer']
        self._clf.X_transformed_fit_ = params['X_transformed_fit_']
        self._input_column_names = params['input_column_names']
        self._training_indices = params['training_indices_']
        self._target_names = params['target_names_']
        self._target_column_indices = params['target_column_indices_']
        self._target_columns_metadata = params['target_columns_metadata_']
        
        if params['lambdas_'] is not None:
            self._fitted = True
        if params['alphas_'] is not None:
            self._fitted = True
        if params['dual_coef_'] is not None:
            self._fitted = True
        if params['X_fit_'] is not None:
            self._fitted = True
        if params['_centerer'] is not None:
            self._fitted = True
        if params['X_transformed_fit_'] is not None:
            self._fitted = True



    
    
    @classmethod
    def _get_columns_to_fit(cls, inputs: Inputs, hyperparams: Hyperparams):
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
        column_metadata = inputs_metadata.query((metadata_base.ALL_ELEMENTS, column_index))

        accepted_structural_types = (int, float, numpy.integer, numpy.float64)
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
        outputs_metadata = metadata_base.DataMetadata().generate(value=outputs)

        for column_index, column_metadata in enumerate(target_columns_metadata):
            column_metadata.pop("structural_type", None)
            outputs_metadata = outputs_metadata.update_column(column_index, column_metadata)

        return outputs_metadata

    def _wrap_predictions(self, inputs: Inputs, predictions: ndarray) -> Outputs:
        outputs = d3m_dataframe(predictions, generate_metadata=True)
        target_columns_metadata = self._copy_inputs_metadata(inputs.metadata, self._training_indices, outputs.metadata, self.hyperparams)
        outputs.metadata = self._update_predictions_metadata(inputs.metadata, outputs, target_columns_metadata)
        return outputs


    @classmethod
    def _copy_inputs_metadata(cls, inputs_metadata: metadata_base.DataMetadata, input_indices: List[int],
                                        outputs_metadata: metadata_base.DataMetadata, hyperparams):
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

        return target_columns_metadata


SKKernelPCA.__doc__ = KernelPCA.__doc__