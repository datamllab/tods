from typing import Any, Callable, List, Dict, Union, Optional, Sequence, Tuple
from numpy import ndarray
from collections import OrderedDict
from scipy import sparse
import os
import sklearn
import numpy
import typing

# Custom import commands if any
from sklearn.feature_extraction.text import TfidfVectorizer


from d3m.container.numpy import ndarray as d3m_ndarray
from d3m.container import DataFrame as d3m_dataframe
from d3m.metadata import hyperparams, params, base as metadata_base
from d3m import utils
from d3m.base import utils as base_utils
from d3m.exceptions import PrimitiveNotFittedError
from d3m.primitive_interfaces.base import CallResult, DockerContainer
from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase
from d3m.metadata.base import ALL_ELEMENTS


Inputs = d3m_dataframe
Outputs = d3m_dataframe


class Params(params.Params):
    vocabulary_: Optional[Sequence[dict]]
    stop_words_: Optional[Sequence[set]]
    _tfidf: Optional[Sequence[object]]
    fixed_vocabulary_: Optional[Sequence[bool]]
    _stop_words_id: Optional[Sequence[int]]
    target_names_: Optional[Sequence[Any]]
    training_indices_: Optional[Sequence[int]]


class Hyperparams(hyperparams.Hyperparams):
    strip_accents = hyperparams.Union(
        configuration=OrderedDict({
            'accents': hyperparams.Enumeration[str](
                default='ascii',
                values=['ascii', 'unicode'],
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            ),
            'none': hyperparams.Constant(
                default=None,
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            )
        }),
        default='none',
        description='Remove accents during the preprocessing step. \'ascii\' is a fast method that only works on characters that have an direct ASCII mapping. \'unicode\' is a slightly slower method that works on any characters. None (default) does nothing.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    analyzer = hyperparams.Enumeration[str](
        default='word',
        values=['word', 'char', 'char_wb'],
        description='Whether the feature should be made of word or character n-grams.  If a callable is passed it is used to extract the sequence of features out of the raw, unprocessed input.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    ngram_range = hyperparams.SortedList(
        elements=hyperparams.Bounded[int](1, None, 1),
        default=(1, 1),
        min_size=2,
        max_size=2,
        description='The lower and upper boundary of the range of n-values for different n-grams to be extracted. All values of n such that min_n <= n <= max_n will be used.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    stop_words = hyperparams.Union(
        configuration=OrderedDict({
            'string': hyperparams.Hyperparameter[str](
                default='english',
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            ),
            'list': hyperparams.List(
                elements=hyperparams.Hyperparameter[str](''),
                default=[],
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            ),
            'none': hyperparams.Constant(
                default=None,
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            )
        }),
        default='none',
        description='If a string, it is passed to _check_stop_list and the appropriate stop list is returned. \'english\' is currently the only supported string value.  If a list, that list is assumed to contain stop words, all of which will be removed from the resulting tokens. Only applies if ``analyzer == \'word\'``.  If None, no stop words will be used. max_df can be set to a value in the range [0.7, 1.0) to automatically detect and filter stop words based on intra corpus document frequency of terms.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    lowercase = hyperparams.UniformBool(
        default=True,
        description='Convert all characters to lowercase before tokenizing.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    token_pattern = hyperparams.Hyperparameter[str](
        default='(?u)\\b\w\w+\\b',
        description='Regular expression denoting what constitutes a "token", only used if ``analyzer == \'word\'``. The default regexp selects tokens of 2 or more alphanumeric characters (punctuation is completely ignored and always treated as a token separator).',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    max_df = hyperparams.Union(
        configuration=OrderedDict({
            'proportion': hyperparams.Bounded[float](
                default=1.0,
                lower=0.0,
                upper=1.0,
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            ),
            'absolute': hyperparams.Bounded[int](
                default=1,
                lower=0,
                upper=None,
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            )
        }),
        default='proportion',
        description='When building the vocabulary ignore terms that have a document frequency strictly higher than the given threshold (corpus-specific stop words). If float, the parameter represents a proportion of documents, integer absolute counts. This parameter is ignored if vocabulary is not None.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    min_df = hyperparams.Union(
        configuration=OrderedDict({
            'proportion': hyperparams.Bounded[float](
                default=1.0,
                lower=0.0,
                upper=1.0,
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            ),
            'absolute': hyperparams.Bounded[int](
                default=1,
                lower=0,
                upper=None,
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            )
        }),
        default='absolute',
        description='When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold. This value is also called cut-off in the literature. If float, the parameter represents a proportion of documents, integer absolute counts. This parameter is ignored if vocabulary is not None.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    max_features = hyperparams.Union(
        configuration=OrderedDict({
            'absolute': hyperparams.Bounded[int](
                default=1,
                lower=0,
                upper=None,
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            ),
            'none': hyperparams.Constant(
                default=None,
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            )
        }),
        default='none',
        description='If not None, build a vocabulary that only consider the top max_features ordered by term frequency across the corpus.  This parameter is ignored if vocabulary is not None.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    binary = hyperparams.UniformBool(
        default=False,
        description='If True, all non-zero term counts are set to 1. This does not mean outputs will have only 0/1 values, only that the tf term in tf-idf is binary. (Set idf and normalization to False to get 0/1 outputs.)',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    norm = hyperparams.Union(
        configuration=OrderedDict({
            'str': hyperparams.Enumeration[str](
                default='l2',
                values=['l1', 'l2'],
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            ),
            'none': hyperparams.Constant(
                default=None,
                semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            )
        }),
        default='none',
        description='Norm used to normalize term vectors. None for no normalization.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    use_idf = hyperparams.UniformBool(
        default=True,
        description='Enable inverse-document-frequency reweighting.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    smooth_idf = hyperparams.UniformBool(
        default=True,
        description='Smooth idf weights by adding one to document frequencies, as if an extra document was seen containing every term in the collection exactly once. Prevents zero divisions.',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
    )
    sublinear_tf = hyperparams.UniformBool(
        default=False,
        description='Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter']
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
    
    

class SKTfidfVectorizer(UnsupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    Primitive wrapping for sklearn TfidfVectorizer
    `sklearn documentation <https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.TfidfVectorizer.html>`_
    
    """
    
    __author__ = "JPL MARVIN"
    metadata = metadata_base.PrimitiveMetadata({ 
         "algorithm_types": [metadata_base.PrimitiveAlgorithmType.MINIMUM_REDUNDANCY_FEATURE_SELECTION, ],
         "name": "sklearn.feature_extraction.text.TfidfVectorizer",
         "primitive_family": metadata_base.PrimitiveFamily.DATA_PREPROCESSING,
         "python_path": "d3m.primitives.data_preprocessing.tfidf_vectorizer.SKlearn",
         "source": {'name': 'JPL', 'contact': 'mailto:shah@jpl.nasa.gov', 'uris': ['https://gitlab.com/datadrivendiscovery/sklearn-wrap/issues', 'https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.TfidfVectorizer.html']},
         "version": "2019.11.13",
         "id": "1f7ce2c7-1ec8-3483-9a65-eedd4b5811d6",
         "hyperparams_to_tune": ['max_df', 'min_df'],
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
        
        # True
        
        self._clf = list()
        
        self._training_inputs = None
        self._target_names = None
        self._training_indices = None
        self._fitted = False
        
        
    def set_training_data(self, *, inputs: Inputs) -> None:
        self._inputs = inputs
        self._fitted = False
        
    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        if self._fitted:
            return CallResult(None)

        self._training_inputs, self._training_indices = self._get_columns_to_fit(self._inputs, self.hyperparams)

        if self._training_inputs is None:
            raise ValueError("Missing training data.")

        if len(self._training_indices) > 0:
            for column_index in range(len(self._training_inputs.columns)):
                clf = self._create_new_sklearn_estimator()
                clf.fit(self._training_inputs.iloc[:, column_index])
                self._clf.append(clf)

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
            sk_inputs, training_indices = self._get_columns_to_fit(inputs, self.hyperparams)
        else:
            training_indices = list(range(len(inputs)))

        # Iterating over all estimators and call transform on them.
        # No. of estimators should be equal to the number of columns in the input
        if len(self._clf) != len(sk_inputs.columns):
            raise RuntimeError("Input data does not have the same number of columns as training data")
        outputs = []
        if len(self._training_indices) > 0:
            for column_index in range(len(sk_inputs.columns)):
                clf = self._clf[column_index]
                output = clf.transform(sk_inputs.iloc[:, column_index])
                column_name = sk_inputs.columns[column_index]

                if sparse.issparse(output):
                    output = output.toarray()
                output = self._wrap_predictions(inputs, output)

                # Updating column names.
                output.columns = map(lambda x: "{}_{}".format(column_name, x), clf.get_feature_names())
                for i, name in enumerate(clf.get_feature_names()):
                    output.metadata = output.metadata.update((ALL_ELEMENTS, i), {'name': name})

                outputs.append(output)
        else:
            if self.hyperparams['error_on_no_input']:
                raise RuntimeError("No input columns were selected")
            self.logger.warn("No input columns were selected")

        outputs = base_utils.combine_columns(return_result=self.hyperparams['return_result'],
                                               add_index_columns=self.hyperparams['add_index_columns'],
                                               inputs=inputs, column_indices=self._training_indices,
                                               columns_list=outputs)

        return CallResult(outputs)
        

    def get_params(self) -> Params:
        if not self._fitted:
            return Params(
                vocabulary_=None,
                stop_words_=None,
                _tfidf=None,
                fixed_vocabulary_=None,
                _stop_words_id=None,
                training_indices_=self._training_indices,
                target_names_=self._target_names
            )

        return Params(
            vocabulary_=list(map(lambda clf: getattr(clf, 'vocabulary_', None), self._clf)),
            stop_words_=list(map(lambda clf: getattr(clf, 'stop_words_', None), self._clf)),
            _tfidf=list(map(lambda clf: getattr(clf, '_tfidf', None), self._clf)),
            fixed_vocabulary_=list(map(lambda clf: getattr(clf, 'fixed_vocabulary_', None), self._clf)),
            _stop_words_id=list(map(lambda clf: getattr(clf, '_stop_words_id', None), self._clf)),
            training_indices_=self._training_indices,
            target_names_=self._target_names
        )

    def set_params(self, *, params: Params) -> None:
        for param, val in params.items():
            if val is not None and param not in ['target_names_', 'training_indices_']:
                self._clf = list(map(lambda x: self._create_new_sklearn_estimator(), val))
                break
        for index in range(len(self._clf)):
            for param, val in params.items():
                if val is not None:
                    setattr(self._clf[index], param, val[index])
                else:
                    setattr(self._clf[index], param, None)
        self._training_indices = params['training_indices_']
        self._target_names = params['target_names_']
        self._fitted = False
        
        if params['vocabulary_'] is not None:
            self._fitted = True
        if params['stop_words_'] is not None:
            self._fitted = True
        if params['_tfidf'] is not None:
            self._fitted = True
        if params['fixed_vocabulary_'] is not None:
            self._fitted = True
        if params['_stop_words_id'] is not None:
            self._fitted = True

    def _create_new_sklearn_estimator(self):
        clf = TfidfVectorizer(
                  strip_accents=self.hyperparams['strip_accents'],
                  analyzer=self.hyperparams['analyzer'],
                  ngram_range=self.hyperparams['ngram_range'],
                  stop_words=self.hyperparams['stop_words'],
                  lowercase=self.hyperparams['lowercase'],
                  token_pattern=self.hyperparams['token_pattern'],
                  max_df=self.hyperparams['max_df'],
                  min_df=self.hyperparams['min_df'],
                  max_features=self.hyperparams['max_features'],
                  binary=self.hyperparams['binary'],
                  norm=self.hyperparams['norm'],
                  use_idf=self.hyperparams['use_idf'],
                  smooth_idf=self.hyperparams['smooth_idf'],
                  sublinear_tf=self.hyperparams['sublinear_tf'],
                )
        return clf




    
    
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

        accepted_structural_types = (str,)
        accepted_semantic_types = set(["http://schema.org/Text",])
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
    def _get_targets(cls, data: d3m_dataframe, hyperparams: Hyperparams):
        if not hyperparams['use_semantic_types']:
            return data, list(data.columns), []
        target_names = []
        target_semantic_type = []
        target_column_indices = []
        metadata = data.metadata
        target_column_indices.extend(metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/TrueTarget'))

        for column_index in target_column_indices:
            if column_index is metadata_base.ALL_ELEMENTS:
                continue
            column_index = typing.cast(metadata_base.SimpleSelectorSegment, column_index)
            column_metadata = metadata.query((metadata_base.ALL_ELEMENTS, column_index))
            target_names.append(column_metadata.get('name', str(column_index)))
            target_semantic_type.append(column_metadata.get('semantic_types', []))

        targets = data.iloc[:, target_column_indices]
        return targets, target_names, target_semantic_type

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
        target_columns_metadata = self._add_target_columns_metadata(outputs.metadata)
        outputs.metadata = self._update_predictions_metadata(inputs.metadata, outputs, target_columns_metadata)
        return outputs


    @classmethod
    def _add_target_columns_metadata(cls, outputs_metadata: metadata_base.DataMetadata):
        outputs_length = outputs_metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length']

        target_columns_metadata: List[OrderedDict] = []
        for column_index in range(outputs_length):
            column_metadata = OrderedDict()
            semantic_types = []
            semantic_types.append('https://metadata.datadrivendiscovery.org/types/Attribute')
            column_name = outputs_metadata.query((metadata_base.ALL_ELEMENTS, column_index)).get("name")
            if column_name is None:
                column_name = "output_{}".format(column_index)
            column_metadata["semantic_types"] = semantic_types
            column_metadata["name"] = str(column_name)
            target_columns_metadata.append(column_metadata)

        return target_columns_metadata


SKTfidfVectorizer.__doc__ = TfidfVectorizer.__doc__