import copy
import uuid
import numpy

from d3m.metadata.pipeline import Pipeline, PrimitiveStep, Resolver
from d3m import index
from d3m import runtime as runtime_module
from d3m import utils as d3m_utils

from axolotl.algorithms.base import PipelineSearchBase
from axolotl.utils import schemas as schemas_utils, pipeline as pipeline_utils
from d3m.metadata.base import ArgumentType, ALL_ELEMENTS
from axolotl.algorithms.dummy import dummy_ranking_function
from axolotl.algorithms.bayesian_search import BayesianSearch
import multiprocessing

PREP_PRIMITIVES = {
    'Denormalize': 'd3m.primitives.data_transformation.denormalize.Common',
    'DatasetToDataFrame': 'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
    'ColumnParser': 'd3m.primitives.data_transformation.column_parser.Common',
    'ExtractColumnsBySemanticTypes': 'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',
    'Imputer': 'd3m.primitives.data_cleaning.imputer.SKlearn',
    'SimpleProfiler': 'd3m.primitives.schema_discovery.profiler.Common',
    'ReplaceSemanticTypes': 'd3m.primitives.data_transformation.replace_semantic_types.Common',
    'DropColumns': 'd3m.primitives.data_transformation.remove_columns.Common',
    'OneHotMaker': 'd3m.primitives.data_preprocessing.one_hot_encoder.MakerCommon',
    'ExtractColumns': 'd3m.primitives.data_transformation.extract_columns.Common',
    'GeneralHorizontalConcat': 'd3m.primitives.data_transformation.horizontal_concat.TAMU',
    'Imputer': 'd3m.primitives.data_cleaning.imputer.SKlearn',
    'FeatureSelection': 'd3m.primitives.feature_selection.select_fwe.SKlearn',
    'ConstructPredictions': 'd3m.primitives.data_transformation.construct_predictions.Common',
    'OrdinalEncoder': 'd3m.primitives.data_transformation.ordinal_encoder.SKlearn',
    'RobustScale': 'd3m.primitives.data_preprocessing.robust_scaler.SKlearn',
    'TimeSeriesToList': 'd3m.primitives.data_preprocessing.time_series_to_list.DSBOX',
    'TimeSeriesFeaturization': 'd3m.primitives.feature_extraction.random_projection_timeseries_featurization.DSBOX',
    'TextReader': 'd3m.primitives.data_preprocessing.text_reader.Common',
    'TextEncoder': 'd3m.primitives.data_transformation.encoder.DistilTextEncoder',
    'AddSemanticTypes': 'd3m.primitives.data_transformation.add_semantic_types.Common',
    'SemiClassification': 'd3m.primitives.semisupervised_classification.iterative_labeling.AutonBox'
}

LOADED_PRIMITIVES = {}

DATA_TYPES = {
    'attribute': 'https://metadata.datadrivendiscovery.org/types/Attribute',
    'target': 'https://metadata.datadrivendiscovery.org/types/Target',
    'suggested_target': 'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',
    'true_target': 'https://metadata.datadrivendiscovery.org/types/TrueTarget',
    'float': 'http://schema.org/Float',
    'int': 'http://schema.org/Integer',
    'unknown_type': 'https://metadata.datadrivendiscovery.org/types/UnknownType',
    'categorical': 'https://metadata.datadrivendiscovery.org/types/CategoricalData',
    'text': 'http://schema.org/Text',
    'bool': 'http://schema.org/Boolean',
    'file': 'https://metadata.datadrivendiscovery.org/types/FileName',
    'time_series': 'https://metadata.datadrivendiscovery.org/types/Timeseries',
    'date': 'http://schema.org/DateTime',
    'time': 'https://metadata.datadrivendiscovery.org/types/Time'
}

with d3m_utils.silence():
    for key, value in PREP_PRIMITIVES.items():
        LOADED_PRIMITIVES[key] = index.get_primitive(value)


def get_semantic_types(input_dataframe):
    semantic_types = []
    for i in range(input_dataframe.metadata.query((ALL_ELEMENTS,))['dimension']['length']):
        semantic_types.append(input_dataframe.metadata.query((ALL_ELEMENTS, i,))['semantic_types'])
    return semantic_types


def get_indexes_by_semantic_type(input_dataframe, semantic_type):
    semantic_types = get_semantic_types(input_dataframe)
    indexes = []
    for i in range(len(semantic_types)):
        if semantic_type in semantic_types[i]:
            indexes.append(i)
    return indexes


def get_index_data_to_profile(input_dataframe):
    indexes_to_profile = []
    for i in range(input_dataframe.metadata.query((ALL_ELEMENTS,))['dimension']['length']):
        if DATA_TYPES['unknown_type'] in input_dataframe.metadata.query((ALL_ELEMENTS, i,))['semantic_types'] and \
                input_dataframe.metadata.query((ALL_ELEMENTS, i,))['structural_type'] == str:
            indexes_to_profile.append(i)
    return indexes_to_profile


def run_primitive(primitive, arguments, hyperparams=()):
    # TODO add static support for static file
    _hyperparams = primitive.metadata.get_hyperparams().defaults()
    hp_to_update = {}
    for hyperparam in hyperparams:
        name, argument_type, data = hyperparam
        hp_to_update[name] = data
    _hyperparams = _hyperparams.replace(hp_to_update)
    primitive_instance = primitive(hyperparams=_hyperparams)
    use_set_training_data = pipeline_utils.query_multiple_terms(
        primitive.metadata, ['primitive_code', 'instance_methods', 'set_training_data', 'arguments'])
    if use_set_training_data is not None and use_set_training_data:
        primitive_instance.set_training_data(**arguments)
        primitive_instance.fit()

    produce_arguments = pipeline_utils.query_multiple_terms(
        primitive.metadata, ['primitive_code', 'instance_methods', 'produce', 'arguments'])

    arguments_keys = list(arguments.keys())
    for argument in arguments_keys:
        if argument not in produce_arguments:
            print('removing argument', argument)
            del arguments[argument]
    return primitive_instance.produce(**arguments).value


def add_primitive_step_to_pipeline(pipeline, primitive, arguments=(), hyperparams=(), resolver=Resolver()):
    step = PrimitiveStep(primitive=primitive, resolver=resolver)
    for argument in arguments:
        name, argument_type, data_reference = argument
        step.add_argument(name=name, argument_type=argument_type, data_reference=data_reference)
    for hyperparam in hyperparams:
        name, argument_type, data = hyperparam
        step.add_hyperparameter(name=name, argument_type=argument_type, data=data)
    step.add_output('produce')
    pipeline.add_step(step)


def fix_arguments(arguments):
    _arguments = []
    for name, reference in arguments.items():
        _arguments.append((name, ArgumentType.CONTAINER, reference))
    return  _arguments


def prepare_arguments(available_data, arguments):
    _arguments = {}
    for name, reference in arguments.items():
        if isinstance(reference, list):
            _arguments[name] = []
            for elem in reference:
                _arguments[name].append(available_data[elem])
        else:
            _arguments[name] = available_data[reference]
    return _arguments


def shrink_dataset(dataset, n_rows=10000):
    if 'learningData' not in dataset or len(dataset.keys()) > 1 or len(dataset['learningData']) <= n_rows:
        return dataset

    print('=' * 100)
    print('Shrinking dataset from {} to {}'.format(len(dataset['learningData']), n_rows))
    df = dataset['learningData'].sample(n=n_rows)
    df['d3mIndex'] = df['d3mIndex'].apply(lambda x: int(x))
    df = df.sort_values(by=['d3mIndex'])
    df['d3mIndex'] = df['d3mIndex'].apply(lambda x: str(x))
    df = df.reset_index(drop=True)

    dataset['learningData'] = df
    metadata = dataset.metadata

    metadata = metadata.update(('learningData',), {
            'structural_type': metadata.query(('learningData',))['structural_type'],
            'semantic_types': [
                'https://metadata.datadrivendiscovery.org/types/Table',
                'https://metadata.datadrivendiscovery.org/types/DatasetEntryPoint',
            ],
            'dimension': {
                'name': 'rows',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                'length': n_rows,
            },
        })

    dataset.metadata = metadata
    return dataset


def get_primitives_by_family(family_type):
    pass


def index_to_operate(input_data, data_type, exclude_targets):
    indexes = []
    semantic_types = get_semantic_types(input_data)
    for i in range(len(semantic_types)):
        if data_type in semantic_types[i]:
            if DATA_TYPES['target'] in semantic_types[i]:
                if not exclude_targets:
                    indexes.append(i)
            else:
                indexes.append(i)
    return indexes


DEFAULT_HYPERPARAMS = {
    'ColumnParser': [
        ('parse_semantic_types', ArgumentType.VALUE,
         ['http://schema.org/Integer', 'http://schema.org/Float',
          'https://metadata.datadrivendiscovery.org/types/FloatVector', 'http://schema.org/DateTime']
         )
    ],
    'SimpleProfiler': [
        ('return_result', ArgumentType.VALUE, 'replace'),
        ('categorical_max_absolute_distinct_values', ArgumentType.VALUE, None),
        ('categorical_max_ratio_distinct_values', ArgumentType.VALUE, 0.20)
    ],
    'ReplaceSemanticTypes': [
        ('return_result', ArgumentType.VALUE, 'replace'),
        ('from_semantic_types', ArgumentType.VALUE, [DATA_TYPES['unknown_type']]),
        ('to_semantic_types', ArgumentType.VALUE, [DATA_TYPES['categorical']])
    ],
    'OneHotMaker': [
        ('return_result', ArgumentType.VALUE, 'replace'),
        ('encode_target_columns', ArgumentType.VALUE, True),
        ('handle_unseen', ArgumentType.VALUE, 'column'),
        ('handle_missing_value', ArgumentType.VALUE, 'column')
    ],
    "Imputer": [
        ('return_result', ArgumentType.VALUE, 'replace'),
        ('use_semantic_types', ArgumentType.VALUE, True),
    ],
    'OrdinalEncoder': [
        ('return_result', ArgumentType.VALUE, 'replace'),
        ('use_semantic_types', ArgumentType.VALUE, True),
    ],
    'RobustScale': [
        ('return_result', ArgumentType.VALUE, 'replace'),
        ('use_semantic_types', ArgumentType.VALUE, True),
    ]
}


class PrimitiveHandler:
    def __init__(self, primitive, hyperparams=[], resolver=Resolver()):
        self.primitive = primitive
        self.hyperparams = hyperparams
        self.resolver = resolver

    def add_produce(self, available_data, pipeline, arguments, indexes=[]):
        _arguments = fix_arguments(arguments)

        hyperparams = self.hyperparams
        if indexes and 'use_columns' in self.primitive.metadata.get_hyperparams().defaults():
            hyperparams = self.hyperparams + [('use_columns', ArgumentType.VALUE, indexes)]
        add_primitive_step_to_pipeline(pipeline, self.primitive, _arguments, hyperparams, resolver=self.resolver)
        output = run_primitive(self.primitive, prepare_arguments(available_data, arguments), hyperparams)
        current_data_ref = 'steps.{}.produce'.format(len(pipeline.steps) - 1)
        available_data[current_data_ref] = output
        return current_data_ref

    def run_primitive(self, arguments, hyperparams=[], indexes=[]):
        _hyperparams = self.hyperparams
        if hyperparams:
            _hyperparams = self.hyperparams + hyperparams
        _hyperparams = _hyperparams if not indexes else _hyperparams + [('use_columns', ArgumentType.VALUE, indexes)]
        return run_primitive(self.primitive, arguments, hyperparams)


class FileHandler:

    def __init__(self, resolver=Resolver()):
        self.use_colummns = True
        self.resolver = resolver
        self.exclude_targets = None
        self.problem_description = None
        self.task_description = None

    def add_produce(self, available_data, pipeline, arguments, indexes=[]):
        last_valid_data_ref = arguments['inputs']
        origindal_data_ref = last_valid_data_ref
        current_data_ref = self.add_output_time_series(available_data, pipeline, arguments, indexes=[])

        if current_data_ref is not None:
            arguments = {'inputs': current_data_ref}
            last_valid_data_ref = current_data_ref

        current_data_ref = self.add_output_text(available_data, pipeline, arguments, indexes=[])

        if current_data_ref is not None:
            arguments = {'inputs': current_data_ref}
            last_valid_data_ref = current_data_ref

        if last_valid_data_ref == origindal_data_ref:
            last_valid_data_ref = None

        return True, last_valid_data_ref

    def add_output_time_series(self, available_data, pipeline, arguments, indexes=[]):
        initial_ref = arguments['inputs']
        semantic_types = get_semantic_types(available_data[initial_ref])
        indexes_to_remove = []
        for i, _type in enumerate(semantic_types):
            if DATA_TYPES['file'] in _type and DATA_TYPES['time_series'] in _type:
                indexes_to_remove.append(i)
        if not indexes_to_remove:
            return
        print('File TimeSeriesHandler')
        primitive_handler = PrimitiveHandler(LOADED_PRIMITIVES['TimeSeriesToList'], resolver=self.resolver)
        current_data_ref = primitive_handler.add_produce(available_data, pipeline, arguments)

        primitive_handler = PrimitiveHandler(LOADED_PRIMITIVES['TimeSeriesFeaturization'], resolver=self.resolver)
        current_data_ref_to_concat = primitive_handler.add_produce(available_data, pipeline, {'inputs': current_data_ref})

        drop_hp = [('columns', ArgumentType.VALUE, indexes_to_remove)]
        primitive_handler = PrimitiveHandler(LOADED_PRIMITIVES['DropColumns'], drop_hp, resolver=self.resolver)
        current_data_ref = primitive_handler.add_produce(available_data, pipeline, {'inputs': initial_ref})

        data_refs_to_concat = [current_data_ref, current_data_ref_to_concat]
        primitive_handler = PrimitiveHandler(LOADED_PRIMITIVES['GeneralHorizontalConcat'], resolver=self.resolver)
        last_data_ref = primitive_handler.add_produce(available_data, pipeline, {'inputs': data_refs_to_concat})
        return last_data_ref

    def add_output_text(self, available_data, pipeline, arguments, indexes=[]):
        initial_ref = arguments['inputs']
        semantic_types = get_semantic_types(available_data[initial_ref])
        indexes_to_remove = []
        for i, _type in enumerate(semantic_types):
            if DATA_TYPES['file'] in _type and DATA_TYPES['text'] in _type:
                indexes_to_remove.append(i)
        if not indexes_to_remove:
            return

        print('File TextReader Handler')
        text_rd_hp = [('return_result', ArgumentType.VALUE, 'replace')]
        primitive_handler = PrimitiveHandler(LOADED_PRIMITIVES['TextReader'], text_rd_hp, resolver=self.resolver)
        text_data_ref = primitive_handler.add_produce(available_data, pipeline, arguments)

        primitive_handler = PrimitiveHandler(LOADED_PRIMITIVES['ExtractColumnsBySemanticTypes'],
                                             [('semantic_types', ArgumentType.VALUE, [DATA_TYPES['attribute']])],
                                             self.resolver)
        attributes_data_ref = primitive_handler.add_produce(available_data, pipeline, {'inputs': text_data_ref})

        primitive_handler = PrimitiveHandler(LOADED_PRIMITIVES['ExtractColumnsBySemanticTypes'],
                                             [('semantic_types', ArgumentType.VALUE, [DATA_TYPES['target']])],
                                             self.resolver)
        target_data_ref = primitive_handler.add_produce(available_data, pipeline, {'inputs': text_data_ref})

        primitive_handler = PrimitiveHandler(LOADED_PRIMITIVES['TextEncoder'],
                                             [('encoder_type', ArgumentType.VALUE, 'tfidf')],
                                             self.resolver)
        current_data_ref = primitive_handler.add_produce(
            available_data, pipeline, {'inputs': attributes_data_ref, 'outputs': target_data_ref})

        no_semantic_types = []
        for i in range(available_data[current_data_ref].metadata.query((ALL_ELEMENTS,))['dimension']['length']):
            if 'semantic_types' not in available_data[current_data_ref].metadata.query((ALL_ELEMENTS, i,)) and \
                    available_data[current_data_ref].metadata.query((ALL_ELEMENTS, i,))['structural_type'] == numpy.float64:
                no_semantic_types.append(i)

        add_semantic_hp = [('columns', ArgumentType.VALUE, no_semantic_types),
                           ('semantic_types', ArgumentType.VALUE, [DATA_TYPES['float'], DATA_TYPES['attribute']])]
        primitive_handler = PrimitiveHandler(LOADED_PRIMITIVES['AddSemanticTypes'], add_semantic_hp, resolver=self.resolver)
        current_data_ref = primitive_handler.add_produce(available_data, pipeline, {'inputs': current_data_ref})

        data_refs_to_concat = [target_data_ref, current_data_ref]
        primitive_handler = PrimitiveHandler(LOADED_PRIMITIVES['GeneralHorizontalConcat'], resolver=self.resolver)
        last_data_ref = primitive_handler.add_produce(available_data, pipeline, {'inputs': data_refs_to_concat})

        return last_data_ref


class CategoricalHandler:
    def __init__(self, resolver=Resolver()):
        self.use_colummns = True
        self.resolver = resolver
        self.exclude_targets = None
        self.problem_description = None
        self.task_description = None

    def _get_criteria(self, input_data, indexes=[]):
        index_to_ordinal = []
        index_to_drop = []
        index_to_one_hot = []

        total_n_values = len(input_data)
        for _index in indexes:
            n_categories = len(input_data.iloc[:, _index].unique())
            categories_ratio = n_categories/total_n_values
            if categories_ratio >= 0.8:
                index_to_drop.append(_index)
            else:
                if n_categories <= 10:
                    index_to_one_hot.append(_index)
                else:
                    if n_categories <= 100 and not input_data.iloc[:, _index].isnull().values.any():
                        index_to_ordinal.append(_index)
                    else:
                        index_to_drop.append(_index)
        return index_to_ordinal, index_to_one_hot, index_to_drop

    def add_produce(self, available_data, pipeline, arguments, indexes=[]):
        index_to_ordinal, index_to_one_hot, index_to_drop = self._get_criteria(
            available_data[arguments['inputs']], indexes)
        _arguments = fix_arguments(arguments)
        current_data_ref = arguments['inputs']

        index_to_drop += index_to_ordinal

        if index_to_drop:
            print('Drop columns', index_to_drop)
            drop_hp = [('columns', ArgumentType.VALUE, index_to_drop)]
            primitive_handler = PrimitiveHandler(LOADED_PRIMITIVES['DropColumns'], drop_hp, resolver=self.resolver)
            current_data_ref = primitive_handler.add_produce(available_data, pipeline, {'inputs': current_data_ref})

        if index_to_one_hot:
            new_indexes = index_to_operate(available_data[current_data_ref], DATA_TYPES['categorical'], self.exclude_targets)
            _, index_to_one_hot, _ = self._get_criteria(available_data[current_data_ref], new_indexes)
            print('OneHot', index_to_one_hot)

            one_hot_hp = DEFAULT_HYPERPARAMS['OneHotMaker'] + [('use_columns', ArgumentType.VALUE, index_to_one_hot)]
            primitive_handler = PrimitiveHandler(LOADED_PRIMITIVES['OneHotMaker'], one_hot_hp, resolver=self.resolver)
            current_data_ref = primitive_handler.add_produce(available_data, pipeline, {'inputs': current_data_ref})

        # if index_to_ordinal:
        #     new_indexes = index_to_operate(available_data[current_data_ref], DATA_TYPES['categorical'],
        #                                    self.exclude_targets)
        #     index_to_ordinal, _, _ = self._get_criteria(available_data[current_data_ref], new_indexes)
        #     primitive = LOADED_PRIMITIVES['OrdinalEncoder']
        #     ordinal_hp = DEFAULT_HYPERPARAMS['OrdinalEncoder'] + [('use_columns', ArgumentType.VALUE, index_to_ordinal)]
        #     add_primitive_step_to_pipeline(pipeline, primitive, _arguments, ordinal_hp, resolver=self.resolver)
        #     output = run_primitive(primitive, prepare_arguments(available_data, arguments), ordinal_hp)
        #     current_data_ref = 'steps.{}.produce'.format(len(pipeline.steps) - 1)
        #     available_data[current_data_ref] = output
        #     arguments = {'inputs': current_data_ref}
        #     _arguments = fix_arguments(arguments)
        #
        #     cat_indexes = get_indexes_by_semantic_type(available_data[current_data_ref], DATA_TYPES['categorical'])
        #     index_to_fix = []
        #     for _index in cat_indexes:
        #         if available_data[current_data_ref].metadata.query((ALL_ELEMENTS, _index,))['structural_type'] == numpy.float64:
        #             index_to_fix.append(_index)
        #
        #     if index_to_fix:
        #         primitive = LOADED_PRIMITIVES['ReplaceSemanticTypes']
        #         replace_sem_hp = [
        #             ('return_result', ArgumentType.VALUE, 'replace'),
        #             ('from_semantic_types', ArgumentType.VALUE, [DATA_TYPES['categorical']]),
        #             ('to_semantic_types', ArgumentType.VALUE, [DATA_TYPES['float']]),
        #             ('use_columns', ArgumentType.VALUE, index_to_fix)
        #         ]
        #         add_primitive_step_to_pipeline(pipeline, primitive, _arguments, replace_sem_hp, resolver=self.resolver)
        #         output = run_primitive(primitive, prepare_arguments(available_data, arguments), replace_sem_hp)
        #         current_data_ref = 'steps.{}.produce'.format(len(pipeline.steps) - 1)
        #         available_data[current_data_ref] = output
        return True, current_data_ref


class BooleanHandler:
    def __init__(self, resolver=Resolver()):
        self.use_colummns = True
        self.resolver = resolver
        self.exclude_targets = None
        self.problem_description = None
        self.task_description = None

    def add_produce(self, available_data, pipeline, arguments, indexes=[]):
        indexes = index_to_operate(available_data[arguments['inputs']], DATA_TYPES['bool'], self.exclude_targets)
        if not indexes:
            print("Skipping Boolean no columns to operate")
            return True, None


        replace_sem_hp = [
            ('return_result', ArgumentType.VALUE, 'replace'),
            ('from_semantic_types', ArgumentType.VALUE, [DATA_TYPES['bool']]),
            ('to_semantic_types', ArgumentType.VALUE, [DATA_TYPES['categorical']]),
            ('use_columns', ArgumentType.VALUE, indexes)
        ]
        primitive_handler = PrimitiveHandler(
            LOADED_PRIMITIVES['ReplaceSemanticTypes'], replace_sem_hp, resolver=self.resolver)
        current_data_ref = primitive_handler.add_produce(available_data, pipeline, arguments)

        one_hot_hp = [
            ('return_result', ArgumentType.VALUE, 'replace'),
            ('encode_target_columns', ArgumentType.VALUE, True),
            ('handle_missing_value', ArgumentType.VALUE, 'column'),
            ('use_columns', ArgumentType.VALUE, indexes)
        ]
        primitive_handler = PrimitiveHandler(
            LOADED_PRIMITIVES['OneHotMaker'], one_hot_hp, resolver=self.resolver)
        current_data_ref = primitive_handler.add_produce(available_data, pipeline, {'inputs': current_data_ref})
        return True, current_data_ref


class DateHandler:
    def __init__(self, resolver=Resolver()):
        self.use_colummns = True
        self.resolver = resolver
        self.exclude_targets = None
        self.problem_description = None
        self.task_description = None

    def add_produce(self, available_data, pipeline, arguments, indexes=[]):
        indexes = []
        semantic_types = get_semantic_types(available_data[arguments['inputs']])
        for i in range(len(semantic_types)):
            if DATA_TYPES['date'] in semantic_types[i] and DATA_TYPES['time'] in semantic_types[i]:
                if DATA_TYPES['target'] in semantic_types[i]:
                    if not self.exclude_targets:
                        indexes.append(i)
                else:
                    indexes.append(i)

        if not indexes:
            print("Skipping Boolean no columns to operate")
            return True, None

        replace_sem_hp = [
            ('return_result', ArgumentType.VALUE, 'replace'),
            ('from_semantic_types', ArgumentType.VALUE, [DATA_TYPES['date'], DATA_TYPES['time']]),
            ('to_semantic_types', ArgumentType.VALUE, [DATA_TYPES['float']]),
            ('use_columns', ArgumentType.VALUE, indexes)
        ]
        primitive_handler = PrimitiveHandler(
            LOADED_PRIMITIVES['ReplaceSemanticTypes'], replace_sem_hp, resolver=self.resolver)
        current_data_ref = primitive_handler.add_produce(available_data, pipeline, arguments)

        return True, current_data_ref


class TextHandler:
    def __init__(self, resolver=Resolver()):
        self.use_colummns = True
        self.resolver = resolver
        self.exclude_targets = None
        self.problem_description = None
        self.task_description = None

    def add_produce(self, available_data, pipeline, arguments, indexes=[]):
        indexes = []
        semantic_types = get_semantic_types(available_data[arguments['inputs']])
        for i in range(len(semantic_types)):
            if DATA_TYPES['text'] in semantic_types[i] and not DATA_TYPES['file'] in semantic_types[i]:
                if DATA_TYPES['target'] in semantic_types[i]:
                    if not self.exclude_targets:
                        indexes.append(i)
                else:
                    indexes.append(i)

        if not indexes:
            print("Skipping Text no columns to operate")
            return True, None

        print('TextHandler')
        primitive_handler = PrimitiveHandler(LOADED_PRIMITIVES['ExtractColumnsBySemanticTypes'],
                                             [('semantic_types', ArgumentType.VALUE, [DATA_TYPES['attribute']])],
                                             self.resolver)
        attributes_data_ref = primitive_handler.add_produce(available_data, pipeline, arguments)

        primitive_handler = PrimitiveHandler(LOADED_PRIMITIVES['ExtractColumnsBySemanticTypes'],
                                             [('semantic_types', ArgumentType.VALUE, [DATA_TYPES['target']])],
                                             self.resolver)
        target_data_ref = primitive_handler.add_produce(available_data, pipeline, arguments)

        primitive_handler = PrimitiveHandler(LOADED_PRIMITIVES['TextEncoder'],
                                             [('encoder_type', ArgumentType.VALUE, 'tfidf')],
                                             self.resolver)
        current_data_ref = primitive_handler.add_produce(
            available_data, pipeline, {'inputs': attributes_data_ref, 'outputs': target_data_ref})

        no_semantic_types = []
        for i in range(available_data[current_data_ref].metadata.query((ALL_ELEMENTS,))['dimension']['length']):
            if 'semantic_types' not in available_data[current_data_ref].metadata.query((ALL_ELEMENTS, i,)) and \
                    available_data[current_data_ref].metadata.query((ALL_ELEMENTS, i,))[
                        'structural_type'] == numpy.float64:
                no_semantic_types.append(i)

        add_semantic_hp = [('columns', ArgumentType.VALUE, no_semantic_types),
                           ('semantic_types', ArgumentType.VALUE, [DATA_TYPES['float'], DATA_TYPES['attribute']])]
        primitive_handler = PrimitiveHandler(LOADED_PRIMITIVES['AddSemanticTypes'], add_semantic_hp,
                                             resolver=self.resolver)
        current_data_ref = primitive_handler.add_produce(available_data, pipeline, {'inputs': current_data_ref})

        data_refs_to_concat = [target_data_ref, current_data_ref]
        primitive_handler = PrimitiveHandler(LOADED_PRIMITIVES['GeneralHorizontalConcat'], resolver=self.resolver)
        last_data_ref = primitive_handler.add_produce(available_data, pipeline, {'inputs': data_refs_to_concat})

        return True, last_data_ref



class DataTypesHandler:
    def __init__(self, problem_description, task_description,
                 handlers=None, use_default_handlers=True, exclude_targets=True, resolver=Resolver()):
        DEFAULT_DATA_HANDLERS = {
            DATA_TYPES['float']: None,
            DATA_TYPES['int']: None,
            DATA_TYPES['bool']: BooleanHandler(resolver=resolver),
            DATA_TYPES['categorical']: CategoricalHandler(resolver=resolver),
            DATA_TYPES['date']: DateHandler(resolver=resolver),
            DATA_TYPES['file']: FileHandler(resolver=resolver),
            DATA_TYPES['text']: TextHandler(resolver=resolver)
        }
        self.problem_description = problem_description
        self.task_description = task_description
        self.resolver = resolver
        self.exclude_targets = exclude_targets
        if handlers is None:
            self.handlers = DEFAULT_DATA_HANDLERS
        else:
            if use_default_handlers:
                self.handlers = DEFAULT_DATA_HANDLERS
                for name, handler in handlers.items():
                    self.handlers[name] = handlers
            else:
                self.handlers = handlers

    def add_produce(self, pipeline, input_dataframe):
        data_ref = 'steps.{}.produce'.format(len(pipeline.steps) - 1)
        available_data = {data_ref: input_dataframe}
        last_data_ref = data_ref

        use_columns = []
        not_use_columns = []
        for handler_name in self.handlers.keys():
            if self.check_use_columns_in_handler(handler_name):
                use_columns.append(handler_name)
            else:
                not_use_columns.append(handler_name)

        last_use_column_handler_index = len(use_columns) - 1
        handler_names = use_columns + not_use_columns
        last_use_column_handler_data_ref = None
        data_refs_to_concat = []

        # We execute the handler in order according to whether or not the support use_columns.
        for i, handler_name in enumerate(handler_names):
            print(i, handler_name)
            use_columns, new_data_ref = self.execute_handler(available_data, pipeline, last_data_ref, handler_name)
            if new_data_ref is not None:
                last_data_ref = new_data_ref
                if i == last_use_column_handler_index:
                    last_use_column_handler_data_ref = last_data_ref
                elif i > last_use_column_handler_index:
                    data_refs_to_concat.append(new_data_ref)

        # we get the columns of the ones that we use by using negation of excluiding types.
        # we do this if there are not_use_columns
        if not_use_columns:
            # get the columns that columns that were not modified or used use_columns
            primitive_handler = PrimitiveHandler(
                LOADED_PRIMITIVES['ExtractColumnsBySemanticTypes'],
                [('semantic_types', ArgumentType.VALUE, not_use_columns), ('negate', ArgumentType.VALUE, True)],
                self.resolver)
            new_data_ref = primitive_handler.add_produce(available_data, pipeline, {'inputs': last_use_column_handler_data_ref})
            data_refs_to_concat.insert(0, new_data_ref)

            # We concatenate all together
            primitive_handler = PrimitiveHandler(LOADED_PRIMITIVES['GeneralHorizontalConcat'], resolver=self.resolver)
            last_data_ref = primitive_handler.add_produce(available_data, pipeline, {'inputs': data_refs_to_concat})

        return available_data[last_data_ref], pipeline

    def check_use_columns_in_handler(self, handler_name):
        use_columns = True
        if self.handlers[handler_name] is not None:
            if isinstance(self.handlers[handler_name], PrimitiveHandler):
                use_columns = 'use_columns' in self.handlers[handler_name].primitive.metadata.get_hyperparams().defaults()
            else:
                use_columns = self.handlers[handler_name].use_colummns
        return use_columns

    def execute_handler(self, available_data, pipeline, data_ref, handler_name):
        new_data_ref = None
        use_columns = False
        if self.handlers[handler_name] is not None:
            if isinstance(self.handlers[handler_name], PrimitiveHandler):
                use_columns, new_data_ref = self._execute_primitive_handler(available_data, pipeline, data_ref, handler_name)
            else:
                self.handlers[handler_name].exclude_targets = self.exclude_targets
                self.handlers[handler_name].problem_description = self.problem_description
                self.handlers[handler_name].task_description = self.task_description
                indexes = self._index_to_operate(available_data[data_ref], handler_name)
                if indexes:
                    use_columns, new_data_ref = self.handlers[handler_name].add_produce(
                        available_data, pipeline, {'inputs': data_ref}, indexes)
                else:
                    print('Skipping', handler_name)
        return use_columns, new_data_ref

    def _index_to_operate(self, input_data, data_type):
        indexes = []
        semantic_types = get_semantic_types(input_data)
        for i in range(len(semantic_types)):
            if data_type in semantic_types[i]:
                if DATA_TYPES['target'] in semantic_types[i]:
                    if not self.exclude_targets:
                        indexes.append(i)
                else:
                    indexes.append(i)
        return indexes

    def _execute_primitive_handler(self, available_data, pipeline, data_ref, handler_name):
        use_columns = 'use_columns' in self.handlers[handler_name].primitive.metadata.get_hyperparams().defaults()
        indexes = self._index_to_operate(available_data[data_ref],handler_name)
        # if no columns to operate, return
        if not indexes:
            return [], None

        if use_columns:
            new_data_ref = self.handlers[handler_name].add_produce(
                available_data, pipeline, {'inputs': data_ref}, indexes)
        else:
            # get the columns with specific semnatic types and then we run the primitive with the inputs
            primitive_handler = PrimitiveHandler(
                LOADED_PRIMITIVES['ExtractColumnsBySemanticTypes'], [('columns', ArgumentType.VALUE, indexes)], self.resolver)
            new_data_ref = primitive_handler.add_produce(available_data, pipeline, {'inputs': data_ref})
            new_data_ref = self.handlers[handler_name].add_produce(
                available_data, pipeline, {'inputs': available_data[new_data_ref]}, indexes)
        return use_columns, new_data_ref


class Preprocessing:
    def __init__(self, problem_description, task_description, *, primitives_blocklist=None, resolver=None):
        self.problem_description = problem_description
        self.task_description = task_description
        self.primitives_blocklist = [] if primitives_blocklist is None else primitives_blocklist
        self.resolver = Resolver(primitives_blocklist=primitives_blocklist) if resolver is None else resolver

        self.profile_pipeline = None
        self.parsed_pipeline = None
        self.featurization_pipeline = None
        self.imputed_pipeline = None
        self.feature_selection_pipeline = None
        self.dataframe_data = None
        self.dataframe_reference = None

    def get_imputed_pipline(self, input_data, pipeline=None, handler=None):
        if pipeline is None:
            pipeline = copy.deepcopy(self.featurization_pipeline)
        if handler is None:
            self.imputed_pipeline = pipeline
            return
        if not input_data.isnull().values.any():
            print('No Nan Values found')
            self.imputed_pipeline = pipeline
            return

        current_data_ref = 'steps.{}.produce'.format(len(pipeline.steps) - 1)
        available_data = {current_data_ref: input_data}
        current_data_ref = handler.add_produce(available_data, pipeline, {'inputs': current_data_ref})
        self.dataframe_data = available_data[current_data_ref]
        self.imputed_pipeline = pipeline

    def get_feature_selection_pipeline(self, input_data, pipeline=None, handler=None):
        if pipeline is None:
            pipeline = copy.deepcopy(self.imputed_pipeline)
        if handler is None:
            self.feature_selection_pipeline = pipeline
            return
        current_data_ref = 'steps.{}.produce'.format(len(pipeline.steps) - 1)
        available_data = {current_data_ref: input_data}
        current_data_ref = handler.add_produce(available_data, pipeline, {'inputs': current_data_ref})
        self.dataframe_data = available_data[current_data_ref]
        self.feature_selection_pipeline = pipeline

    def get_data_handler_pipeline(self, input_data, pipeline=None):
        if pipeline is None:
            pipeline = copy.deepcopy(self.parsed_pipeline)
        type_handler = DataTypesHandler(self.problem_description, self.task_description)
        self.dataframe_data, self.featurization_pipeline = type_handler.add_produce(pipeline, input_data)

    def get_parsed_dataframe(self, input_data, pipeline=None):
        if pipeline is None:
            pipeline = copy.deepcopy(self.profile_pipeline)
        current_data_ref = 'steps.{}.produce'.format(len(pipeline.steps) - 1)
        available_data = {current_data_ref: input_data}
        primitive_handler = PrimitiveHandler(LOADED_PRIMITIVES['ColumnParser'], DEFAULT_HYPERPARAMS['ColumnParser'], self.resolver)
        current_data_ref = primitive_handler.add_produce(available_data, pipeline, {'inputs': current_data_ref})
        self.dataframe_data = available_data[current_data_ref]
        self.parsed_pipeline = pipeline

    def get_dataset_to_dataframe_pipeline(self, input_data, pipeline=None):
        if pipeline is None:
            pipeline = Pipeline()
            pipeline.add_input('input_data')
        current_data_ref = 'inputs.0'
        available_data = {}

        if len(input_data) > 1:
            raise ValueError('Search with multiple inputs is not supported yet.')
        _input_data, _ = runtime_module.Runtime._mark_columns(self.problem_description.get('inputs', []), input_data[-1])
        available_data[current_data_ref] = _input_data

        # Add denormalize
        if len(_input_data.keys()) > 1:
            print('There are multiple resources, adding denormalize')
            primitive_handler = PrimitiveHandler(LOADED_PRIMITIVES['Denormalize'], resolver=self.resolver)
            current_data_ref = primitive_handler.add_produce(available_data, pipeline, {'inputs': current_data_ref})

        # Add dataset to dataframe
        print('Adding dataset to dataframe')
        primitive_handler = PrimitiveHandler(LOADED_PRIMITIVES['DatasetToDataFrame'], resolver=self.resolver)
        current_data_ref = primitive_handler.add_produce(available_data, pipeline, {'inputs': current_data_ref})

        # add profiling
        index_to_profile = get_index_data_to_profile(available_data[current_data_ref])
        if index_to_profile:
            current_data_ref = self.profile_data(available_data, pipeline, current_data_ref, index_to_profile)

        self.dataframe_reference = current_data_ref
        self.dataframe_data = available_data[current_data_ref]
        self.profile_pipeline = pipeline

    def profile_data(self, available_data, pipeline, data_ref, index_to_profile):
        # Thi sfunction helps to abstract the process when the data is profiled.
        target_indexes = get_indexes_by_semantic_type(available_data[data_ref], DATA_TYPES['target'])

        primitive_handler = PrimitiveHandler(LOADED_PRIMITIVES['SimpleProfiler'], DEFAULT_HYPERPARAMS['SimpleProfiler'],
                                             self.resolver)
        profiled_output = primitive_handler.run_primitive({'inputs': available_data[data_ref]},
                                                          indexes=index_to_profile)
        profiles_semantic_types = get_semantic_types(profiled_output)

        # TODO make a list of tasks that has discrete target
        # If the task is classification we need to make sure that the targets are categorical,
        # otherwise there is a chance that the targets are considered as numerical an wrongly parse.
        categorical_indexes = []
        if self.task_description['task_type'] == 'CLASSIFICATION':
            for i in target_indexes:
                if DATA_TYPES['categorical'] not in profiles_semantic_types[i]:
                    index_to_profile.remove(i)
                    categorical_indexes.append(i)
        current_data_ref = primitive_handler.add_produce(available_data, pipeline, {'inputs': data_ref},
                                                         indexes=index_to_profile)
        if categorical_indexes:
            primitive_handler = PrimitiveHandler(LOADED_PRIMITIVES['ReplaceSemanticTypes'],
                                                 DEFAULT_HYPERPARAMS['ReplaceSemanticTypes'], self.resolver)
            current_data_ref = primitive_handler.add_produce(available_data, pipeline, {'inputs': current_data_ref},
                                                             indexes=categorical_indexes)

        return current_data_ref

    def generate_preprocessing_by_step(self, input_data=None, feature_selection_handler=None, impute_handler=None):
        if self.profile_pipeline is None:
            print('=' * 100)
            print('profiled pipeline')
            self.get_dataset_to_dataframe_pipeline(input_data)
            return []
        elif self.parsed_pipeline is None:
            print('=' * 100)
            print('parsing')
            self.get_parsed_dataframe(self.dataframe_data)
            self.dataframe_data.metadata.pretty_print()
            return []
        elif self.featurization_pipeline is None:
            print('=' * 100)
            print('feature')
            self.get_data_handler_pipeline(self.dataframe_data)
            return []
        elif self.imputed_pipeline is None:
            print('=' * 100)
            print('Imputer')
            self.get_imputed_pipline(self.dataframe_data, handler=impute_handler)
            return []
        elif self.feature_selection_pipeline is None:
            print('=' * 100)
            print('selection')
            self.get_feature_selection_pipeline(self.dataframe_data, handler=feature_selection_handler)
            print(self.dataframe_data)
            self.dataframe_data.metadata.pretty_print()
            return []


class DataDrivenSearch(PipelineSearchBase):
    def __init__(self, problem_description, backend, *, primitives_blocklist=None,
                 ranking_function=None, hyperparameter_tuner=BayesianSearch, n_workers=1):
        super().__init__(problem_description=problem_description, backend=backend,
                         primitives_blocklist=primitives_blocklist, ranking_function=ranking_function)
        if self.ranking_function is None:
            self.ranking_function = dummy_ranking_function

        self.task_description = schemas_utils.get_task_description(self.problem_description['problem']['task_keywords'])
        self.resolver = Resolver(primitives_blocklist=self.primitives_blocklist)

        print(self.task_description)
        print(self.problem_description['problem'])

        self.preprocessing = Preprocessing(self.problem_description, self.task_description,
                                           primitives_blocklist=self.primitives_blocklist)
        self.preprocessing_handlers = None
        self.max_num_pipelines_to_eval = n_workers
        print('max_num_pipelines_to_eval', self.max_num_pipelines_to_eval)
        # self.max_num_pipelines_to_eval = 1

        self.search_started = False
        self.total_time = None
        self.learner_candidates = None
        self.failed_learner = []
        self.successful_learner = []
        # TODO update this to be defined on problem/metrics terms
        self.data_preparation_pipeline = schemas_utils.get_splitting_pipeline("TRAINING_DATA")
        self.metrics = self.problem_description['problem']['performance_metrics']

        self.scoring_pipeline = schemas_utils.get_scoring_pipeline()
        self.data_preparation_params = schemas_utils.DATA_PREPARATION_PARAMS['no_split']

        self.tuner_enable = False
        self.hyperparameter_tunner_init = False
        self.hyperparameter_tunner = hyperparameter_tuner(
            self.problem_description, self.backend, primitives_blocklist=self.primitives_blocklist,
            max_trials=100000, directory=self.backend.scratch_dir)
        self.n_pipelines_to_tune = self.max_num_pipelines_to_eval

    def _search(self, time_left):
        if self.preprocessing.profile_pipeline is None:
            self.preprocessing_handlers = {
                'input_data': self.input_data,
                'impute_handler': PrimitiveHandler(primitive=LOADED_PRIMITIVES['Imputer'],
                                                   hyperparams=DEFAULT_HYPERPARAMS['Imputer'],
                                                   resolver=self.resolver),
                'feature_selection_handler': PrimitiveHandler(primitive=LOADED_PRIMITIVES['RobustScale'],
                                                              hyperparams=DEFAULT_HYPERPARAMS['RobustScale'],
                                                              resolver=self.resolver),
            }
        if self.preprocessing.feature_selection_pipeline is None:
            return self.preprocessing.generate_preprocessing_by_step(**self.preprocessing_handlers)

        if self.learner_candidates is None:
            self.input_data = [shrink_dataset(self.input_data[0])]
            terms_to_block = ['data_augmentation', 'data_preprocessing', 'data_cleaning',
                              'data_transformation', 'evaluation', 'feature_construction',
                              'feature_extraction', 'layer', 'loss_function', 'metalearning',
                              'operator', 'schema_discovery',
                              'd3m.primitives.semisupervised_classification.iterative_labeling.AutonBox']
            mapped_task = False
            learner_candidates = pipeline_utils.filter_primitives_by_dataframe_input(
                pipeline_utils.get_primitive_candidates(
                    self.task_description['task_type'], self.task_description['data_types'],
                    self.task_description['semi'], extra_block=terms_to_block)
            )
            if not learner_candidates:
                mapped_task = True
                learner_candidates = pipeline_utils.filter_primitives_by_dataframe_input(
                    pipeline_utils.get_primitive_candidates(
                        schemas_utils.get_task_mapping(self.task_description['task_type']),
                        self.task_description['data_types'], self.task_description['semi'], extra_block=terms_to_block)
                )
            if self.task_description['task_type'] != 'CLASSIFICATION' and \
                    self.task_description['task_type'] != 'REGRESSION' and \
                    learner_candidates and not mapped_task:
                learner_candidates = pipeline_utils.filter_primitives_by_dataframe_input(
                    pipeline_utils.get_primitive_candidates(
                        schemas_utils.get_task_mapping(self.task_description['task_type']), self.task_description['data_types'],
                        self.task_description['semi'], extra_block=terms_to_block)
                ) + learner_candidates

            self.learner_candidates = list(set([info[0] for info in learner_candidates]))
            print(len(self.learner_candidates), self.learner_candidates)
            return []

        if len(self.learner_candidates) > len(self.failed_learner) + len(self.successful_learner):
            print('Model Selection')
            pipelines_to_eval = []
            for leaner_candidate in self.learner_candidates:
                if len(pipelines_to_eval) >= self.max_num_pipelines_to_eval:
                    break

                if leaner_candidate not in self.failed_learner and leaner_candidate not in self.successful_learner:
                    pipeline = self.complete_pipeline(self.preprocessing.feature_selection_pipeline, leaner_candidate)
                    if pipeline is None:
                        self.failed_learner.append(leaner_candidate)
                    else:
                        print('Evaluating', leaner_candidate)
                        self.successful_learner.append(leaner_candidate)
                        pipelines_to_eval.append(pipeline)
            pipeline_results = self.backend.evaluate_pipelines(
                problem_description=self.problem_description, pipelines=pipelines_to_eval, input_data=self.input_data,
                metrics=self.metrics, data_preparation_pipeline=self.data_preparation_pipeline,
                scoring_pipeline=self.scoring_pipeline, data_preparation_params=self.data_preparation_params)

            return [self.ranking_function(pipeline_result) for pipeline_result in pipeline_results]

        if not self.hyperparameter_tunner_init and not self.tuner_enable:
            print('init tuner')
            self.hyperparameter_tunner_init = True
            completed_pipelines = [result for result in self.history if result.status == 'COMPLETED']
            if not completed_pipelines:
                print('No pipelines to tune')
                return []
            completed_pipelines.sort(key=lambda x: x.rank)
            pipeline_candidates = completed_pipelines[:self.n_pipelines_to_tune]
            pipeline_candidates = [candidate.pipeline for candidate in pipeline_candidates]
            self.hyperparameter_tunner.set_pipeline_candidates(self.input_data, pipeline_candidates)
            self.hyperparameter_tunner.init_search_space()
            self.hyperparameter_tunner.input_data = self.input_data
            self.tuner_enable = True

        if self.hyperparameter_tunner_init and self.tuner_enable:
            return self.hyperparameter_tunner._search(time_left)
        return []

    def complete_pipeline(self, pipeline, primitive):

        def add_construct_predictions(_pipeline, _dataframe_ref, _resolver):
            _data_ref = 'steps.{}.produce'.format(len(_pipeline.steps) - 1)
            _arguments={'inputs': _data_ref, 'reference': _dataframe_ref}
            add_primitive_step_to_pipeline(
                _pipeline, LOADED_PRIMITIVES['ConstructPredictions'], fix_arguments(_arguments), resolver=_resolver)
            _data_ref = 'steps.{}.produce'.format(len(_pipeline.steps) - 1)
            _pipeline.add_output(_data_ref, 'output')

        new_pipeline = copy.deepcopy(pipeline)
        new_pipeline.id = str(uuid.uuid4())
        new_pipeline.created = Pipeline().created

        data_ref = 'steps.{}.produce'.format(len(new_pipeline.steps) - 1)

        primitive_arguments = pipeline_utils.query_multiple_terms(
            primitive.metadata, ['primitive_code', 'arguments'])

        failed = False

        if not self.task_description['semi']:

            # we check if the primitive has use_semantic_types
            # if that is the case, it is straight forward to complete the pipeline
            try:
                if 'use_semantic_types' in primitive.metadata.get_hyperparams().defaults():
                    arguments = {'inputs': data_ref}
                    hyperparams = [('use_semantic_types', ArgumentType.VALUE, True)]
                    if 'outputs' in primitive_arguments:
                        arguments['outputs'] = data_ref
                    if 'return_result' in primitive.metadata.get_hyperparams().defaults():
                        hyperparams.append(('return_result', ArgumentType.VALUE, 'replace'))

                    add_primitive_step_to_pipeline(new_pipeline, primitive, fix_arguments(arguments), hyperparams, self.resolver)
                    add_construct_predictions(new_pipeline, self.preprocessing.dataframe_reference, self.resolver)
                else:
                    # Otherwise, we need to get the inputs and outputs via extract columns by semantic_types
                    # for this case, we are assuming that th interface has inputs and outputs
                    arguments = {'inputs': data_ref}
                    attributes_hyperparams = [('semantic_types', ArgumentType.VALUE, [DATA_TYPES['attribute']])]
                    target_hyperparams = [('semantic_types', ArgumentType.VALUE, [DATA_TYPES['target']])]
                    add_primitive_step_to_pipeline(
                        new_pipeline, LOADED_PRIMITIVES['ExtractColumnsBySemanticTypes'], fix_arguments(arguments),
                        attributes_hyperparams, self.resolver)
                    attributes_data_ref = 'steps.{}.produce'.format(len(new_pipeline.steps) - 1)

                    add_primitive_step_to_pipeline(
                        new_pipeline, LOADED_PRIMITIVES['ExtractColumnsBySemanticTypes'], fix_arguments(arguments),
                        target_hyperparams, self.resolver)
                    targets_data_ref = 'steps.{}.produce'.format(len(new_pipeline.steps) - 1)

                    arguments = {'inputs': attributes_data_ref, 'outputs': targets_data_ref}
                    hyperparams = []
                    if 'return_result' in primitive.metadata.get_hyperparams().defaults():
                        hyperparams.append(('return_result', ArgumentType.VALUE, 'replace'))
                    add_primitive_step_to_pipeline(new_pipeline, primitive, fix_arguments(arguments), hyperparams, self.resolver)
                    add_construct_predictions(new_pipeline, self.preprocessing.dataframe_reference, self.resolver)
            except Exception as e:
                print(e)
                failed = True
        else:
            try:
                print('=====task_description semi: {} estimator: {} ====='.format(self.task_description['semi'], primitive))
                arguments = {'inputs': data_ref}
                attributes_hyperparams = [('semantic_types', ArgumentType.VALUE, [DATA_TYPES['attribute']])]
                target_hyperparams = [('semantic_types', ArgumentType.VALUE, [DATA_TYPES['target']])]
                add_primitive_step_to_pipeline(
                    new_pipeline, LOADED_PRIMITIVES['ExtractColumnsBySemanticTypes'], fix_arguments(arguments),
                    attributes_hyperparams, self.resolver)
                attributes_data_ref = 'steps.{}.produce'.format(len(new_pipeline.steps) - 1)

                add_primitive_step_to_pipeline(
                    new_pipeline, LOADED_PRIMITIVES['ExtractColumnsBySemanticTypes'], fix_arguments(arguments),
                    target_hyperparams, self.resolver)
                targets_data_ref = 'steps.{}.produce'.format(len(new_pipeline.steps) - 1)

                arguments = {'inputs': attributes_data_ref, 'outputs': targets_data_ref}
                hyperparams = [('blackbox', ArgumentType.VALUE, primitive)]
                add_primitive_step_to_pipeline(new_pipeline, LOADED_PRIMITIVES['SemiClassification'],
                                               fix_arguments(arguments), hyperparams,self.resolver)
                add_construct_predictions(new_pipeline, self.preprocessing.dataframe_reference, self.resolver)
            except Exception as e:
                print(e)
                failed = True

        if failed:
            return None
        else:
            return new_pipeline


