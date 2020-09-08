import typing

import abc
from d3m import index
from d3m.metadata.base import Context, ArgumentType
from d3m.metadata.pipeline import Pipeline, Resolver, PrimitiveStep

from axolotl.utils import pipeline as pipeline_utils

DEFAULT_OUTPUT = '.'


class Preprocessor(abc.ABC):
    task: str
    treatment: str
    expected_data_types: set
    unsupported_data_types: set
    semi: bool

    def __init__(self, metadata, main_resource, data_types, loaded_primitives, problem=None, start_resource='inputs.0'):
        self.metadata = metadata
        self.main_resource = main_resource
        self.data_types = data_types
        self.loaded_primitives = loaded_primitives
        self.start_resource = start_resource
        self.problem = problem
        # Creating pipeline
        pipeline_description = Pipeline(context=Context.TESTING)
        pipeline_description.add_input(name='inputs')
        self.pipeline = pipeline_description
        self.d2d_step = None
        self.attr_step = None
        self.targ_step = None
        self._generate_pipeline()

    def __init_subclass__(cls, task: str, treatment: str, expected_data_types: set, **kargs):
        cls.task = task
        cls.treatment = treatment
        cls.expected_data_types = expected_data_types
        cls.unsupported_data_types = kargs['unsupported_data_types'] if 'unsupported_data_types' in kargs else None
        cls.semi = kargs['semi'] if 'semi' in kargs else False

    @classmethod
    def check_task_treatment(cls, task, treatment):
        if not cls.task:
            return True
        if not cls.treatment:
            return cls.task == task
        return cls.task == task and cls.treatment == treatment

    @classmethod
    def check_expected_data_types(cls, data_types):
        if not cls.expected_data_types:
            return True
        return any(data_type in cls.expected_data_types for data_type in data_types)

    @classmethod
    def check_unsupported_data_types(cls, data_types):
        if not cls.unsupported_data_types:
            return True
        return not any(data_type in cls.unsupported_data_types for data_type in data_types)

    @property
    def pipeline_description(self) -> Pipeline:
        return self.pipeline

    @property
    def dataset_to_dataframe_step(self) -> typing.Optional[str]:
        return self.get_output_str(self.d2d_step) if self.d2d_step else None

    @property
    def attributes(self) -> typing.Optional[str]:
        return self.get_output_str(self.attr_step) if self.attr_step else None

    @property
    def targets(self) -> typing.Optional[str]:
        return self.get_output_str(self.targ_step) if self.targ_step else None

    @property
    def resolver(self) -> Resolver:
        return pipeline_utils.BlackListResolver()

    @abc.abstractmethod
    def _generate_pipeline(self):
        raise NotImplementedError()

    @property
    def gpu_budget(self) -> float:
        return 0

    def get_primitive(self, name):
        primitive = index.get_primitive(name)
        self.download_static_files(primitive)
        return primitive

    def common_boilerplate(self):
        """
        This boilerplate provides the basic init pipline that contains denormalize and dataset_to_dataframe.

        Arguments
        ---------
        include_dataset_to_dataframe: bool
            Whether to include dataset_to_dataframe step.
        include_simple_profiler:  bool
            whether or not to include simple profiler
        """
        metadata = self.metadata
        main_resource_id = self.main_resource
        start_resource = self.start_resource

        # if there is more that one resource we denormalize
        if len(metadata.get_elements(())) > 1:
            start_resource = self.add_denormalize_step(start_resource, main_resource_id)

        # Finally we transfer to a dataframe.
        dtd_step = self.add_dataset_to_dataframe_step(start_resource)

        simple_profiler_step = self.add_primitive_to_pipeline(
            primitive=self.loaded_primitives['SimpleProfiler'],
            attributes=dtd_step,
            hyperparameters=[
                ('categorical_max_ratio_distinct_values', ArgumentType.VALUE, 1),
                ('categorical_max_absolute_distinct_values', ArgumentType.VALUE, None)
            ]
        )
        self.set_d2d_step(simple_profiler_step)

    def tabular_common(self, target_at_column_parser=False):
        self.common_boilerplate()

        # Simple preprocessor
        attributes, targets = self.base(target_at_column_parser=target_at_column_parser)

        # Adding Imputer
        imputer = self.add_imputer(attributes=attributes)

        attributes = self.add_simple_text_handler(imputer, targets)
        self.set_attribute_step(attributes)
        self.set_target_step(targets)

    def base(self, target_at_column_parser=False, exclude_attr_columns=None):
        dataset_dataframe_step_pos = self.d2d_step

        # Step 2: ColumnParser
        column_parser_step = self.add_column_parser_step(data_reference=dataset_dataframe_step_pos)

        # Step 3: ExtractAttributes
        attributes_step = self.add_extract_col_by_semantic_types_step(
            column_parser_step,
            ['https://metadata.datadrivendiscovery.org/types/Attribute'],
            exclude_attr_columns
        )
        target_source = column_parser_step if target_at_column_parser else dataset_dataframe_step_pos

        # Step 4: ExtractTargets
        targets_step = self.add_extract_col_by_semantic_types_step(
            target_source,
            ['https://metadata.datadrivendiscovery.org/types/TrueTarget']
        )
        return attributes_step, targets_step

    def add_imputer(self, attributes):
        # SklearnImputer
        primitive = self.loaded_primitives['Imputer']
        configuration = \
            primitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams'].configuration
        hyperparameters = []
        if 'return_result' in configuration:
            hyperparameters.append(
                ('return_result', ArgumentType.VALUE, 'replace')
            )
        if 'use_semantic_types' in configuration:
            hyperparameters.append(
                ('use_semantic_types', ArgumentType.VALUE, True)
            )
        hyperparameters.append(
            ('error_on_no_input', ArgumentType.VALUE, False)
        )
        imputer = self.add_primitive_to_pipeline(
            primitive=primitive,
            attributes=attributes,
            hyperparameters=hyperparameters
        )
        return imputer

    def add_extract_col_by_semantic_types_step(self, data_reference, target_semantic_types, exclude_columns=None):
        if exclude_columns:
            hyperparameters = [
                ('exclude_columns', ArgumentType.VALUE, exclude_columns),
                ('semantic_types', ArgumentType.VALUE, target_semantic_types)
            ]
        else:
            hyperparameters = [
                ('semantic_types', ArgumentType.VALUE, target_semantic_types)
            ]
        step = self.add_primitive_to_pipeline(
            primitive=self.loaded_primitives['ExtractColumnsBySemanticTypes'],
            attributes=data_reference,
            hyperparameters=hyperparameters
        )
        return step

    def add_denormalize_step(self, start_resource, data):
        denormalize_step = self.add_primitive_to_pipeline(
            primitive=self.loaded_primitives['Denormalize'],
            attributes=start_resource,
            hyperparameters=[
                ('starting_resource', ArgumentType.VALUE, data)
            ]
        )
        return denormalize_step

    def add_dataset_to_dataframe_step(self, start_resource):
        d2d_step = self.add_primitive_to_pipeline(
            primitive=self.loaded_primitives['DatasetToDataFrame'],
            attributes=start_resource
        )
        return d2d_step

    def add_column_parser_step(self, data_reference, to_parse=None):
        if to_parse:
            hyperparameters = [
                ('parse_semantic_types', ArgumentType.VALUE, to_parse)
            ]
        else:
            hyperparameters = []
        column_parser = self.add_primitive_to_pipeline(
            primitive=self.loaded_primitives['ColumnParser'],
            attributes=data_reference,
            hyperparameters=hyperparameters
        )
        return column_parser

    def add_simple_text_handler(self, attributes, targets):
        text_encoder = self.add_primitive_to_pipeline(
            primitive=self.loaded_primitives['TextEncoder'],
            attributes=attributes,
            hyperparameters=[
                ('encoder_type', ArgumentType.VALUE, 'tfidf')
            ],
            targets=targets
        )
        return text_encoder

    def download_static_files(self, primitive):
        primitive_metadata = primitive.metadata.query()
        output = DEFAULT_OUTPUT
        redownload = False
        index.download_files(primitive_metadata, output, redownload)

    def add_primitive_to_pipeline(self, primitive, attributes, hyperparameters=[], targets=None,
                                  produce_collection=False):
        inputs_ref = attributes if isinstance(attributes, str) else self.get_output_str(attributes)
        step = PrimitiveStep(primitive=primitive, resolver=self.resolver)
        step.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=inputs_ref)
        for hyperparam in hyperparameters:
            name, argument_type, data = hyperparam
            step.add_hyperparameter(name=name, argument_type=argument_type, data=data)
        if targets:
            outputs_ref = targets if isinstance(targets, str) else self.get_output_str(targets)
            step.add_argument(name='outputs', argument_type=ArgumentType.CONTAINER, data_reference=outputs_ref)
        step.add_output('produce')
        if produce_collection:
            step.add_output('produce_collection')
        self.pipeline.add_step(step)
        return step

    def get_output_str(self, step):
        return pipeline_utils.int_to_step(step.index)

    def set_attribute_step(self, attributes):
        self.attr_step = attributes

    def set_target_step(self, targets):
        self.targ_step = targets

    def set_d2d_step(self, dataset_2_dataframe):
        self.d2d_step = dataset_2_dataframe