from d3m import index
from d3m.metadata import base as metadata_base
from d3m.metadata.base import ArgumentType
from d3m.metadata.problem import TaskKeyword

from axolotl.predefined_pipelines.base_preprocessor import Preprocessor
from axolotl.utils import pipeline as pipeline_utils, schemas as schemas_utils


def get_preprocessor(input_data, problem, treatment):
    metadata = input_data.metadata
    task_description = schemas_utils.get_task_description(problem['problem']['task_keywords'])
    task_type = task_description['task_type']
    semi = task_description['semi']
    data_types = task_description['data_types']
    task = pipeline_utils.infer_primitive_family(task_type=task_type, data_types=data_types, is_semi=semi)
    main_resource = pipeline_utils.get_tabular_resource_id(dataset=input_data)

    # Loading primitives
    primitives = {
        'DatasetToDataFrame': 'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
        'ColumnParser': 'd3m.primitives.data_transformation.column_parser.Common',
        'ExtractColumnsBySemanticTypes': 'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',
        'Denormalize': 'd3m.primitives.data_transformation.denormalize.Common',
        'Imputer': 'd3m.primitives.data_cleaning.imputer.SKlearn',
        'SimpleProfiler': 'd3m.primitives.schema_discovery.profiler.Common',
        'TextEncoder': 'd3m.primitives.data_transformation.encoder.DistilTextEncoder',
    }
    loaded_primitives = dict()

    try:
        for primitive_name in primitives.keys():
            loaded_primitives[primitive_name] = index.get_primitive(primitives[primitive_name])
    except Exception as e:
        print("Cannot load primitive {}".format(e))

    candidates = []
    for preprocessor in preprocessors:
        if preprocessor.check_task_treatment(task, treatment) \
                and preprocessor.check_expected_data_types(data_types) \
                and preprocessor.check_unsupported_data_types(data_types):
            candidates.append(preprocessor(metadata, main_resource, data_types, loaded_primitives, problem))
    if not candidates:
        candidates.append(TabularPreprocessor(metadata, main_resource, data_types, loaded_primitives))
    return candidates


class TimeSeriesTabularPreprocessor(Preprocessor, task=metadata_base.PrimitiveFamily.TIME_SERIES_CLASSIFICATION.name,
                                    treatment=metadata_base.PrimitiveFamily.CLASSIFICATION.name,
                                    expected_data_types=None,
                                    unsupported_data_types={TaskKeyword.TABULAR, TaskKeyword.RELATIONAL}):
    def _generate_pipeline(self):
        time_series_featurization_primitive = self.get_primitive(
            'd3m.primitives.feature_extraction.random_projection_timeseries_featurization.DSBOX'
        )
        time_series_to_list_primitive = self.get_primitive(
            'd3m.primitives.data_preprocessing.time_series_to_list.DSBOX'
        )

        # denormalize -> dataset_to_df
        self.common_boilerplate()
        dataset_to_dataframe_step = self.d2d_step

        # timeseries_to_list
        timeseries_tolist_step = self.add_primitive_to_pipeline(
            primitive=time_series_to_list_primitive,
            attributes=dataset_to_dataframe_step,
        )
        # timeseries_featurization
        timeseries_featurization_step = self.add_primitive_to_pipeline(
            primitive=time_series_featurization_primitive,
            attributes=timeseries_tolist_step,
        )
        # extract_col_by_semantic
        attr_step = self.add_extract_col_by_semantic_types_step(
            timeseries_featurization_step,
            ['https://metadata.datadrivendiscovery.org/types/Attribute']
        )
        # extract_col_by_semantic
        targ_step = self.add_extract_col_by_semantic_types_step(
            dataset_to_dataframe_step,
            ['https://metadata.datadrivendiscovery.org/types/TrueTarget']
        )
        self.set_attribute_step(attr_step)
        self.set_target_step(targ_step)


class TimeSeriesPreprocessor(Preprocessor, task=metadata_base.PrimitiveFamily.TIME_SERIES_CLASSIFICATION.name,
                             treatment=metadata_base.PrimitiveFamily.TIME_SERIES_CLASSIFICATION.name,
                             expected_data_types=None,
                             unsupported_data_types={TaskKeyword.TABULAR, TaskKeyword.RELATIONAL}):
    def _generate_pipeline(self):
        time_series_formatter_primitive = self.get_primitive(
            'd3m.primitives.data_preprocessing.data_cleaning.DistilTimeSeriesFormatter'
        )
        ts_formatter = self.add_primitive_to_pipeline(
            primitive=time_series_formatter_primitive,
            attributes=self.start_resource
        )

        dtd_step = self.add_dataset_to_dataframe_step(ts_formatter)
        dtd_without_ts_format = self.add_dataset_to_dataframe_step(self.start_resource)

        extract_target_step = self.add_extract_col_by_semantic_types_step(
            dtd_without_ts_format,
            ['https://metadata.datadrivendiscovery.org/types/TrueTarget']
        )
        target_column_parser_step = self.add_column_parser_step(
            extract_target_step,
            to_parse=[
                "http://schema.org/Boolean",
                "http://schema.org/Integer",
                "http://schema.org/Float",
                "https://metadata.datadrivendiscovery.org/types/FloatVector"
            ]
        )
        self.set_d2d_step(dtd_without_ts_format)
        self.set_attribute_step(dtd_step)
        self.set_target_step(target_column_parser_step)


class TimeSeriesForecastingTabularPreprocessor(Preprocessor,
                                               task=metadata_base.PrimitiveFamily.TIME_SERIES_FORECASTING.name,
                                               treatment=metadata_base.PrimitiveFamily.TIME_SERIES_FORECASTING.name,
                                               expected_data_types={TaskKeyword.GROUPED.name}):
    # TODO: Pipeline will fail for integer target because simple_profiler profiles it as Categorical data,
    #  not Float or Integer.
    def _generate_pipeline(self):
        grouping_compose_primitive = self.get_primitive(
            'd3m.primitives.data_transformation.grouping_field_compose.Common'
        )

        self.common_boilerplate()

        # Do not parse categorical data or GroupingCompose will fail.
        column_parser = self.add_column_parser_step(
            self.d2d_step, [
                "http://schema.org/DateTime",
                "http://schema.org/Boolean",
                "http://schema.org/Integer",
                "http://schema.org/Float",
                "https://metadata.datadrivendiscovery.org/types/FloatVector"
            ]
        )

        attribute_step = self.add_extract_col_by_semantic_types_step(
            column_parser, ['https://metadata.datadrivendiscovery.org/types/Attribute']
        )

        grouping = self.add_primitive_to_pipeline(
            primitive=grouping_compose_primitive,
            attributes=attribute_step
        )

        target_step = self.add_extract_col_by_semantic_types_step(column_parser, [
            'https://metadata.datadrivendiscovery.org/types/TrueTarget'
        ])
        self.set_attribute_step(grouping)
        self.set_target_step(target_step)


class AudioPreprocessor(Preprocessor, task=metadata_base.PrimitiveFamily.DIGITAL_SIGNAL_PROCESSING.name,
                        treatment=None,
                        expected_data_types=None):

    def _generate_pipeline(self):
        audio_reader_primitive = self.get_primitive(
            'd3m.primitives.data_preprocessing.audio_reader.DistilAudioDatasetLoader'
        )
        audio_feature_extraction_primitive = self.get_primitive(
            'd3m.primitives.feature_extraction.audio_transfer.DistilAudioTransfer'
        )
        audio_reader = self.add_primitive_to_pipeline(
            primitive=audio_reader_primitive,
            attributes=self.start_resource,
            produce_collection=True
        )
        column_parser = self.add_column_parser_step(
            data_reference=audio_reader,
            to_parse=[
                'http://schema.org/Boolean',
                'http://schema.org/Integer',
                'http://schema.org/Float',
                'https://metadata.datadrivendiscovery.org/types/FloatVector'
            ]
        )
        audio_feature = self.add_primitive_to_pipeline(
            primitive=audio_feature_extraction_primitive,
            attributes='steps.{}.produce_collection'.format(audio_reader.index),
        )
        target_step = self.add_extract_col_by_semantic_types_step(
            column_parser,
            [
                'https://metadata.datadrivendiscovery.org/types/TrueTarget',
                'https://metadata.datadrivendiscovery.org/types/SuggestedTarget'
            ]
        )
        self.set_d2d_step(audio_reader)
        self.set_attribute_step(audio_feature)
        self.set_target_step(target_step)


class ImageDataFramePreprocessor(Preprocessor, task=metadata_base.PrimitiveFamily.DIGITAL_IMAGE_PROCESSING.name,
                                 treatment=None,
                                 expected_data_types={TaskKeyword.IMAGE.name}):
    def _generate_pipeline(self):
        image_reader_primitive = self.get_primitive('d3m.primitives.data_preprocessing.image_reader.Common')
        image_feature_extraction_primitive = self.get_primitive(
            'd3m.primitives.feature_extraction.image_transfer.DistilImageTransfer')

        self.common_boilerplate()
        dataset_to_dataframe_step = self.d2d_step

        image_reader = self.add_primitive_to_pipeline(
            primitive=image_reader_primitive,
            attributes=dataset_to_dataframe_step,
            hyperparameters=[('return_result', ArgumentType.VALUE, 'replace')]
        )
        column_parser = self.add_column_parser_step(
            data_reference=image_reader,
            to_parse=[
                'http://schema.org/Boolean',
                'http://schema.org/Integer',
                'http://schema.org/Float',
                'https://metadata.datadrivendiscovery.org/types/FloatVector'
            ]
        )
        image_feature_extraction = self.add_primitive_to_pipeline(
            primitive=image_feature_extraction_primitive,
            attributes=column_parser
        )
        target_step = self.add_extract_col_by_semantic_types_step(
            data_reference=dataset_to_dataframe_step,
            target_semantic_types=['https://metadata.datadrivendiscovery.org/types/TrueTarget'],
        )
        self.set_attribute_step(image_feature_extraction)
        self.set_target_step(target_step)


class ImageTensorPreprocessor(Preprocessor, task=metadata_base.PrimitiveFamily.DIGITAL_IMAGE_PROCESSING.name,
                              treatment=None,
                              expected_data_types={TaskKeyword.IMAGE.name}):
    def _generate_pipeline(self):
        dataframe_to_tensor_primitive = self.get_primitive(
            'd3m.primitives.data_preprocessing.dataframe_to_tensor.DSBOX'
        )
        resnet50_featurizer_primitive = self.get_primitive(
            'd3m.primitives.feature_extraction.resnet50_image_feature.DSBOX'
        )

        self.common_boilerplate()
        dataset_to_dataframe_step = self.d2d_step

        dataframe_to_tensor = self.add_primitive_to_pipeline(
            primitive=dataframe_to_tensor_primitive,
            attributes=dataset_to_dataframe_step,
            hyperparameters=[('return_result', ArgumentType.VALUE, 'replace')]
        )
        resnet50_featurizer = self.add_primitive_to_pipeline(
            primitive=resnet50_featurizer_primitive,
            attributes=dataframe_to_tensor,
            hyperparameters=[('return_result', ArgumentType.VALUE, 'replace')]
        )
        target_step = self.add_extract_col_by_semantic_types_step(
            dataset_to_dataframe_step,
            ['https://metadata.datadrivendiscovery.org/types/TrueTarget']
        )
        self.set_attribute_step(resnet50_featurizer)
        self.set_target_step(target_step)


class TabularPreprocessor(Preprocessor, task=None, treatment=None, expected_data_types={TaskKeyword.TABULAR.name}):
    def _generate_pipeline(self):
        return self.tabular_common()


class CollaborativeFilteringPreprocessor(Preprocessor, task=metadata_base.PrimitiveFamily.COLLABORATIVE_FILTERING.name,
                                         treatment=None,
                                         expected_data_types=None):
    def _generate_pipeline(self):
        return self.tabular_common(target_at_column_parser=True)


class TextPreprocessor(Preprocessor, task=None, treatment=None,
                       expected_data_types={TaskKeyword.TEXT}):
    def _generate_pipeline(self):
        text_reader_primitive = self.get_primitive('d3m.primitives.data_preprocessing.text_reader.Common')

        self.common_boilerplate()

        # Simple preprocessor
        attributes, targets = self.base()

        text_reader_step = self.add_primitive_to_pipeline(
            primitive=text_reader_primitive,
            attributes=attributes,
            hyperparameters=[('return_result', ArgumentType.VALUE, 'replace')]
        )
        imputer = self.add_imputer(text_reader_step)
        attributes = self.add_simple_text_handler(imputer, targets)
        self.set_attribute_step(attributes)
        self.set_target_step(targets)


class TextSent2VecPreprocessor(Preprocessor, task=None, treatment=None, expected_data_types={TaskKeyword.TEXT.name}):
    def _generate_pipeline(self):
        sent2_vec_primitive =self.get_primitive('d3m.primitives.feature_extraction.nk_sent2vec.Sent2Vec')

        self.common_boilerplate()

        # Simple preprocessor
        attributes, targets = self.base()

        sent2vec = self.add_primitive_to_pipeline(
            primitive=sent2_vec_primitive,
            attributes=attributes,
        )

        imputer = self.add_imputer(sent2vec)
        self.set_attribute_step(imputer)
        self.set_target_step(targets)


class LupiPreprocessor(Preprocessor, task=None, treatment=None,
                       expected_data_types={TaskKeyword.LUPI.name}):
    def _generate_pipeline(self):
        self.common_boilerplate()

        privileged_column_indices = [info['column_index'] for info in self.problem['inputs'][0]['privileged_data']]
        attributes, targets = self.base(exclude_attr_columns=privileged_column_indices)

        imputer = self.add_imputer(attributes)
        self.set_attribute_step(imputer)
        self.set_target_step(targets)


preprocessors = [
    # TODO DSBOX installation has error
    # TimeSeriesTabularPreprocessor,
    TimeSeriesPreprocessor,
    TimeSeriesForecastingTabularPreprocessor,
    AudioPreprocessor,
    ImageDataFramePreprocessor,
    # TODO DSBOX installation has error
    # ImageTensorPreprocessor,
    CollaborativeFilteringPreprocessor,
    TextSent2VecPreprocessor,
    TextPreprocessor,
    LupiPreprocessor
]
