import argparse
import pathlib

import shutil
import sys

import os
import unittest
from pprint import pprint
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, PROJECT_ROOT)

# from TimeSeriesD3MWrappers.primitives.classification_knn import Kanine
from d3m import container as container_module, index
from d3m.metadata import base as metadata_base, problem as problem_module
from d3m.metadata.base import ArgumentType
from d3m.metadata.pipeline import PrimitiveStep
from d3m.metadata.problem import TaskKeyword
from d3m.runtime import Runtime
from sklearn_wrap.SKLogisticRegression import SKLogisticRegression

from axolotl.predefined_pipelines import preprocessor
from axolotl.utils import pipeline as pipeline_utils


def run_pipeline(pipeline_description, data, volume_dir='/volumes'):
    runtime = Runtime(pipeline=pipeline_description, context=metadata_base.Context.TESTING, volumes_dir=volume_dir)
    fit_result = runtime.fit([data])
    return fit_result


def add_classifier(pipeline_description, dataset_to_dataframe_step, attributes, targets):
    lr = PrimitiveStep(primitive=SKLogisticRegression)
    lr.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER,
                    data_reference=attributes)
    lr.add_argument(name='outputs', argument_type=ArgumentType.CONTAINER,
                    data_reference=targets)
    lr.add_output('produce')
    pipeline_description.add_step(lr)

    construct_pred = PrimitiveStep(
        primitive=index.get_primitive('d3m.primitives.data_transformation.construct_predictions.Common'))
    construct_pred.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER,
                                data_reference=pipeline_utils.int_to_step(lr.index))
    construct_pred.add_argument(name='reference', argument_type=ArgumentType.CONTAINER,
                                data_reference=dataset_to_dataframe_step)
    construct_pred.add_output('produce')
    pipeline_description.add_step(construct_pred)
    # Final Output
    pipeline_description.add_output(name='output predictions',
                                    data_reference=pipeline_utils.int_to_step(construct_pred.index))


# def add_time_series_specific_classifier(pipeline_description, attributes, targets):
#     k = PrimitiveStep(primitive=Kanine)
#     k.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER,
#                    data_reference=attributes)
#     k.add_argument(name='outputs', argument_type=ArgumentType.CONTAINER,
#                    data_reference=targets)
#     k.add_output('produce')
#     pipeline_description.add_step(k)
#     pipeline_description.add_output(name='output predictions',
#                                     data_reference=pipeline_utils.int_to_step(k.index))
#     return k


def _remove_volatile(target_pipe, predef_pipe):
    target_pipe = target_pipe.to_json_structure()
    for step in target_pipe['steps']:
        del step['primitive']['digest']
    subset = {k: v for k, v in target_pipe.items() if k in predef_pipe}
    return subset


class TestPreprocessor(unittest.TestCase):
    time_series_data: container_module.Dataset = None
    temp_dir: str = os.path.join(os.path.dirname(__file__), 'temp')

    @classmethod
    def setUpClass(cls) -> None:
        cls.maxDiff = None
        cls.test_data = os.path.join(PROJECT_ROOT, 'tests', 'data')
        # cls.time_series_data = datasets.get('timeseries_dataset_2')
        # cls.tabular_classification_data = datasets.get('iris_dataset_1')
        # cls.image_data = datasets.get('image_dataset_1')
        # cls.audio_dataset = datasets.get('audio_dataset_1')

    @classmethod
    def tearDownClass(cls):
        for dir_name in (
                # cls.test_dir + 'solutions',
                # cls.test_dir + 'fitted_solutions',
        ):
            if os.path.exists(dir_name):
                shutil.rmtree(dir_name)

    # def test_timeseries_tabular(self):
    #     pp = preprocessor.get_preprocessor(task=metadata_base.PrimitiveFamily.TIME_SERIES_CLASSIFICATION.name,
    #                                        treatment=metadata_base.PrimitiveFamily.CLASSIFICATION.name,
    #                                        data_types=[TaskKeyword.TIME_SERIES], semi=False,
    #                                        inputs_metadata=self.time_series_data.metadata, problem=None,
    #                                        main_resource='learningData')[0]
    #     add_classifier(pp.pipeline_description, pp.dataset_to_dataframe_step, pp.attributes, pp.targets)
    #     result = run_pipeline(pp.pipeline_description, self.time_series_data)
    #     result.check_success()
    #     self.assertEqual(result.error, None)
    #
    # def test_timeseries_specific(self):
    #     pp = preprocessor.get_preprocessor(task=metadata_base.PrimitiveFamily.TIME_SERIES_CLASSIFICATION.name,
    #                                        treatment=metadata_base.PrimitiveFamily.TIME_SERIES_CLASSIFICATION.name,
    #                                        data_types=[TaskKeyword.TIME_SERIES], semi=False,
    #                                        inputs_metadata=self.time_series_data.metadata, problem=None,
    #                                        main_resource='learningData')[0]
    #
    #     add_time_series_specific_classifier(pp.pipeline_description, pp.attributes, pp.targets)
    #     result = run_pipeline(pp.pipeline_description, self.time_series_data)
    #     result.check_success()
    #     self.assertEqual(result.error, None)

    def test_TabularPreprocessor(self):
        dataset_name = 'iris_dataset_1'
        problem = self.__get_problem(dataset_name)
        dataset = self.__get_dataset(dataset_name)
        pp = preprocessor.get_preprocessor(
            input_data=dataset,
            problem=problem,
            treatment=metadata_base.PrimitiveFamily.CLASSIFICATION.name,
        )[0]
        add_classifier(pp.pipeline_description, pp.dataset_to_dataframe_step, pp.attributes, pp.targets)
        result = run_pipeline(pp.pipeline_description, dataset)
        # pprint(pp.pipeline_description.to_json_structure())
        result.check_success()
        self.assertEqual(result.error, None)

    # def test_image_tensor(self):
    #     pp = preprocessor.get_preprocessor(task=metadata_base.PrimitiveFamily.DIGITAL_IMAGE_PROCESSING.name,
    #                                        treatment=metadata_base.PrimitiveFamily.DIGITAL_IMAGE_PROCESSING.name,
    #                                        data_types=[TaskKeyword.IMAGE], semi=False,
    #                                        inputs_metadata=self.image_data.metadata, problem=None,
    #                                        main_resource='learningData')[0]
    #     add_classifier(pp.pipeline_description, pp.dataset_to_dataframe_step, pp.attributes, pp.targets)
    #     # pprint(pp.pipeline_description.to_json_structure())
    #     result = run_pipeline(pp.pipeline_description, self.image_data)
    #     result.check_success()
    #     self.assertEqual(result.error, None)

    # TODO update static files on the CI
    # def test_ImageDataFramePreprocessor(self):
    #     dataset_name = 'image_dataset_2'
    #     problem = self.__get_problem(dataset_name)
    #     dataset = self.__get_dataset(dataset_name)
    #     problem['problem']['task_keywords'].append(TaskKeyword.IMAGE)
    #     pp = preprocessor.get_preprocessor(
    #         input_data=dataset,
    #         problem=problem,
    #         treatment=metadata_base.PrimitiveFamily.CLASSIFICATION.name,
    #     )[0]
    #     volume = os.path.join(PROJECT_ROOT, 'tests')
    #     add_classifier(pp.pipeline_description, pp.dataset_to_dataframe_step, pp.attributes, pp.targets)
    #     # pprint(pp.pipeline_description.to_json_structure())
    #     result = run_pipeline(pp.pipeline_description, dataset, volume_dir=volume)
    #     result.check_success()
    #     self.assertEqual(result.error, None)

    # TODO need to augment text_dataset_1
    # def test_TextPreprocessor(self):
    #     dataset_name = 'text_dataset_1'
    #     # No text_problem_1, so I use iris_problem instead
    #     problem = self.__get_problem('iris_problem_1')
    #     problem['problem']['task_keywords'] = [TaskKeyword.CLASSIFICATION, TaskKeyword.TEXT]
    #     dataset = self.__get_dataset(dataset_name)
    #     # TextSent2VecPreprocessor, TextPreprocessor
    #     pp = preprocessor.get_preprocessor(
    #         input_data=dataset,
    #         problem=problem,
    #         treatment=metadata_base.PrimitiveFamily.CLASSIFICATION.name,
    #     )[-1]
    #     add_classifier(pp.pipeline_description, pp.dataset_to_dataframe_step, pp.attributes, pp.targets)
    #     pprint(pp.pipeline_description.to_json_structure())
    #     result = run_pipeline(pp.pipeline_description, dataset)
    #     result.check_success()
    #     self.assertEqual(result.error, None)

    # def test_timeseries_forecasting_tabular(self):
    #     dataset = datasets.get('timeseries_dataset_1')
    #     pp = preprocessor.get_preprocessor(task=metadata_base.PrimitiveFamily.TIME_SERIES_FORECASTING.name,
    #                                        treatment=metadata_base.PrimitiveFamily.TIME_SERIES_FORECASTING.name,
    #                                        data_types=[TaskKeyword.TIME_SERIES.name, TaskKeyword.TABULAR.name],
    #                                        semi=False, inputs_metadata=dataset.metadata, problem=None,
    #                                        main_resource='learningData')[0]
    #
    #     add_classifier(pp.pipeline_description, pp.dataset_to_dataframe_step, pp.attributes, pp.targets)
    #     result = run_pipeline(pp.pipeline_description, dataset)
    #     pprint(pp.pipeline_description.to_json_structure())
    #     result.check_success()
    #     self.assertEqual(result.error, None)

    # TODO need to update tests/data/datasets/audio_dataset_1
    # def test_AudioPreprocessor(self):
    #     dataset_name = 'audio_dataset_1'
    #     # No audio_problem_1, so I use iris_problem instead
    #     problem = self.__get_problem('iris_problem_1')
    #     problem['problem']['task_keywords'] = [TaskKeyword.AUDIO, TaskKeyword.VIDEO]
    #     dataset = self.__get_dataset(dataset_name)
    #     pp = preprocessor.get_preprocessor(
    #         input_data=dataset,
    #         problem=problem,
    #         treatment=metadata_base.PrimitiveFamily.DIGITAL_SIGNAL_PROCESSING.name,
    #     )[-1]
    #     volume = os.path.join(PROJECT_ROOT, 'tests')
    #     add_classifier(pp.pipeline_description, pp.dataset_to_dataframe_step, pp.attributes, pp.targets)
    #     pprint(pp.pipeline_description.to_json_structure())
    #     result = run_pipeline(pp.pipeline_description, dataset, volume_dir=volume)
    #     result.check_success()
    #
    #     self.assertEqual(result.error, None)

    def __get_uri(self, path):
        return pathlib.Path(os.path.abspath(path)).as_uri()

    def __get_problem(self, dataset_name):
        problem_path = os.path.join(
            self.test_data, 'problems', dataset_name.replace('dataset', 'problem'), 'problemDoc.json')
        problem_uri = self.__get_uri(problem_path)
        problem = problem_module.Problem.load(problem_uri)
        return problem

    def __get_dataset(self, dataset_name):
        dataset_path = os.path.join(
            self.test_data, 'datasets', dataset_name, 'datasetDoc.json')
        dataset_uri = self.__get_uri(dataset_path)
        dataset = container_module.dataset.get_dataset(dataset_uri)
        return dataset


# if __name__ == '__main__':
#     suite = unittest.TestSuite()
#     for test_case in (
#         # 'test_ImageDataFramePreprocessor',
#         'test_TabularPreprocessor',
#         # 'test_AudioPreprocessor',
#         # 'test_TextPreprocessor',
#
#     ):
#         suite.addTest(TestPreprocessor(test_case))
#     unittest.TextTestRunner(verbosity=2).run(suite)
