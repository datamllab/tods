import json
import uuid

from d3m.metadata.pipeline import Pipeline

from axolotl.algorithms.base import PipelineSearchBase
from axolotl.utils import schemas as schemas_utils, pipeline as pipeline_utils


def dummy_ranking_function(pipeline_result):
    if pipeline_result.status == 'COMPLETED':
        summarize_performance = schemas_utils.summarize_performance_metrics(pipeline_result.scores)
        rank = schemas_utils.compute_rank(summarize_performance)
        pipeline_result.rank = rank
    return pipeline_result


class DummySearch(PipelineSearchBase):
    def __init__(self, problem_description, backend, *, primitives_blocklist=None, ranking_function=None):
        super().__init__(problem_description=problem_description, backend=backend,
                         primitives_blocklist=primitives_blocklist, ranking_function=ranking_function)
        if self.ranking_function is None:
            self.ranking_function = dummy_ranking_function
        self.task_description = schemas_utils.get_task_description(self.problem_description['problem']['task_keywords'])

        self.available_pipelines = self._return_pipelines(
            self.task_description['task_type'], self.task_description['task_subtype'], self.task_description['data_types'])

        # TODO update this to be defined on problem/metrics terms
        self.data_preparation_pipeline = schemas_utils.get_splitting_pipeline("TRAINING_DATA")
        self.metrics = self.problem_description['problem']['performance_metrics']

        self.scoring_pipeline = schemas_utils.get_scoring_pipeline()
        self.data_preparation_params = schemas_utils.DATA_PREPARATION_PARAMS['no_split']

        self.offset = 10
        self.current_pipeline_index = 0

    def _search(self, time_left):
        pipelines_to_eval = self.available_pipelines[self.current_pipeline_index: self.current_pipeline_index+self.offset]
        self.current_pipeline_index += self.offset
        pipeline_results = self.backend.evaluate_pipelines(
            problem_description=self.problem_description, pipelines=pipelines_to_eval, input_data=self.input_data,
            metrics=self.metrics, data_preparation_pipeline=self.data_preparation_pipeline,
            scoring_pipeline=self.scoring_pipeline, data_preparation_params=self.data_preparation_params)

        return [self.ranking_function(pipeline_result) for pipeline_result in pipeline_results]

    def _return_pipelines(self, task_type, task_subtype, data_type):
        """
        A function that return predefined pipelines given a task type.

        Returns
        -------
            A predefined pipelines if there are pipelines left, also if there is template
            returns the new pipeline with the template.

        """
        # TODO incorporate task_subtype and data_type for future problems
        with open(schemas_utils.PIPELINES_DB_DIR) as file:
            possible_pipelines_dict = json.load(file)

        if task_type not in possible_pipelines_dict:
            self.pipeline_left = False
            return None

        possible_pipelines_dict = possible_pipelines_dict[task_type]

        if not possible_pipelines_dict:
            return []

        possible_pipelines = []
        for pipeline_dict in possible_pipelines_dict:
            try:
                pipeline = pipeline_utils.load_pipeline(pipeline_dict)

                # update id
                pipeline.id = str(uuid.uuid4())

                # update time
                pipeline.created = Pipeline().created

                possible_pipelines.append(pipeline)
            except Exception:
                pass

        return possible_pipelines
