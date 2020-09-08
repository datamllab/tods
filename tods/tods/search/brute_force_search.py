# A Brute-Force Search
import uuid

from d3m.metadata.pipeline import Pipeline

from axolotl.algorithms.base import PipelineSearchBase
from axolotl.utils import pipeline as pipeline_utils, schemas as schemas_utils

def random_rank(pipeline_result):
    if pipeline_result.status == 'COMPLETED':
        pipeline_result.rank = random.uniform(0, 1)
        return pipeline_result

class BruteForceSearch(PipelineSearchBase):
    def __init__(self, problem_description, backend, *, primitives_blocklist=None, ranking_function=None):
        super().__init__(problem_description=problem_description, backend=backend,
                primitives_blocklist=primitives_blocklist, ranking_function=ranking_function)
        if self.ranking_function is None:
                        self.ranking_function = random_rank

        # Find th candidates
        self.task_description = schemas_utils.get_task_description(self.problem_description['problem']['task_keywords'])
        print('task_description:', self.task_description)
        self.available_pipelines = self._return_pipelines(
                            self.task_description['task_type'], self.task_description['task_subtype'], self.task_description['data_types'])
        print('available_pipelines:', self.available_pipelines)

    def _return_pipelines(self, task_type, task_subtype, data_type):
        pipeline_candidates = []
        for pipeline_dict in schemas_utils.get_pipelines_db()['CLASSIFICATION']:
            pipeline = pipeline_utils.load_pipeline(pipeline_dict)
            pipeline.id = str(uuid.uuid4())
            pipeline.created = Pipeline().created
            pipeline_candidates.append(pipeline)

        return pipeline_candidates
