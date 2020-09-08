import copy
import json
import os
import unittest

import jsonschema

import d3m
from d3m import utils
from d3m.environment_variables import D3M_BASE_IMAGE_NAME, D3M_BASE_IMAGE_DIGEST, D3M_IMAGE_NAME, D3M_IMAGE_DIGEST
from d3m.metadata import base as metadata_base
from d3m.metadata.pipeline_run import RuntimeEnvironment


class TestComputeResources(unittest.TestCase):
    # todo
    pass


class TestRuntimeEnvironment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_path = os.path.realpath(d3m.__file__).rsplit('d3m',1)[0]
        cls.original_git_path = os.path.join(cls.repo_path, '.git')
        cls.moved_git_path = os.path.join(cls.repo_path, '.git_moved')

    @classmethod
    def tearDown(cls):
        if os.path.exists(cls.moved_git_path):
            os.rename(cls.moved_git_path, cls.original_git_path)

    def test_empty_instantiation(self):
        with utils.silence():
            RuntimeEnvironment()

    def test_deterministic_id(self):
        with utils.silence():
            env = RuntimeEnvironment()
        id_ = env['id']
        del env['id']
        gen_id = utils.compute_hash_id(env)
        self.assertEqual(id_, gen_id, 'environment.id not deterministically generated')

    def _set_env_vars(self):
        self.D3M_BASE_IMAGE_NAME_set_previously = False
        if D3M_BASE_IMAGE_NAME in os.environ:
            self.D3M_BASE_IMAGE_NAME_set_previously = True
            self.D3M_BASE_IMAGE_NAME_previous_value = os.environ[D3M_BASE_IMAGE_NAME]
        os.environ[D3M_BASE_IMAGE_NAME] = 'D3M_BASE_IMAGE_NAME_VALUE'

        self.D3M_BASE_IMAGE_DIGEST_set_previously = False
        if D3M_BASE_IMAGE_DIGEST in os.environ:
            self.D3M_BASE_IMAGE_DIGEST_set_previously = True
            self.D3M_BASE_IMAGE_DIGEST_previous_value = os.environ[D3M_BASE_IMAGE_DIGEST]
        os.environ[D3M_BASE_IMAGE_DIGEST] = 'D3M_BASE_IMAGE_DIGEST_VALUE'

        self.D3M_IMAGE_NAME_set_previously = False
        if D3M_IMAGE_NAME in os.environ:
            self.D3M_IMAGE_NAME_set_previously = True
            self.D3M_IMAGE_NAME_previous_value = os.environ[D3M_IMAGE_NAME]
        os.environ[D3M_IMAGE_NAME] = 'D3M_IMAGE_NAME_VALUE'

        self.D3M_IMAGE_DIGEST_set_previously = False
        if D3M_IMAGE_DIGEST in os.environ:
            self.D3M_IMAGE_DIGEST_set_previously = True
            self.D3M_IMAGE_DIGEST_previous_value = os.environ[D3M_IMAGE_DIGEST]
        os.environ[D3M_IMAGE_DIGEST] = 'D3M_IMAGE_DIGEST_VALUE'

    def _unset_env_vars(self):
        if self.D3M_BASE_IMAGE_NAME_set_previously:
            os.environ[D3M_BASE_IMAGE_NAME] = self.D3M_BASE_IMAGE_NAME_previous_value
        else:
            del os.environ[D3M_BASE_IMAGE_NAME]
        if self.D3M_BASE_IMAGE_DIGEST_set_previously:
            os.environ[D3M_BASE_IMAGE_DIGEST] = self.D3M_BASE_IMAGE_DIGEST_previous_value
        else:
            del os.environ[D3M_BASE_IMAGE_DIGEST]
        if self.D3M_IMAGE_NAME_set_previously:
            os.environ[D3M_IMAGE_NAME] = self.D3M_IMAGE_NAME_previous_value
        else:
            del os.environ[D3M_IMAGE_NAME]
        if self.D3M_IMAGE_DIGEST_set_previously:
            os.environ[D3M_IMAGE_DIGEST] = self.D3M_IMAGE_DIGEST_previous_value
        else:
            del os.environ[D3M_IMAGE_DIGEST]

    def test_env_vars(self):
        self._set_env_vars()
        try:
            with utils.silence():
                env = RuntimeEnvironment()

            self.assertEqual(
                env['base_docker_image']['image_name'],
                os.environ[D3M_BASE_IMAGE_NAME],
                'base_image_name incorrectly extracted from environment variables'
            )
            self.assertEqual(
                env['base_docker_image']['image_digest'],
                os.environ[D3M_BASE_IMAGE_DIGEST],
                'base_image_digest incorrectly extracted from environment variables'
            )
            self.assertEqual(
                env['docker_image']['image_name'],
                os.environ[D3M_IMAGE_NAME],
                'image_name incorrectly extracted from environment variables'
            )
            self.assertEqual(
                env['docker_image']['image_digest'],
                os.environ[D3M_IMAGE_DIGEST],
                'image_digest incorrectly extracted from environment variables'
            )

        finally:
            self._unset_env_vars()

    def test_no_git_repo(self):
        git_path_moved = False
        if os.path.exists(self.original_git_path):
            os.rename(self.original_git_path, self.moved_git_path)
            git_path_moved = True
        try:
            with utils.silence():
                env = RuntimeEnvironment()

            self.assertEqual(
                env['reference_engine_version'], d3m.__version__,
                'reference_engine_version incorrectly extracted from d3m repo'
            )

            self.assertEqual(
                env['engine_version'], d3m.__version__,
                'reference_engine_version incorrectly extracted from d3m repo'
            )
        finally:
            if git_path_moved:
                os.rename(self.moved_git_path, self.original_git_path)


class TestPipelineRunSchema(unittest.TestCase):
    def test_scoring(self):
        """
        When scoring of a pipeline is performed without a data preparation pipeline,
        the scoring datasets must be recorded in the pipeline run.
        When scoring pipeline information is not recored in pipeline run, results.scores
        should also not be recorded.
        """

        schemas = copy.copy(metadata_base.SCHEMAS)
        schemas['http://example.com/testing_run.json'] = copy.copy(metadata_base.DEFINITIONS_JSON)
        schemas['http://example.com/testing_run.json']['id'] = 'http://example.com/testing_run.json'
        schemas['http://example.com/testing_run.json'].update(metadata_base.DEFINITIONS_JSON['definitions']['pipeline_run'])

        validator, = utils.load_schema_validators(schemas, ('testing_run.json',))

        id_digest = {
            'id': '0000000000000000000000000000000000000000000000000000000000000000',
            'digest': '0000000000000000000000000000000000000000000000000000000000000000'
        }
        status = {'state': 'SUCCESS'}
        run_base_json = {'phase': 'FIT'}
        data_preparation_base_json = {
            'pipeline': id_digest,
            'steps': [
                {
                    'type': 'PRIMITIVE',
                    'status': status
                }
            ],
            'status': status
        }
        scoring_base_json = data_preparation_base_json
        results_scores = {
            'results': {
                'scores': [
                    {
                        'metric': {
                            'metric': 'ACCURACY',
                        },
                        'value': 0.5,
                    }
                ]
            }
        }

        valid_cases = [
            {
                **run_base_json
            },
            {
                **run_base_json,
                'data_preparation': data_preparation_base_json
            },
            {
                **run_base_json,
                'data_preparation': data_preparation_base_json,
                'scoring': scoring_base_json
            },
            {
                **run_base_json,
                'scoring': {
                    **scoring_base_json,
                    'datasets': [
                        id_digest
                    ]
                }
            },
            {
                **run_base_json,
                'data_preparation': data_preparation_base_json,
                'scoring': scoring_base_json,
                **results_scores,
            },
            {
                **run_base_json,
                'scoring': {
                    **scoring_base_json,
                    'datasets': [
                        id_digest
                    ]
                },
                **results_scores,
            },
        ]

        invalid_cases = [
            {
                **run_base_json,
                'scoring': scoring_base_json
            },
            {
                **run_base_json,
                **results_scores,
            },
            {
                **run_base_json,
                'data_preparation': data_preparation_base_json,
                **results_scores,
            },
            {
                **run_base_json,
                'scoring': scoring_base_json,
                **results_scores,
            },
            {
                **run_base_json,
                'data_preparation': data_preparation_base_json,
                'scoring': {
                    **scoring_base_json,
                    'datasets': [
                        id_digest
                    ]
                }
            },
            {
                **run_base_json,
                'data_preparation': data_preparation_base_json,
                'scoring': {
                    **scoring_base_json,
                    'datasets': [
                        id_digest
                    ]
                },
                **results_scores,
            },
        ]

        for i, valid_case in enumerate(valid_cases):
            try:
                validator.validate(valid_case)
            except jsonschema.exceptions.ValidationError as e:
                self.fail(f'{i}: {e}')

        for i, invalid_case in enumerate(invalid_cases):
            with self.assertRaises(jsonschema.exceptions.ValidationError, msg=str(i)):
                validator.validate(invalid_case)


if __name__ == '__main__':
    unittest.main()
