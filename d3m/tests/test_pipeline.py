import collections
import copy
import datetime
import json
import logging
import os
import sys
import typing
import unittest
import uuid

COMMON_PRIMITIVES_DIR = os.path.join(os.path.dirname(__file__), 'common-primitives')
# NOTE: This insertion should appear before any code attempting to resolve or load primitives,
# so the git submodule version of `common-primitives` is looked at first.
sys.path.insert(0, COMMON_PRIMITIVES_DIR)

TEST_PRIMITIVES_DIR = os.path.join(os.path.dirname(__file__), 'data', 'primitives')
sys.path.insert(0, TEST_PRIMITIVES_DIR)

from common_primitives.dataset_to_dataframe import DatasetToDataFramePrimitive
from common_primitives.column_parser import ColumnParserPrimitive
from common_primitives.random_forest import RandomForestClassifierPrimitive

from test_primitives.monomial import MonomialPrimitive
from test_primitives.random import RandomPrimitive
from test_primitives.sum import SumPrimitive
from test_primitives.increment import IncrementPrimitive

from d3m import container, exceptions, index, utils
from d3m.metadata import base as metadata_base, hyperparams, params, pipeline
from d3m.primitive_interfaces import base, transformer, supervised_learning


TEST_PIPELINE_1 = """
{
  "id": "2b50a7db-c5e2-434c-b02d-9e595bd56788",
  "digest": "b87dbbd5b8bcc1470050a756cf22d6def2662a61482debf55c09948225372411",
  "schema": "https://metadata.datadrivendiscovery.org/schemas/v0/pipeline.json",
  "source": {
    "name": "Test author",
    "contact": "mailto:test@example.com"
  },
  "created": "2018-02-28T09:42:27.443844Z",
  "name": "Test pipeline",
  "description": "Just a test pipeline",
  "users": [
    {
      "id": "f32467bc-698c-4ab6-a489-2e8f73fcfdaa",
      "reason": "User was making a test",
      "rationale": "I made a test"
    }
  ],
  "inputs": [
    {
      "name": "dataframe inputs"
    },
    {
      "name": "dataframe outputs"
    },
    {
      "name": "extra data"
    }
  ],
  "outputs": [
    {
      "name": "dataframe predictions",
      "data": "steps.6.main"
    }
  ],
  "steps": [
    {
      "type": "PRIMITIVE",
      "primitive": {
        "id": "efa24fae-49c4-4482-b49f-ceb351c0d916",
        "version": "0.1.0",
        "python_path": "d3m.primitives.test.LossPrimitive",
        "name": "Loss Primitive"
      },
      "arguments": {
        "inputs": {
          "type": "CONTAINER",
          "data": "inputs.0"
        },
        "outputs": {
          "type": "CONTAINER",
          "data": "inputs.1"
        }
      },
      "outputs": [
        {
          "id": "produce"
        }
      ]
    },
    {
      "type": "PRIMITIVE",
      "primitive": {
        "id": "00c3a435-a87c-405b-bed9-3a8c402d4431",
        "version": "0.1.0",
        "python_path": "d3m.primitives.test.Model1Primitive",
        "name": "Model 1 Primitive"
      },
      "arguments": {
        "inputs": {
          "type": "CONTAINER",
          "data": "inputs.0"
        },
        "outputs": {
          "type": "CONTAINER",
          "data": "inputs.1"
        }
      },
      "outputs": [
        {
          "id": "produce"
        }
      ]
    },
    {
      "type": "PRIMITIVE",
      "primitive": {
        "id": "4987c4b0-cf4c-4f7f-9bcc-557a6d72589d",
        "version": "0.1.0",
        "python_path": "d3m.primitives.test.Model2Primitive",
        "name": "Model 2 Primitive"
      },
      "arguments": {
        "inputs": {
          "type": "CONTAINER",
          "data": "inputs.0"
        },
        "outputs": {
          "type": "CONTAINER",
          "data": "inputs.1"
        }
      },
      "outputs": [
        {
          "id": "produce"
        }
      ]
    },
    {
      "type": "PRIMITIVE",
      "primitive": {
        "id": "9c00d42d-382d-4177-a0e7-082da88a29c8",
        "version": "0.1.0",
        "python_path": "d3m.primitives.operator.sum.Test",
        "name": "Sum Values",
        "digest": "__SUM_DIGEST__"
      },
      "arguments": {
        "inputs": {
            "type": "CONTAINER",
            "data": "inputs.0"
        }
      },
      "outputs": [
        {
          "id": "produce"
        }
      ]
    },
    {
      "type": "PRIMITIVE",
      "primitive": {
        "id": "e42e6f17-77cc-4611-8cca-bba36a46e806",
        "version": "0.1.0",
        "python_path": "d3m.primitives.test.PipelineTestPrimitive",
        "name": "Pipeline Test Primitive"
      },
      "arguments": {
        "inputs": {
          "type": "CONTAINER",
          "data": "inputs.0"
        },
        "extra_data": {
          "type": "CONTAINER",
          "data": "inputs.2"
        },
        "offset": {
          "type": "DATA",
          "data": "steps.3.produce"
        }
      },
      "outputs": [
        {
          "id": "produce"
        },
        {
          "id": "produce_score"
        }
      ],
      "hyperparams": {
        "loss": {
          "type": "PRIMITIVE",
          "data": 0
        },
        "column_to_operate_on": {
          "type": "VALUE",
          "data": 5
        },
        "ensemble": {
          "type": "PRIMITIVE",
          "data": [
            1,
            2
          ]
        },
        "columns_to_operate_on": {
          "type": "VALUE",
          "data": [3, 6, 7]
        }
      },
      "users": [
        {
          "id": "98e5cc4a-7edc-41a3-ac98-ee799fb6a41b",
          "reason": "User clicked on a button",
          "rationale": "I dragged an icon"
        }
      ]
    },
    {
      "type": "SUBPIPELINE",
      "pipeline": {
        "id": "0113b91f-3010-4a47-bd56-a50c4e28a4a4",
        "digest": "83430addfcb9430ad02fd59f114ac7c723806058ca90d6b0f226d1031826ac8d"
      },
      "inputs": [
        {
          "data": "steps.4.produce"
        }
      ],
      "outputs": [
        {
          "id": "pipeline_output"
        }
      ]
    },
    {
      "type": "PLACEHOLDER",
      "inputs": [
        {
          "data": "steps.5.pipeline_output"
        },
        {
          "data": "steps.4.produce_score"
        }
      ],
      "outputs": [
        {
          "id": "main"
        }
      ]
    }
  ]
}
""".replace('__SUM_DIGEST__', SumPrimitive.metadata.query()['digest'])

TEST_PIPELINE_2 = """
{
  "id": "0113b91f-3010-4a47-bd56-a50c4e28a4a4",
  "digest": "83430addfcb9430ad02fd59f114ac7c723806058ca90d6b0f226d1031826ac8d",
  "schema": "https://metadata.datadrivendiscovery.org/schemas/v0/pipeline.json",
  "created": "2018-02-28T09:42:27.443844Z",
  "name": "Test pipeline",
  "description": "Just a test pipeline",
  "inputs": [
    {}
  ],
  "outputs": [
    {
      "data": "steps.0.produce"
    }
  ],
  "steps": [
    {
      "type": "PRIMITIVE",
      "primitive": {
        "id": "4987c4b0-cf4c-4f7f-9bcc-557a6d72589d",
        "version": "0.1.0",
        "python_path": "d3m.primitives.test.Model2Primitive",
        "name": "Model 2 Primitive"
      },
      "arguments": {
        "inputs": {
          "type": "CONTAINER",
          "data": "inputs.0"
        },
        "outputs": {
          "type": "CONTAINER",
          "data": "inputs.0"
        }
      },
      "outputs": [
        {
          "id": "produce"
        }
      ]
    }
  ]
}
"""


class MockPrimitiveBuilder:
    """
    This class helps build mock primitives from scratch without checking.
    """

    def __init__(self, inputs, hyperparams, primitive_id=None, version='0.0.0', name='mock_primitive_name', python_path='d3m.primitives.mock.foobar', digest='f' * 64):
        """
        inputs : Dict
            It will be used to fill the 'arguments' field.
        outputs : List
            List of output names
        """

        self.primitive_dict = {
            'type': 'PRIMITIVE',
            'primitive': {
                'id': primitive_id if primitive_id is not None else str(uuid.uuid4()),
                'version': version,
                'python_path': python_path,
                'name': name,
                'digest': digest,
            },
            'arguments': inputs,
            'hyperparams': hyperparams,
            'outputs': None,
        }

    def build(self, **inputs_data):
        primitive_dict = copy.deepcopy(self.primitive_dict)
        primitive_dict['arguments'] = copy.deepcopy({name: self.primitive_dict['arguments'][name] for name in inputs_data.keys() if name in self.primitive_dict['arguments']})
        primitive_dict['hyperparams'] = copy.deepcopy({name: self.primitive_dict['hyperparams'][name] for name in inputs_data.keys() if name in self.primitive_dict['hyperparams']})

        for name, data in inputs_data.items():
            if name in primitive_dict['arguments']:
                primitive_dict['arguments'][name]['data'] = data
            elif name in primitive_dict['hyperparams']:
                primitive_dict['hyperparams'][name]['data'] = data
            else:
                raise IndexError("No match name found for '{name}' in primitive {primitive_name}".format(name=name, primitive_name=self.primitive_dict['primitive']['name']))
        return primitive_dict


class MockPipelineBuilder:
    """
    This class helps build pipelines for testing from scratch without checking.
    """

    def __init__(self, input_names, pipeline_id=None, name='mock_name', description='mock_description'):
        self._increase_counter = 0
        self.pipeline_dict = {
            'id': pipeline_id if pipeline_id is not None else str(uuid.uuid4()),
            'schema': 'https://metadata.datadrivendiscovery.org/schemas/v0/pipeline.json',
            'source': {'name': 'Test author'},
            'created': datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
            'name': name,
            'description': description,
            'inputs': [{'name': n} for n in input_names],
            'outputs': [],
            'steps': []
        }
        self._subpipelines = {}

    def add_primitive(self, primitive_dict, outputs):
        """
        Add primitives.
        """

        primitive_dict['outputs'] = [{'id': o} for o in outputs]
        self.pipeline_dict['steps'].append(primitive_dict)

    def add_placeholder(self, inputs, outputs):
        placeholder_dict = {
            'type': 'PLACEHOLDER',
            'inputs': [{'data': input_data_ref} for input_data_ref in inputs],
            'outputs': [{'id': output_id for output_id in outputs}]
        }
        self.pipeline_dict['steps'].append(placeholder_dict)

    def add_subpipeline(self, pipeline: pipeline.Pipeline, inputs, outputs):
        self._subpipelines[pipeline.id] = pipeline
        subpipeline_dict = {
            'type': 'SUBPIPELINE',
            'pipeline': {'id': pipeline.id},
            'inputs': [{'data': input_data_ref} for input_data_ref in inputs],
            'outputs': [{'id': output_id for output_id in outputs}]
        }
        self.pipeline_dict['steps'].append(subpipeline_dict)

    def add_output(self, name, data):
        self.pipeline_dict['outputs'].append({'name': name, 'data': data})

    def build(self, primitive_loading='ignore') -> pipeline.Pipeline:
        """
        Output built pipeline instance.

        Parameters
        ----------
        primitive_loading : str or callable
            If `primitive_loading` == 'ignore', the primitive resolving function will be skipped.
            If `primitive_loading` == 'default', a default primitive resolving function will be loaded.
            If `primitive_loading` is a function, it will become the resolving function.

        Returns
        -------
        Pipeline
            A pipeline instance.
        """

        resolver = pipeline.Resolver()
        resolver.get_pipeline = lambda pipeline_description: self._subpipelines[pipeline_description['id']]
        if primitive_loading == 'ignore':
            resolver.get_primitive = lambda primitive_description: None
        elif primitive_loading == 'full':
            pass
        elif callable(primitive_loading):
            resolver.get_primitive = primitive_loading
        else:
            raise ValueError("unknown value of 'primitive_loading'")

        return pipeline.Pipeline.from_json_structure(self.pipeline_dict, resolver=resolver)


class Resolver(pipeline.Resolver):
    def _get_primitive(self, primitive_description: typing.Dict) -> typing.Optional[typing.Type[base.PrimitiveBase]]:
        # To hide any logging or stdout output.
        with utils.silence():
            return super()._get_primitive(primitive_description)

    def get_pipeline(self, pipeline_description: typing.Dict) -> pipeline.Pipeline:
        if pipeline_description['id'] == '0113b91f-3010-4a47-bd56-a50c4e28a4a4':
            return pipeline.Pipeline.from_json(TEST_PIPELINE_2, resolver=self)

        return super().get_pipeline(pipeline_description)


Inputs = container.DataFrame
Outputs = container.DataFrame


class Hyperparams(hyperparams.Hyperparams):
    pass


class Params(params.Params):
    pass


# Silence any validation warnings.
with utils.silence():
    class LossPrimitive(supervised_learning.SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
        metadata = metadata_base.PrimitiveMetadata({
            'id': 'efa24fae-49c4-4482-b49f-ceb351c0d916',
            'version': '0.1.0',
            'name': "Loss Primitive",
            'python_path': 'd3m.primitives.test.LossPrimitive',
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.CROSS_ENTROPY,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.LOSS_FUNCTION,
        })

        def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
            pass

        def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
            pass

        def fit(self, *, timeout: float = None, iterations: int = None) -> base.CallResult[None]:
            pass

        def get_params(self) -> Params:
            pass

        def set_params(self, *, params: Params) -> None:
            pass

    class Model1Primitive(supervised_learning.SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
        metadata = metadata_base.PrimitiveMetadata({
            'id': '00c3a435-a87c-405b-bed9-3a8c402d4431',
            'version': '0.1.0',
            'name': "Model 1 Primitive",
            'python_path': 'd3m.primitives.test.Model1Primitive',
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.RANDOM_FOREST,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.CLASSIFICATION,
        })

        def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
            pass

        def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
            pass

        def fit(self, *, timeout: float = None, iterations: int = None) -> base.CallResult[None]:
            pass

        def get_params(self) -> Params:
            pass

        def set_params(self, *, params: Params) -> None:
            pass

    class Model2Hyperparams(hyperparams.Hyperparams):
        # To test that a primitive instance can be a default value.
        base_estimator = hyperparams.Hyperparameter[base.PrimitiveBase](
            default=LossPrimitive(hyperparams=Hyperparams.defaults()),
            semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        )

    class Model2Primitive(supervised_learning.SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Model2Hyperparams]):
        metadata = metadata_base.PrimitiveMetadata({
            'id': '4987c4b0-cf4c-4f7f-9bcc-557a6d72589d',
            'version': '0.1.0',
            'name': "Model 2 Primitive",
            'python_path': 'd3m.primitives.test.Model2Primitive',
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.SUPPORT_VECTOR_MACHINE,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.CLASSIFICATION,
        })

        def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
            pass

        def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
            pass

        def fit(self, *, timeout: float = None, iterations: int = None) -> base.CallResult[None]:
            pass

        def get_params(self) -> Params:
            pass

        def set_params(self, *, params: Params) -> None:
            pass


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        column_index = hyperparams.Hyperparameter[int](-1)

        class PipelineTestHyperparams(hyperparams.Hyperparams):
            loss = hyperparams.Hyperparameter[typing.Optional[base.PrimitiveBase]](default=None, semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'])
            column_to_operate_on = hyperparams.Hyperparameter[int](default=-1, semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'])
            ensemble = hyperparams.Set(elements=hyperparams.Hyperparameter[base.PrimitiveBase](default=MonomialPrimitive), default=(), max_size=10, semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'])
            columns_to_operate_on = hyperparams.Set(column_index, (), 0, None, semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'])

        PipelineTestInputs = typing.Union[container.Dataset, container.DataFrame]

        # Silence any validation warnings.
        with utils.silence():
            class PipelineTestPrimitive(transformer.TransformerPrimitiveBase[PipelineTestInputs, Outputs, PipelineTestHyperparams]):
                metadata = metadata_base.PrimitiveMetadata({
                    'id': 'e42e6f17-77cc-4611-8cca-bba36a46e806',
                    'version': '0.1.0',
                    'name': "Pipeline Test Primitive",
                    'python_path': 'd3m.primitives.test.PipelineTestPrimitive',
                    'algorithm_types': [
                        metadata_base.PrimitiveAlgorithmType.CROSS_ENTROPY,
                    ],
                    'primitive_family': metadata_base.PrimitiveFamily.LOSS_FUNCTION,
                })

                def produce(self, *, inputs: PipelineTestInputs, extra_data: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
                    pass

                def produce_score(self, *, inputs: PipelineTestInputs, offset: float, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
                    pass

                def multi_produce(self, *, produce_methods: typing.Sequence[str], inputs: PipelineTestInputs, extra_data: Inputs, offset: float, timeout: float = None, iterations: int = None) -> base.MultiCallResult:
                    return self._multi_produce(produce_methods=produce_methods, timeout=timeout, iterations=iterations, inputs=inputs, extra_data=extra_data, offset=offset)

                def fit_multi_produce(self, *, produce_methods: typing.Sequence[str], inputs: PipelineTestInputs, extra_data: Inputs, offset: float, timeout: float = None, iterations: int = None) -> base.MultiCallResult:
                    return self._fit_multi_produce(produce_methods=produce_methods, timeout=timeout, iterations=iterations, inputs=inputs, extra_data=extra_data, offset=offset)

            class SimplePipelineTestPrimitive(transformer.TransformerPrimitiveBase[PipelineTestInputs, Outputs, PipelineTestHyperparams]):
                metadata = metadata_base.PrimitiveMetadata({
                    'id': '02d966d6-4e4f-465b-ad93-83b14c7c47be',
                    'version': '0.1.0',
                    'name': "Simple Pipeline Test Primitive",
                    'python_path': 'd3m.primitives.test.SimplePipelineTestPrimitive',
                    'algorithm_types': [
                        metadata_base.PrimitiveAlgorithmType.CROSS_ENTROPY,
                    ],
                    'primitive_family': metadata_base.PrimitiveFamily.LOSS_FUNCTION,
                })

                def produce(self, *, inputs: PipelineTestInputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
                    pass

        ColumnsInputs = container.Dataset
        ColumnsOutputs = container.List

        # Silence any validation warnings.
        with utils.silence():
            class ColumnSelectionPrimitive(transformer.TransformerPrimitiveBase[ColumnsInputs, ColumnsOutputs, Hyperparams]):
                metadata = metadata_base.PrimitiveMetadata({
                    'id': 'fdabb0c2-0555-4188-8f08-eeda722e1f04',
                    'version': '0.1.0',
                    'name': "Column Selection Primitive",
                    'python_path': 'd3m.primitives.test.ColumnSelectionPrimitive',
                    'algorithm_types': [
                        metadata_base.PrimitiveAlgorithmType.PRINCIPAL_COMPONENT_ANALYSIS,
                    ],
                    'primitive_family': metadata_base.PrimitiveFamily.FEATURE_SELECTION,
                })

                def produce(self, *, inputs: ColumnsInputs, timeout: float = None, iterations: int = None) -> base.CallResult[ColumnsOutputs]:
                    pass

        # To hide any logging or stdout output.
        with utils.silence():
            index.register_primitive('d3m.primitives.regression.monomial.Test', MonomialPrimitive)
            index.register_primitive('d3m.primitives.data_generation.random.Test', RandomPrimitive)
            index.register_primitive('d3m.primitives.operator.sum.Test', SumPrimitive)
            index.register_primitive('d3m.primitives.operator.increment.Test', IncrementPrimitive)
            index.register_primitive('d3m.primitives.test.LossPrimitive', LossPrimitive)
            index.register_primitive('d3m.primitives.test.Model1Primitive', Model1Primitive)
            index.register_primitive('d3m.primitives.test.Model2Primitive', Model2Primitive)
            index.register_primitive('d3m.primitives.test.PipelineTestPrimitive', PipelineTestPrimitive)
            index.register_primitive('d3m.primitives.test.SimplePipelineTestPrimitive', SimplePipelineTestPrimitive)
            index.register_primitive('d3m.primitives.test.ColumnSelectionPrimitive', ColumnSelectionPrimitive)

    def test_basic(self):
        self.maxDiff = None

        p = pipeline.Pipeline.from_json(TEST_PIPELINE_2, resolver=Resolver())

        p_json_input = json.loads(TEST_PIPELINE_2)
        p_json_output = p.to_json_structure()

        self.assertEqual(p_json_input, p_json_output)

        p.check(standard_pipeline=False)

        p = pipeline.Pipeline.from_json(TEST_PIPELINE_1, resolver=Resolver())

        p.check(allow_placeholders=True, input_types={'inputs.0': container.DataFrame, 'inputs.1': container.DataFrame, 'inputs.2': container.DataFrame})

        with self.assertRaisesRegex(exceptions.InvalidPipelineError, 'Step .* of pipeline \'.*\' is a placeholder but there should be no placeholders'):
            p.check(allow_placeholders=False, input_types={'inputs.0': container.DataFrame, 'inputs.1': container.DataFrame, 'inputs.2': container.DataFrame})

        p_json_input = json.loads(TEST_PIPELINE_1)
        p_json_output = p.to_json_structure()

        p_json_input.pop('digest', None)
        p_json_output.pop('digest', None)
        self.assertEqual(p_json_input, p_json_output)

        p_from_json = pipeline.Pipeline.from_json(p.to_json(), resolver=Resolver()).to_json_structure()
        p_from_json.pop('digest', None)
        self.assertEqual(p_json_input, p_from_json)

        p_from_yaml = pipeline.Pipeline.from_yaml(p.to_yaml(), resolver=Resolver()).to_json_structure()
        p_from_yaml.pop('digest', None)
        self.assertEqual(p_json_input, p_from_yaml)

        self.assertEqual(p.get_producing_outputs(), {'outputs.0', 'steps.0.produce', 'steps.1.produce', 'steps.2.produce', 'steps.3.produce', 'steps.4.produce', 'steps.4.produce_score', 'steps.5.outputs.0', 'steps.5.pipeline_output', 'steps.5.steps.0.produce', 'steps.6.main'})

    def test_non_strict_resolving(self):
        test_pipeline = json.loads(TEST_PIPELINE_1)

        full_primitive_description = copy.deepcopy(test_pipeline['steps'][3]['primitive'])
        full_pipeline_description = copy.deepcopy(test_pipeline['steps'][5]['pipeline'])

        test_pipeline['steps'][3]['primitive']['version'] = '0.0.1'
        test_pipeline['steps'][3]['primitive']['name'] = 'Something Else'
        del test_pipeline['steps'][3]['primitive']['digest']
        del test_pipeline['steps'][5]['pipeline']['digest']
        test_pipeline['digest'] = utils.compute_digest(test_pipeline)

        logger = logging.getLogger('d3m.metadata.pipeline')

        with self.assertLogs(logger=logger, level=logging.DEBUG) as cm:
            p = pipeline.Pipeline.from_json(json.dumps(test_pipeline), resolver=Resolver())

        self.assertEqual(len(cm.records), 2)
        self.assertEqual(cm.records[0].msg, "Version for primitive '%(primitive_id)s' does not match the one specified in the primitive description. Primitive description version: '%(primitive_version)s'. Resolved primitive version: '%(resolved_primitive_version)s'.")
        self.assertEqual(cm.records[1].msg, "Name for primitive '%(primitive_id)s' does not match the one specified in the primitive description. Primitive description name: '%(primitive_name)s'. Resolved primitive name: '%(resolved_primitive_name)s'.")

        # After loading, primitive and pipeline information should be updated and fully populated.
        self.assertEqual(p.to_json_structure()['steps'][3]['primitive'], full_primitive_description)
        self.assertEqual(p.to_json_structure()['steps'][5]['pipeline'], full_pipeline_description)

    def test_nested_to_json_structure(self):
        p = pipeline.Pipeline.from_json(TEST_PIPELINE_1, resolver=Resolver())

        self.assertEqual(p.to_json_structure()['steps'][5]['pipeline'], {
            'id': '0113b91f-3010-4a47-bd56-a50c4e28a4a4',
            'digest': '83430addfcb9430ad02fd59f114ac7c723806058ca90d6b0f226d1031826ac8d',
        })

        p2 = pipeline.Pipeline.from_json_structure(p.to_json_structure(), resolver=Resolver())

        self.assertEqual(p.to_json_structure(nest_subpipelines=True), p2.to_json_structure(nest_subpipelines=True))

        self.assertEqual(p.to_json_structure(nest_subpipelines=True)['steps'][5]['pipeline'], json.loads(TEST_PIPELINE_2))

        class TestResolver(Resolver):
            def _from_file(self, pipeline_description):
                raise AssertionError("Should not be called.")

        p2 = pipeline.Pipeline.from_json_structure(p.to_json_structure(nest_subpipelines=True), resolver=TestResolver())

        self.assertEqual(p.to_json_structure(nest_subpipelines=True), p2.to_json_structure(nest_subpipelines=True))

    def test_primitive_annotation(self):
        # This test does not really belong here but it is easiest to make it here.
        # Test that hyper-parameter can have a primitive instance as a default value
        # and that such primitive can have its metadata converted to JSON.

        self.assertEqual(index.get_primitive('d3m.primitives.test.Model2Primitive').metadata.to_json_structure()['primitive_code']['hyperparams']['base_estimator'], {
            'type': 'd3m.metadata.hyperparams.Hyperparameter',
            'default': 'd3m.primitives.test.LossPrimitive(hyperparams=Hyperparams({}), random_seed=0)',
            'structural_type': 'd3m.primitive_interfaces.base.PrimitiveBase',
            'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        })

    @unittest.skipUnless(sys.version_info >= (3, 7), "Pickling of generic types does not work before Python 3.7.")
    def test_primitive_annotation_python37(self):
        # This test does not really belong here but it is easiest to make it here.
        # Test that hyper-parameter can have a primitive instance as a default value
        # and that such primitive can have its metadata converted to JSON.

        self.assertEqual(index.get_primitive('d3m.primitives.test.Model2Primitive').metadata.to_internal_json_structure()['primitive_code']['hyperparams']['base_estimator'], {
            'type': 'd3m.metadata.hyperparams.Hyperparameter',
            'default': 'd3m.primitives.test.LossPrimitive(hyperparams=Hyperparams({}), random_seed=0)',
            'structural_type': 'd3m.primitive_interfaces.base.PrimitiveBase',
            'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        })

    def test_pipeline_digest_mismatch(self):
        logger = logging.getLogger('d3m.metadata.pipeline')

        with self.assertLogs(logger=logger, level=logging.DEBUG) as cm:
            pipeline.Pipeline.from_json("""
                {
                  "id": "c12a8de1-d4d7-4d4b-b51f-66488e1adcc6",
                  "digest": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                  "schema": "https://metadata.datadrivendiscovery.org/schemas/v0/pipeline.json",
                  "created": "2018-02-28T09:42:27.443844Z",
                  "name": "Test pipeline",
                  "description": "Just a test pipeline",
                  "inputs": [
                    {}
                  ],
                  "outputs": [
                    {
                      "data": "steps.0.produce"
                    }
                  ],
                  "steps": [
                    {
                      "type": "PRIMITIVE",
                      "primitive": {
                        "id": "9c00d42d-382d-4177-a0e7-082da88a29c8",
                        "version": "0.1.0",
                        "python_path": "d3m.primitives.operator.sum.Test",
                        "name": "Sum Values",
                        "digest": "__SUM_DIGEST__"
                      },
                      "arguments": {
                        "inputs": {
                          "type": "CONTAINER",
                          "data": "inputs.0"
                        }
                      },
                      "outputs": [
                        {
                          "id": "produce"
                        }
                      ]
                    }
                  ]
                }
            """.replace('__SUM_DIGEST__', SumPrimitive.metadata.query()['digest']), resolver=Resolver())

        self.assertEqual(len(cm.records), 1)
        self.assertEqual(cm.records[0].msg, "Digest for pipeline '%(pipeline_id)s' does not match a computed one. Provided digest: %(pipeline_digest)s. Computed digest: %(new_pipeline_digest)s.")

    def test_digest_mismatch(self):
        test_pipeline = json.loads(TEST_PIPELINE_1)

        full_primitive_description = copy.deepcopy(test_pipeline['steps'][3]['primitive'])
        full_pipeline_description = copy.deepcopy(test_pipeline['steps'][5]['pipeline'])

        test_pipeline['steps'][3]['primitive']['digest'] = 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
        test_pipeline['steps'][5]['pipeline']['digest'] = 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'

        logger = logging.getLogger('d3m.metadata.pipeline')

        with self.assertLogs(logger=logger, level=logging.DEBUG) as cm:
            p = pipeline.Pipeline.from_json(json.dumps(test_pipeline), resolver=Resolver())

        self.assertEqual(len(cm.records), 2)
        self.assertEqual(cm.records[0].msg, "Digest for pipeline '%(pipeline_id)s' does not match a computed one. Provided digest: %(pipeline_digest)s. Computed digest: %(new_pipeline_digest)s.")
        self.assertEqual(cm.records[1].msg, "Digest for primitive '%(primitive_id)s' does not match the one specified in the primitive description. Primitive description digest: %(primitive_digest)s. Resolved primitive digest: %(resolved_primitive_digest)s.")

        # After loading, primitive and pipeline information should be updated and fully populated.
        self.assertEqual(p.to_json_structure()['steps'][3]['primitive'], full_primitive_description)
        self.assertEqual(p.to_json_structure()['steps'][5]['pipeline'], full_pipeline_description)

    def test_invalid_data_reference(self):
        with self.assertRaisesRegex(exceptions.InvalidArgumentValueError, 'Cannot add step .*'):
            pipeline.Pipeline.from_json("""
                {
                  "id": "c12a8de1-d4d7-4d4b-b51f-66488e1adcc6",
                  "schema": "https://metadata.datadrivendiscovery.org/schemas/v0/pipeline.json",
                  "created": "2018-02-28T09:42:27.443844Z",
                  "name": "Test pipeline",
                  "description": "Just a test pipeline",
                  "inputs": [
                    {}
                  ],
                  "outputs": [
                    {
                      "data": "steps.0.produce"
                    }
                  ],
                  "steps": [
                    {
                      "type": "PRIMITIVE",
                      "primitive": {
                        "id": "9c00d42d-382d-4177-a0e7-082da88a29c8",
                        "version": "0.1.0",
                        "python_path": "d3m.primitives.operator.sum.Test",
                        "name": "Sum Values",
                        "digest": "__SUM_DIGEST__"
                      },
                      "arguments": {
                        "inputs": {
                          "type": "CONTAINER",
                          "data": "inputs.1"
                        }
                      },
                      "outputs": [
                        {
                          "id": "produce"
                        }
                      ]
                    }
                  ]
                }
            """.replace('__SUM_DIGEST__', SumPrimitive.metadata.query()['digest']), resolver=Resolver())

    def test_invalid_data_reference_in_argument_list(self):
        with self.assertRaisesRegex(exceptions.InvalidArgumentValueError, 'Cannot add step .*'):
            pipeline.Pipeline.from_json("""
                {
                  "id": "c12a8de1-d4d7-4d4b-b51f-66488e1adcc6",
                  "schema": "https://metadata.datadrivendiscovery.org/schemas/v0/pipeline.json",
                  "created": "2018-02-28T09:42:27.443844Z",
                  "name": "Test pipeline",
                  "description": "Just a test pipeline",
                  "inputs": [
                    {}
                  ],
                  "outputs": [
                    {
                      "data": "steps.0.produce"
                    }
                  ],
                  "steps": [
                    {
                      "type": "PRIMITIVE",
                      "primitive": {
                        "id": "9c00d42d-382d-4177-a0e7-082da88a29c8",
                        "version": "0.1.0",
                        "python_path": "d3m.primitives.operator.sum.Test",
                        "name": "Sum Values",
                        "digest": "__SUM_DIGEST__"
                      },
                      "arguments": {
                        "inputs": {
                          "type": "CONTAINER",
                          "data": [
                            "inputs.1"
                          ]
                        }
                      },
                      "outputs": [
                        {
                          "id": "produce"
                        }
                      ]
                    }
                  ]
                }
            """.replace('__SUM_DIGEST__', SumPrimitive.metadata.query()['digest']), resolver=Resolver())

    def test_invalid_argument_list_type_check(self):
        with self.assertRaisesRegex(exceptions.InvalidPipelineError, 'should have type \'List\' to support getting a list of values'):
            pipeline.Pipeline.from_json("""
                {
                  "id": "c12a8de1-d4d7-4d4b-b51f-66488e1adcc6",
                  "schema": "https://metadata.datadrivendiscovery.org/schemas/v0/pipeline.json",
                  "created": "2018-02-28T09:42:27.443844Z",
                  "name": "Test pipeline",
                  "description": "Just a test pipeline",
                  "inputs": [
                    {}
                  ],
                  "outputs": [
                    {
                      "data": "steps.0.produce"
                    }
                  ],
                  "steps": [
                    {
                      "type": "PRIMITIVE",
                      "primitive": {
                        "id": "9c00d42d-382d-4177-a0e7-082da88a29c8",
                        "version": "0.1.0",
                        "python_path": "d3m.primitives.operator.sum.Test",
                        "name": "Sum Values",
                        "digest": "__SUM_DIGEST__"
                      },
                      "arguments": {
                        "inputs": {
                          "type": "CONTAINER",
                          "data": [
                            "inputs.0"
                          ]
                        }
                      },
                      "outputs": [
                        {
                          "id": "produce"
                        }
                      ]
                    }
                  ]
                }
            """.replace('__SUM_DIGEST__', SumPrimitive.metadata.query()['digest']), resolver=Resolver()).check()

    def test_list_of_columns(self):
        pipeline.Pipeline.from_json("""
            {
              "id": "48fa0619-53f2-4a36-8a90-31ba8e08df02",
              "schema": "https://metadata.datadrivendiscovery.org/schemas/v0/pipeline.json",
              "created": "2018-02-28T09:42:27.443844Z",
              "name": "Test pipeline",
              "description": "Just a test pipeline",
              "inputs": [
                {}
              ],
              "outputs": [
                {
                  "data": "steps.1.produce"
                }
              ],
              "steps": [
                {
                  "type": "PRIMITIVE",
                  "primitive": {
                    "id": "fdabb0c2-0555-4188-8f08-eeda722e1f04",
                    "version": "0.1.0",
                    "python_path": "d3m.primitives.test.ColumnSelectionPrimitive",
                    "name": "Column Selection Primitive"
                  },
                  "arguments": {
                    "inputs": {
                      "type": "CONTAINER",
                      "data": "inputs.0"
                    }
                  },
                  "outputs": [
                    {
                      "id": "produce"
                    }
                  ]
                },
                {
                  "type": "PRIMITIVE",
                  "primitive": {
                    "id": "02d966d6-4e4f-465b-ad93-83b14c7c47be",
                    "version": "0.1.0",
                    "python_path": "d3m.primitives.test.SimplePipelineTestPrimitive",
                    "name": "Simple Pipeline Test Primitive"
                  },
                  "arguments": {
                    "inputs": {
                      "type": "CONTAINER",
                      "data": "inputs.0"
                    }
                  },
                  "outputs": [
                    {
                      "id": "produce"
                    }
                  ],
                  "hyperparams": {
                    "columns_to_operate_on": {
                      "type": "CONTAINER",
                      "data": "steps.0.produce"
                    }
                  }
                }
              ]
            }
        """, resolver=Resolver()).check()

    def test_type_check(self):
        with self.assertRaisesRegex(exceptions.InvalidPipelineError, 'Argument \'.*\' of step .* of pipeline \'.*\' has type \'.*\', but it is getting a type \'.*\''):
            pipeline.Pipeline.from_json("""
                {
                  "id": "e8c4dd86-420d-4e1c-ad25-d592a5b5bb0b",
                  "schema": "https://metadata.datadrivendiscovery.org/schemas/v0/pipeline.json",
                  "created": "2018-02-28T09:42:27.443844Z",
                  "name": "Test pipeline",
                  "description": "Just a test pipeline",
                  "inputs": [
                    {}
                  ],
                  "outputs": [
                    {
                      "data": "steps.0.produce"
                    }
                  ],
                  "steps": [
                    {
                      "type": "PRIMITIVE",
                      "primitive": {
                        "id": "9c00d42d-382d-4177-a0e7-082da88a29c8",
                        "version": "0.1.0",
                        "python_path": "d3m.primitives.operator.sum.Test",
                        "name": "Sum Values",
                        "digest": "__SUM_DIGEST__"
                      },
                      "arguments": {
                        "inputs": {
                          "type": "CONTAINER",
                          "data": "inputs.0"
                        }
                      },
                      "outputs": [
                        {
                          "id": "produce"
                        }
                      ]
                    }
                  ]
                }
            """.replace('__SUM_DIGEST__', SumPrimitive.metadata.query()['digest']), resolver=Resolver()).check()

    def _get_mock_primitive(self, primitive_id, use_set_hyperparamaters=False):
        class _defaultdict(collections.defaultdict):
            def get(self, k, default=None):
                return self[k]

            def __contains__(self, item):
                return True

        def _get_special_hyperparam():
            if use_set_hyperparamaters:
                h = hyperparams.Set(hyperparams.Hyperparameter[object](None), ())
            else:
                h = hyperparams.Hyperparameter[object](None)
            h.value_from_json_structure = lambda x: x
            return h

        class MockMetadata:
            def get_hyperparams(self):
                return self.query()['primitive_code']['class_type_arguments']['Hyperparams']

            def query(self):
                hparams = hyperparams.Hyperparams()
                hparams.configuration = _defaultdict(_get_special_hyperparam)
                arguments = _defaultdict(lambda: {'kind': metadata_base.PrimitiveArgumentKind.PIPELINE})
                produces = _defaultdict(lambda: {'kind': metadata_base.PrimitiveMethodKind.PRODUCE})
                return {
                    'id': primitive_id,
                    'primitive_code': {
                        'class_type_arguments': {
                            'Hyperparams': hparams
                        },
                        'arguments': arguments,
                        'instance_methods': produces
                    }
                }

        class MockPrimitve:
            def __init__(self):
                self.metadata = MockMetadata()

        return MockPrimitve()

    def _quick_build_pipeline_with_real_primitives(self, primitive_dicts: typing.List[dict]) -> pipeline.Pipeline:
        pipe = pipeline.Pipeline()
        pipe.add_input('inputs')

        for primitive_dict in primitive_dicts:
            step = pipeline.PrimitiveStep(primitive=primitive_dict['primitive'])

            for name, data_ref in primitive_dict.get('container_args', {}).items():
                step.add_argument(name, 'CONTAINER', data_ref)
            for name, data in primitive_dict.get('value_args', {}).items():
                step.add_argument(name, 'VALUE', data)

            for name, data_ref in primitive_dict.get('container_hyperparams', {}).items():
                step.add_hyperparameter(name, 'CONTAINER', data_ref)
            for name, data in primitive_dict.get('value_hyperparams', {}).items():
                step.add_hyperparameter(name, 'VALUE', data)

            step.add_output('produce')
            pipe.add_step(step)

        pipe.add_output(name='Output', data_reference=f'steps.{len(primitive_dicts)-1}.produce')
        return pipe

    def test_pipeline_isomorphism_check(self):
        primitive_1 = MockPrimitiveBuilder({
            'dataset': {'type': 'CONTAINER'},
            'mean': {'type': 'CONTAINER'},
        }, {})
        primitive_2 = MockPrimitiveBuilder({
            'a': {'type': 'CONTAINER'},
            'b': {'type': 'CONTAINER'},
        }, {})

        # With hyperparameters
        primitive_h1 = MockPrimitiveBuilder({
            'dataset': {'type': 'CONTAINER'},
            'mean': {'type': 'CONTAINER'},
        }, {
            'index': {'type': 'VALUE'},
            'masks': {'type': 'DATA'},
            'mat': {'type': 'CONTAINER'},
        })

        primitive_h2 = MockPrimitiveBuilder({
            'a': {'type': 'CONTAINER'},
            'b': {'type': 'CONTAINER'},
        }, {
            'v': {'type': 'VALUE'},
            'd': {'type': 'DATA'},
            'p': {'type': 'PRIMITIVE'},
            'c': {'type': 'CONTAINER'},
        })

        builder = MockPipelineBuilder(['input_0', 'input_1', 'input_2'])
        builder.add_primitive(primitive_1.build(dataset='inputs.0', mean='inputs.2'), outputs=['produce'])
        builder.add_primitive(primitive_1.build(dataset='inputs.1', mean='inputs.2'), outputs=['produce'])
        builder.add_primitive(primitive_2.build(a='steps.0.produce', b='steps.1.produce'), outputs=['produce'])
        builder.add_primitive(primitive_2.build(a='steps.1.produce', b='steps.0.produce'), outputs=['produce'])
        builder.add_primitive(primitive_2.build(a='steps.2.produce', b='steps.3.produce'), outputs=['produce'])
        builder.add_output('output', 'steps.4.produce')
        pipeline_1 = builder.build()

        # [Structure invariance test] Another mirrored pipeline
        builder = MockPipelineBuilder(['input_0', 'input_1', 'input_2'])
        builder.add_primitive(primitive_1.build(dataset='inputs.1', mean='inputs.2'), outputs=['produce'])
        builder.add_primitive(primitive_1.build(dataset='inputs.0', mean='inputs.2'), outputs=['produce'])
        builder.add_primitive(primitive_2.build(a='steps.1.produce', b='steps.0.produce'), outputs=['produce'])
        builder.add_primitive(primitive_2.build(a='steps.0.produce', b='steps.1.produce'), outputs=['produce'])
        builder.add_primitive(primitive_2.build(a='steps.2.produce', b='steps.3.produce'), outputs=['produce'])
        builder.add_output('output', 'steps.4.produce')
        pipeline_2 = builder.build()

        # [Primitive output names invariance test] Another pipeline with different output names
        builder = MockPipelineBuilder(['input_0', 'input_1', 'input_2'])
        builder.add_primitive(primitive_1.build(dataset='inputs.1', mean='inputs.2'), outputs=['world'])
        builder.add_primitive(primitive_1.build(dataset='inputs.0', mean='inputs.2'), outputs=['gas'])
        builder.add_primitive(primitive_2.build(a='steps.1.gas', b='steps.0.world'), outputs=['land'])
        builder.add_primitive(primitive_2.build(a='steps.0.world', b='steps.1.gas'), outputs=['produce'])
        builder.add_primitive(primitive_2.build(a='steps.2.land', b='steps.3.produce'), outputs=['moon'])
        builder.add_output('planet', 'steps.4.moon')
        pipeline_3 = builder.build()

        # [Pipeline input names invariance test] Another pipeline with different input names
        builder = MockPipelineBuilder(['cake', 'bread', 'ham'])
        builder.add_primitive(primitive_1.build(dataset='inputs.1', mean='inputs.2'), outputs=['world'])
        builder.add_primitive(primitive_1.build(dataset='inputs.0', mean='inputs.2'), outputs=['gas'])
        builder.add_primitive(primitive_2.build(a='steps.1.gas', b='steps.0.world'), outputs=['land'])
        builder.add_primitive(primitive_2.build(a='steps.0.world', b='steps.1.gas'), outputs=['produce'])
        builder.add_primitive(primitive_2.build(a='steps.2.land', b='steps.3.produce'), outputs=['moon'])
        builder.add_output('planet', 'steps.4.moon')
        pipeline_4 = builder.build()

        self.assertTrue(pipeline_1.hash() == pipeline_2.hash() == pipeline_3.hash() == pipeline_4.hash())
        # Strict order check.
        self.assertFalse(pipeline_1.equals(pipeline_2, strict_order=True))  # Differ in steps order.
        self.assertTrue(pipeline_2.equals(pipeline_3, strict_order=True))  # Only differ in names.

        # Different pipelines
        builder = MockPipelineBuilder(['input_0', 'input_1', 'input_2'])
        builder.add_primitive(primitive_1.build(dataset='inputs.0', mean='inputs.2'), outputs=['produce'])
        builder.add_primitive(primitive_1.build(dataset='inputs.1', mean='inputs.2'), outputs=['produce'])
        builder.add_primitive(primitive_2.build(a='steps.1.produce', b='steps.0.produce'), outputs=['produce'])
        builder.add_primitive(primitive_2.build(a='steps.0.produce', b='steps.1.produce'), outputs=['produce'])
        builder.add_primitive(primitive_2.build(a='steps.2.produce', b='steps.3.produce'), outputs=['produce'])
        builder.add_output('output', 'steps.4.produce')
        pipeline_5 = builder.build()

        builder = MockPipelineBuilder(['input_1', 'input_0', 'input_2'])
        builder.add_primitive(primitive_1.build(dataset='inputs.1', mean='inputs.2'), outputs=['produce'])
        builder.add_primitive(primitive_1.build(dataset='inputs.0', mean='inputs.2'), outputs=['produce'])
        builder.add_primitive(primitive_2.build(a='steps.0.produce', b='steps.1.produce'), outputs=['produce'])
        builder.add_primitive(primitive_2.build(a='steps.1.produce', b='steps.0.produce'), outputs=['produce'])
        builder.add_primitive(primitive_2.build(a='steps.2.produce', b='steps.3.produce'), outputs=['produce'])
        builder.add_output('output', 'steps.4.produce')
        pipeline_inputs_order_matters = builder.build()

        builder = MockPipelineBuilder(['input_0', 'input_1', 'input_2'])
        builder.add_primitive(primitive_1.build(dataset='inputs.0', mean='inputs.2'), outputs=['produce'])
        builder.add_primitive(primitive_1.build(dataset='inputs.1', mean='inputs.2'), outputs=['produce'])
        builder.add_primitive(primitive_2.build(a='steps.0.produce', b='steps.1.produce'), outputs=['produce'])
        builder.add_primitive(primitive_2.build(a='steps.1.produce', b='steps.0.produce'), outputs=['produce'])
        builder.add_primitive(primitive_2.build(a='steps.2.produce', b='steps.3.produce'), outputs=['produce'])
        builder.add_output('output_1', 'steps.4.produce')
        builder.add_output('output_2', 'steps.3.produce')
        pipeline_output_order_matters_1 = builder.build()

        builder = MockPipelineBuilder(['input_0', 'input_1', 'input_2'])
        builder.add_primitive(primitive_1.build(dataset='inputs.0', mean='inputs.2'), outputs=['produce'])
        builder.add_primitive(primitive_1.build(dataset='inputs.1', mean='inputs.2'), outputs=['produce'])
        builder.add_primitive(primitive_2.build(a='steps.0.produce', b='steps.1.produce'), outputs=['produce'])
        builder.add_primitive(primitive_2.build(a='steps.1.produce', b='steps.0.produce'), outputs=['produce'])
        builder.add_primitive(primitive_2.build(a='steps.2.produce', b='steps.3.produce'), outputs=['produce'])
        builder.add_output('output_2', 'steps.3.produce')
        builder.add_output('output_1', 'steps.4.produce')
        pipeline_output_order_matters_2 = builder.build()

        self.assertFalse(pipeline_5.equals(pipeline_1))
        self.assertFalse(pipeline_inputs_order_matters.equals(pipeline_1))
        self.assertFalse(pipeline_output_order_matters_1.equals(pipeline_output_order_matters_2))

        # [Harder structure invariance test that assumes the immutable property of primitives] A extreme test case
        builder = MockPipelineBuilder(['input_0'])
        builder.add_primitive(primitive_1.build(dataset='inputs.0', mean='inputs.0'), outputs=['produce'])
        builder.add_primitive(primitive_1.build(dataset='inputs.0', mean='inputs.0'), outputs=['produce'])
        builder.add_primitive(primitive_2.build(a='steps.0.produce', b='steps.1.produce'), outputs=['produce'])
        builder.add_primitive(primitive_2.build(a='steps.1.produce', b='steps.0.produce'), outputs=['produce'])
        builder.add_primitive(primitive_2.build(a='steps.2.produce', b='steps.3.produce'), outputs=['produce'])
        builder.add_output('output', 'steps.4.produce')
        pipeline_6 = builder.build()

        # pipeline_7 should be same with pipeline_6 because step.0 & step.1 are indistinguishable.
        builder = MockPipelineBuilder(['input_0'])
        builder.add_primitive(primitive_1.build(dataset='inputs.0', mean='inputs.0'), outputs=['produce'])
        builder.add_primitive(primitive_1.build(dataset='inputs.0', mean='inputs.0'), outputs=['produce'])
        builder.add_primitive(primitive_2.build(a='steps.1.produce', b='steps.0.produce'), outputs=['produce'])
        builder.add_primitive(primitive_2.build(a='steps.0.produce', b='steps.1.produce'), outputs=['produce'])
        builder.add_primitive(primitive_2.build(a='steps.2.produce', b='steps.3.produce'), outputs=['produce'])
        builder.add_output('output', 'steps.4.produce')
        pipeline_7 = builder.build()

        self.assertTrue(pipeline_6.equals(pipeline_7))
        self.assertEqual(pipeline_6.hash(), pipeline_7.hash())

        # A pipeline with placeholders
        builder = MockPipelineBuilder(['input_0', 'input_1', 'input_2'])
        builder.add_primitive(primitive_1.build(dataset='inputs.0', mean='inputs.2'), outputs=['produce'])
        builder.add_primitive(primitive_1.build(dataset='inputs.1', mean='inputs.2'), outputs=['produce'])
        builder.add_primitive(primitive_2.build(a='steps.0.produce', b='steps.1.produce'), outputs=['produce'])
        builder.add_placeholder(['steps.1.produce', 'steps.0.produce'], outputs=['produce'])
        builder.add_primitive(primitive_2.build(a='steps.2.produce', b='steps.3.produce'), outputs=['produce'])
        builder.add_output('output', 'steps.4.produce')
        pipeline_placeholder_1 = builder.build()

        # [Placeholder test] Another pipeline with placeholders
        builder = MockPipelineBuilder(['input_0', 'input_1', 'input_2'])
        builder.add_primitive(primitive_1.build(dataset='inputs.1', mean='inputs.2'), outputs=['world'])
        builder.add_primitive(primitive_1.build(dataset='inputs.0', mean='inputs.2'), outputs=['gas'])
        builder.add_primitive(primitive_2.build(a='steps.1.gas', b='steps.0.world'), outputs=['land'])
        builder.add_placeholder(['steps.0.world', 'steps.1.gas'], outputs=['produce'])
        builder.add_primitive(primitive_2.build(a='steps.2.land', b='steps.3.produce'), outputs=['moon'])
        builder.add_output('planet', 'steps.4.moon')
        pipeline_placeholder_2 = builder.build()

        self.assertTrue(pipeline_placeholder_1.equals(pipeline_placeholder_2))

        # [Subgraph expanding test] A pipeline with subpipelines
        builder = MockPipelineBuilder(['steps_0_world', 'steps_1_gas'])
        builder.add_primitive(primitive_2.build(a='inputs.1', b='inputs.0'), outputs=['land'])
        builder.add_placeholder(['inputs.0', 'inputs.1'], outputs=['produce'])
        builder.add_primitive(primitive_2.build(a='steps.0.land', b='steps.1.produce'), outputs=['sun'])
        builder.add_output('blaze', 'steps.2.sun')
        subpipeline_1 = builder.build()
        builder = MockPipelineBuilder(['input_0', 'input_1', 'input_2'])
        builder.add_primitive(primitive_1.build(dataset='inputs.1', mean='inputs.2'), outputs=['world'])
        builder.add_primitive(primitive_1.build(dataset='inputs.0', mean='inputs.2'), outputs=['gas'])
        builder.add_subpipeline(subpipeline_1, ['steps.0.world', 'steps.1.gas'], outputs=['moon'])
        builder.add_output('planet', 'steps.2.moon')
        pipeline_subpipeline_1 = builder.build()

        self.assertTrue(pipeline_placeholder_1.equals(pipeline_subpipeline_1))
        self.assertEqual(pipeline_placeholder_1.hash(), pipeline_subpipeline_1.hash())

        # Pipeline with hyperparameter test
        builder = MockPipelineBuilder(['input_0', 'input_1', 'input_2'])
        builder.add_primitive(primitive_h1.build(dataset='inputs.0', mean='inputs.2', index=0, masks=['inputs.0'], mat='inputs.1'), outputs=['produce'])
        builder.add_primitive(primitive_h1.build(dataset='inputs.1', mean='inputs.2', index=1, masks=['inputs.1'], mat='inputs.2'), outputs=['produce'])
        builder.add_primitive(primitive_h2.build(a='steps.0.produce', b='steps.1.produce', v='aa', p=0, d=['steps.0.produce'], c='steps.1.produce'), outputs=['produce'])
        builder.add_primitive(primitive_h2.build(a='steps.1.produce', b='steps.0.produce', v='bb', p=1, d=['steps.1.produce'], c='steps.0.produce'), outputs=['produce'])
        builder.add_primitive(primitive_h2.build(a='steps.2.produce', b='steps.3.produce', v='cc', p=3, d=['steps.2.produce'], c='steps.3.produce'), outputs=['produce'])
        builder.add_output('output', 'steps.4.produce')
        pipeline_hyperparams_1 = builder.build()

        builder = MockPipelineBuilder(['input_0', 'input_1', 'input_2'])
        builder.add_primitive(primitive_h1.build(dataset='inputs.1', mean='inputs.2', index=1, masks=['inputs.1'], mat='inputs.2'), outputs=['produce'])
        builder.add_primitive(primitive_h1.build(dataset='inputs.0', mean='inputs.2', index=0, masks=['inputs.0'], mat='inputs.1'), outputs=['produce'])
        builder.add_primitive(primitive_h2.build(a='steps.1.produce', b='steps.0.produce', v='aa', p=1, d=['steps.1.produce'], c='steps.0.produce'), outputs=['produce'])
        builder.add_primitive(primitive_h2.build(a='steps.0.produce', b='steps.1.produce', v='bb', p=0, d=['steps.0.produce'], c='steps.1.produce'), outputs=['produce'])
        builder.add_primitive(primitive_h2.build(a='steps.2.produce', b='steps.3.produce', v='cc', p=3, d=['steps.2.produce'], c='steps.3.produce'), outputs=['produce'])
        builder.add_output('output', 'steps.4.produce')
        pipeline_hyperparams_2 = builder.build()

        self.assertTrue(pipeline_hyperparams_1.equals(pipeline_hyperparams_2))
        self.assertEqual(pipeline_hyperparams_1.hash(), pipeline_hyperparams_2.hash())

        # A different pipeline
        builder = MockPipelineBuilder(['input_0', 'input_1', 'input_2'])
        builder.add_primitive(primitive_h1.build(dataset='inputs.1', mean='inputs.2', index=1, masks=['inputs.1'], mat='inputs.2'), outputs=['produce'])
        builder.add_primitive(primitive_h1.build(dataset='inputs.0', mean='inputs.2', index=0, masks=['inputs.0'], mat='inputs.1'), outputs=['produce'])
        builder.add_primitive(primitive_h2.build(a='steps.1.produce', b='steps.0.produce', v='aa', p=0, d=['steps.1.produce'], c='steps.0.produce'), outputs=['produce'])
        builder.add_primitive(primitive_h2.build(a='steps.0.produce', b='steps.1.produce', v='bb', p=1, d=['steps.0.produce'], c='steps.1.produce'), outputs=['produce'])
        builder.add_primitive(primitive_h2.build(a='steps.2.produce', b='steps.3.produce', v='cc', p=3, d=['steps.2.produce'], c='steps.3.produce'), outputs=['produce'])
        builder.add_output('output', 'steps.4.produce')
        pipeline_hyperparams_3 = builder.build()

        self.assertFalse(pipeline_hyperparams_1.equals(pipeline_hyperparams_3))

        # Primitives hyperparameter value encoding test.
        builder = MockPipelineBuilder(['input_0', 'input_1', 'input_2'])
        builder.add_primitive(primitive_h1.build(dataset='inputs.0', mean='inputs.2', index=0, masks=['inputs.0'], mat='inputs.1'), outputs=['produce'])
        builder.add_primitive(primitive_h1.build(dataset='inputs.1', mean='inputs.2', index=1, masks=['inputs.1'], mat='inputs.2'), outputs=['produce'])
        builder.add_primitive(primitive_h2.build(a='steps.0.produce', b='steps.1.produce', v=object(), p=0, d=['steps.0.produce'], c='steps.1.produce'), outputs=['produce'])
        builder.add_primitive(primitive_h2.build(a='steps.1.produce', b='steps.0.produce', v='bb', p=1, d=['steps.1.produce'], c='steps.0.produce'), outputs=['produce'])
        builder.add_primitive(primitive_h2.build(a='steps.2.produce', b='steps.3.produce', v='cc', p=3, d=['steps.2.produce'], c='steps.3.produce'), outputs=['produce'])
        builder.add_output('output', 'steps.4.produce')
        pipeline_hyperparams_encoding = builder.build()
        target_step = pipeline_hyperparams_encoding.steps[2]
        assert isinstance(target_step, pipeline.PrimitiveStep)
        target_step.primitive = self._get_mock_primitive(target_step.primitive_description['id'])
        target_step.primitive_description = None
        repr_1 = pipeline.PipelineHasher(pipeline_hyperparams_1).unique_equivalence_class_repr()
        hasher_2 = pipeline.PipelineHasher(pipeline_hyperparams_encoding)
        hasher_2.graph.step_nodes[3]._serialize_hyperparamter_value = lambda *_: '"aa"'
        self.assertEqual(repr_1, hasher_2.unique_equivalence_class_repr())

        # Orders of sequential hyperparameters matter.
        builder = MockPipelineBuilder(['input_0', 'input_1', 'input_2'])
        builder.add_primitive(primitive_h1.build(dataset='inputs.0', mean='inputs.2', index=0, masks=['inputs.0'], mat='inputs.1'), outputs=['produce'])
        builder.add_primitive(primitive_h1.build(dataset='inputs.1', mean='inputs.2', index=1, masks=['inputs.1'], mat='inputs.2'), outputs=['produce'])
        builder.add_primitive(primitive_h2.build(a='steps.0.produce', b='steps.1.produce', v='aa', p=[0, 1], d=['steps.0.produce'], c='steps.1.produce'), outputs=['produce'])
        builder.add_primitive(primitive_h2.build(a='steps.1.produce', b='steps.0.produce', v='bb', p=1, d=['steps.1.produce', 'steps.2.produce'], c='steps.0.produce'), outputs=['produce'])
        builder.add_primitive(primitive_h2.build(a='steps.2.produce', b='steps.3.produce', v='cc', p=3, d=['steps.2.produce'], c='steps.3.produce'), outputs=['produce'])
        builder.add_output('output', 'steps.4.produce')
        pipeline_hyperparams_sequential_1 = builder.build()

        builder = MockPipelineBuilder(['input_0', 'input_1', 'input_2'])
        builder.add_primitive(primitive_h1.build(dataset='inputs.0', mean='inputs.2', index=0, masks=['inputs.0'], mat='inputs.1'), outputs=['produce'])
        builder.add_primitive(primitive_h1.build(dataset='inputs.1', mean='inputs.2', index=1, masks=['inputs.1'], mat='inputs.2'), outputs=['produce'])
        builder.add_primitive(primitive_h2.build(a='steps.0.produce', b='steps.1.produce', v='aa', p=[0, 1], d=['steps.0.produce'], c='steps.1.produce'), outputs=['produce'])
        builder.add_primitive(primitive_h2.build(a='steps.1.produce', b='steps.0.produce', v='bb', p=1, d=['steps.1.produce', 'steps.2.produce'], c='steps.0.produce'), outputs=['produce'])
        builder.add_primitive(primitive_h2.build(a='steps.2.produce', b='steps.3.produce', v='cc', p=3, d=['steps.2.produce'], c='steps.3.produce'), outputs=['produce'])
        builder.add_output('output', 'steps.4.produce')
        pipeline_hyperparams_sequential_2 = builder.build()

        builder = MockPipelineBuilder(['input_0', 'input_1', 'input_2'])
        builder.add_primitive(primitive_h1.build(dataset='inputs.0', mean='inputs.2', index=0, masks=['inputs.0'], mat='inputs.1'), outputs=['produce'])
        builder.add_primitive(primitive_h1.build(dataset='inputs.1', mean='inputs.2', index=1, masks=['inputs.1'], mat='inputs.2'), outputs=['produce'])
        builder.add_primitive(primitive_h2.build(a='steps.0.produce', b='steps.1.produce', v='aa', p=[1, 0], d=['steps.0.produce'], c='steps.1.produce'), outputs=['produce'])
        builder.add_primitive(primitive_h2.build(a='steps.1.produce', b='steps.0.produce', v='bb', p=1, d=['steps.1.produce', 'steps.2.produce'], c='steps.0.produce'), outputs=['produce'])
        builder.add_primitive(primitive_h2.build(a='steps.2.produce', b='steps.3.produce', v='cc', p=3, d=['steps.2.produce'], c='steps.3.produce'), outputs=['produce'])
        builder.add_output('output', 'steps.4.produce')
        pipeline_hyperparams_sequential_3 = builder.build()

        builder = MockPipelineBuilder(['input_0', 'input_1', 'input_2'])
        builder.add_primitive(primitive_h1.build(dataset='inputs.0', mean='inputs.2', index=0, masks=['inputs.0'], mat='inputs.1'), outputs=['produce'])
        builder.add_primitive(primitive_h1.build(dataset='inputs.1', mean='inputs.2', index=1, masks=['inputs.1'], mat='inputs.2'), outputs=['produce'])
        builder.add_primitive(primitive_h2.build(a='steps.0.produce', b='steps.1.produce', v='aa', p=[0, 1], d=['steps.0.produce'], c='steps.1.produce'), outputs=['produce'])
        builder.add_primitive(primitive_h2.build(a='steps.1.produce', b='steps.0.produce', v='bb', p=1, d=['steps.2.produce', 'steps.1.produce'], c='steps.0.produce'), outputs=['produce'])
        builder.add_primitive(primitive_h2.build(a='steps.2.produce', b='steps.3.produce', v='cc', p=3, d=['steps.2.produce'], c='steps.3.produce'), outputs=['produce'])
        builder.add_output('output', 'steps.4.produce')
        pipeline_hyperparams_sequential_4 = builder.build()

        self.assertTrue(pipeline_hyperparams_sequential_1.equals(pipeline_hyperparams_sequential_2))
        self.assertFalse(pipeline_hyperparams_sequential_1.equals(pipeline_hyperparams_sequential_3))
        self.assertFalse(pipeline_hyperparams_sequential_1.equals(pipeline_hyperparams_sequential_4))

        # `Set` hyperparameters test
        builder = MockPipelineBuilder(['input_0', 'input_1', 'input_2'])
        builder.add_primitive(primitive_h1.build(dataset='inputs.0', mean='inputs.2', index=[0], masks=['inputs.0'], mat='inputs.1'), outputs=['produce'])
        builder.add_primitive(primitive_h1.build(dataset='inputs.1', mean='inputs.2', index=[1], masks=['inputs.1'], mat='inputs.2'), outputs=['produce'])
        builder.add_primitive(primitive_h2.build(a='steps.0.produce', b='steps.1.produce', v=['aa'], p=[0], d=['steps.0.produce'], c='steps.1.produce'), outputs=['produce'])
        builder.add_primitive(primitive_h2.build(a='steps.1.produce', b='steps.0.produce', v=['bb'], p=[1], d=['steps.1.produce'], c='steps.0.produce'), outputs=['produce'])
        builder.add_primitive(primitive_h2.build(a='steps.2.produce', b='steps.3.produce', v=[{'cc': 1, 'dd': 2}, {'ee': 1, 'ff': 2}], p=[3, 2], d=['steps.2.produce'], c='steps.3.produce'), outputs=['produce'])
        builder.add_output('output', 'steps.4.produce')
        pipeline_set_hyperparams_1 = builder.build(primitive_loading=lambda primitive_description: self._get_mock_primitive(primitive_description['id'], True))

        builder = MockPipelineBuilder(['input_0', 'input_1', 'input_2'])
        builder.add_primitive(primitive_h1.build(dataset='inputs.0', mean='inputs.2', index=[0], masks=['inputs.0'], mat='inputs.1'), outputs=['produce'])
        builder.add_primitive(primitive_h1.build(dataset='inputs.1', mean='inputs.2', index=[1], masks=['inputs.1'], mat='inputs.2'), outputs=['produce'])
        builder.add_primitive(primitive_h2.build(a='steps.0.produce', b='steps.1.produce', v=['aa'], p=[0], d=['steps.0.produce'], c='steps.1.produce'), outputs=['produce'])
        builder.add_primitive(primitive_h2.build(a='steps.1.produce', b='steps.0.produce', v=['bb'], p=[1], d=['steps.1.produce'], c='steps.0.produce'), outputs=['produce'])
        builder.add_primitive(primitive_h2.build(a='steps.2.produce', b='steps.3.produce', v=[{'cc': 1, 'dd': 2}, {'ee': 1, 'ff': 2}], p=[2, 3], d=['steps.2.produce'], c='steps.3.produce'), outputs=['produce'])
        builder.add_output('output', 'steps.4.produce')
        pipeline_set_hyperparams_2 = builder.build(primitive_loading=lambda primitive_description: self._get_mock_primitive(primitive_description['id'], True))

        builder = MockPipelineBuilder(['input_0', 'input_1', 'input_2'])
        builder.add_primitive(primitive_h1.build(dataset='inputs.0', mean='inputs.2', index=[0], masks=['inputs.0'], mat='inputs.1'), outputs=['produce'])
        builder.add_primitive(primitive_h1.build(dataset='inputs.1', mean='inputs.2', index=[1], masks=['inputs.1'], mat='inputs.2'), outputs=['produce'])
        builder.add_primitive(primitive_h2.build(a='steps.0.produce', b='steps.1.produce', v=['aa'], p=[0], d=['steps.0.produce'], c='steps.1.produce'), outputs=['produce'])
        builder.add_primitive(primitive_h2.build(a='steps.1.produce', b='steps.0.produce', v=['bb'], p=[1], d=['steps.1.produce'], c='steps.0.produce'), outputs=['produce'])
        builder.add_primitive(primitive_h2.build(a='steps.2.produce', b='steps.3.produce', v=[{'ee': 1, 'ff': 2}, {'cc': 1, 'dd': 2}], p=[3, 2], d=['steps.2.produce'], c='steps.3.produce'), outputs=['produce'])
        builder.add_output('output', 'steps.4.produce')
        pipeline_set_hyperparams_3 = builder.build(primitive_loading=lambda primitive_description: self._get_mock_primitive(primitive_description['id'], True))

        self.assertTrue(pipeline_set_hyperparams_1.equals(pipeline_set_hyperparams_2))
        self.assertFalse(pipeline_set_hyperparams_1.equals(pipeline_set_hyperparams_2, strict_order=True))

        self.assertTrue(pipeline_set_hyperparams_1.equals(pipeline_set_hyperparams_3))
        self.assertFalse(pipeline_set_hyperparams_1.equals(pipeline_set_hyperparams_3, strict_order=True))

    def test_pipeline_isomorphism_check_control_only(self):
        # Pipelines with different tuning hyperparameters should still be equal with
        # the only_control_hyperparams flag set.
        pipeline_diff_tuningparams_a = self._quick_build_pipeline_with_real_primitives([
            {
                'primitive': DatasetToDataFramePrimitive,
                'container_args': {'inputs': 'inputs.0'},
            }, {
                'primitive': RandomForestClassifierPrimitive,
                'container_args': {'inputs': 'steps.0.produce', 'outputs': 'steps.0.produce'},
                'value_hyperparams': {'n_estimators': 250}
            }
        ])
        pipeline_diff_tuningparams_b = self._quick_build_pipeline_with_real_primitives([
            {
                'primitive': DatasetToDataFramePrimitive,
                'container_args': {'inputs': 'inputs.0'},
            }, {
                'primitive': RandomForestClassifierPrimitive,
                'container_args': {'inputs': 'steps.0.produce', 'outputs': 'steps.0.produce'},
                'value_hyperparams': {'n_estimators': 500}  # different value
            }
        ])
        self.assertFalse(pipeline_diff_tuningparams_a.equals(pipeline_diff_tuningparams_b))
        self.assertFalse(pipeline_diff_tuningparams_b.equals(pipeline_diff_tuningparams_a))
        self.assertTrue(pipeline_diff_tuningparams_a.equals(pipeline_diff_tuningparams_b, only_control_hyperparams=True))
        self.assertTrue(pipeline_diff_tuningparams_b.equals(pipeline_diff_tuningparams_a, only_control_hyperparams=True))
        pipeline_diff_tuningparams_a_copy = copy.deepcopy(pipeline_diff_tuningparams_a)
        self.assertTrue(pipeline_diff_tuningparams_a.equals(pipeline_diff_tuningparams_a_copy))
        self.assertTrue(pipeline_diff_tuningparams_a.equals(pipeline_diff_tuningparams_a_copy, only_control_hyperparams=True))

        # Pipelines with different control hyperparameters should not be equal,
        # even with the only_control_hyperparams flag set.
        pipeline_diff_controlparams_a = self._quick_build_pipeline_with_real_primitives([
            {
                'primitive': DatasetToDataFramePrimitive,
                'container_args': {'inputs': 'inputs.0'},
            }, {
                'primitive': ColumnParserPrimitive,
                'container_args': {'inputs': 'steps.0.produce'},
                'value_hyperparams': {'return_result': 'replace'}
            }
        ])
        pipeline_diff_controlparams_b = self._quick_build_pipeline_with_real_primitives([
            {
                'primitive': DatasetToDataFramePrimitive,
                'container_args': {'inputs': 'inputs.0'},
            }, {
                'primitive': ColumnParserPrimitive,
                'container_args': {'inputs': 'steps.0.produce'},
                'value_hyperparams': {'return_result': 'new'}  # different value
            }
        ])
        self.assertFalse(pipeline_diff_controlparams_a.equals(pipeline_diff_controlparams_b))
        self.assertFalse(pipeline_diff_controlparams_b.equals(pipeline_diff_controlparams_a))
        self.assertFalse(pipeline_diff_controlparams_a.equals(pipeline_diff_controlparams_b, only_control_hyperparams=True))
        self.assertFalse(pipeline_diff_controlparams_b.equals(pipeline_diff_controlparams_a, only_control_hyperparams=True))
        pipeline_diff_controlparams_a_copy = copy.deepcopy(pipeline_diff_controlparams_a)
        self.assertTrue(pipeline_diff_controlparams_a.equals(pipeline_diff_controlparams_a_copy))
        self.assertTrue(pipeline_diff_controlparams_a.equals(pipeline_diff_controlparams_a_copy, only_control_hyperparams=True))


if __name__ == '__main__':
    unittest.main()
