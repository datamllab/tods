import os
import sys

TEST_PRIMITIVES_DIR = os.path.join(os.path.dirname(__file__), '..')

sys.path.insert(0, TEST_PRIMITIVES_DIR)

from test_pipeline import MockPrimitiveBuilder, MockPipelineBuilder


class Primitive:
    def time_hash(self):
        primitive_1 = MockPrimitiveBuilder({
            'dataset': {'type': 'CONTAINER'},
            'mean': {'type': 'CONTAINER'},
        }, {})
        primitive_2 = MockPrimitiveBuilder({
            'a': {'type': 'CONTAINER'},
            'b': {'type': 'CONTAINER'},
        }, {})

        builder = MockPipelineBuilder(['input_0', 'input_1', 'input_2'])
        builder.add_primitive(primitive_1.build(dataset='inputs.0', mean='inputs.2'), outputs=['produce'])
        builder.add_primitive(primitive_1.build(dataset='inputs.1', mean='inputs.2'), outputs=['produce'])
        builder.add_primitive(primitive_2.build(a='steps.0.produce', b='steps.1.produce'), outputs=['produce'])
        builder.add_primitive(primitive_2.build(a='steps.1.produce', b='steps.0.produce'), outputs=['produce'])
        builder.add_primitive(primitive_2.build(a='steps.2.produce', b='steps.3.produce'), outputs=['produce'])
        builder.add_output('output', 'steps.4.produce')
        pipeline_1 = builder.build()

        for _ in range(1000):
            pipeline_1.hash()
