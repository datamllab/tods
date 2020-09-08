import copy
import logging
import os.path
import pkg_resources
import sys
import types
import unittest

COMMON_PRIMITIVES_DIR = os.path.join(os.path.dirname(__file__), 'common-primitives')
# NOTE: This insertion should appear before any code attempting to resolve or load primitives,
# so the git submodule version of `common-primitives` is looked at first.
sys.path.insert(0, COMMON_PRIMITIVES_DIR)

from common_primitives.column_parser import ColumnParserPrimitive

from d3m import container, index, utils
from d3m.metadata import base as metadata_base, hyperparams, pipeline_run
from d3m.primitive_interfaces import base, transformer

Inputs = container.DataFrame
Outputs = container.DataFrame


class Hyperparams(hyperparams.Hyperparams):
    pass


def create_primitive(primitive_id, python_path):
    # Silence any validation warnings.
    with utils.silence():
        class Primitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
            metadata = metadata_base.PrimitiveMetadata({
                'id': primitive_id,
                'version': '0.1.0',
                'name': "Test Primitive",
                'python_path': python_path,
                'algorithm_types': [
                    metadata_base.PrimitiveAlgorithmType.PRINCIPAL_COMPONENT_ANALYSIS,
                ],
                'primitive_family': metadata_base.PrimitiveFamily.FEATURE_SELECTION,
            })

            def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
                pass

        return Primitive


FooBar2Primitive = create_primitive('e328012a-56f3-4da4-a422-2a0ade5d05b0', 'd3m.primitives.foo2.bar2.FooBar2Primitive')
FooBar3Primitive = create_primitive('266acf13-b7c3-4115-aaff-230971624a7d', 'd3m.primitives.foo3.bar3.FooBar3Primitive')
FooBar4Primitive = create_primitive('ab699c0f-434a-43eb-ad4a-f1e669cac50e', 'd3m.primitives.foo3.bar3')


class TestIndex(unittest.TestCase):
    def test_register(self):
        FooBarPrimitive = create_primitive('e2fc24f8-5b32-4759-be5b-8126a42522a3', 'd3m.primitives.foo.bar.FooBarPrimitive')

        # To hide any logging or stdout output.
        with self.assertLogs(level=logging.DEBUG) as cm:
            with utils.redirect_to_logging():
                index.register_primitive('d3m.primitives.foo.bar.FooBarPrimitive', FooBarPrimitive)

                # Just to log something, otherwise "assertLogs" can fail.
                logging.getLogger().debug("Start test.")

        index.get_primitive('d3m.primitives.foo.bar.FooBarPrimitive')

    def test_entrypoint(self):
        working_set_entries = copy.copy(pkg_resources.working_set.entries)
        working_set_entry_keys = copy.copy(pkg_resources.working_set.entry_keys)
        working_set_by_key = copy.copy(pkg_resources.working_set.by_key)

        try:
            distribution = pkg_resources.Distribution(__file__)
            entry_point = pkg_resources.EntryPoint.parse('foo2.bar2.FooBar2Primitive = test_index:FooBar2Primitive', dist=distribution)
            distribution._ep_map = {'d3m.primitives': {'foo2.bar2.FooBar2Primitive': entry_point}}
            pkg_resources.working_set.add(distribution)

            python_path = 'd3m.primitives.foo2.bar2.FooBar2Primitive'

            self.assertIn(python_path, index.search())

            self.assertIs(index.get_primitive(python_path), FooBar2Primitive)

        finally:
            pkg_resources.working_set.entries = working_set_entries
            pkg_resources.working_set.entry_keys = working_set_entry_keys
            pkg_resources.working_set.by_key = working_set_by_key

    def test_entrypoint_conflict(self):
        working_set_entries = copy.copy(pkg_resources.working_set.entries)
        working_set_entry_keys = copy.copy(pkg_resources.working_set.entry_keys)
        working_set_by_key = copy.copy(pkg_resources.working_set.by_key)

        try:
            distribution = pkg_resources.Distribution(__file__)
            distribution._ep_map = {
                'd3m.primitives': {
                    'foo3.bar3': pkg_resources.EntryPoint.parse('foo3.bar3 = test_index:FooBar4Primitive', dist=distribution),
                    'foo3.bar3.FooBar3Primitive': pkg_resources.EntryPoint.parse('foo3.bar3.FooBar3Primitive = test_index:FooBar3Primitive', dist=distribution),
                },
            }
            pkg_resources.working_set.add(distribution)

            with self.assertLogs(level=logging.WARNING) as cm:
                from d3m.primitives.foo3 import bar3
                from d3m.primitives.foo3.bar3 import FooBar3Primitive as primitive

            self.assertIsInstance(bar3, types.ModuleType)
            self.assertIs(primitive, FooBar3Primitive)

            self.assertEqual(len(cm.records), 1)
            self.assertEqual(cm.records[0].msg, 'An entry point for a primitive is conflicting with another entry point which has it as a module: %(entry_point_name)s')

        finally:
            pkg_resources.working_set.entries = working_set_entries
            pkg_resources.working_set.entry_keys = working_set_entry_keys
            pkg_resources.working_set.by_key = working_set_by_key

    def test_validate(self):
        # To hide any logging or stdout output.
        with utils.silence():
            index.register_primitive('d3m.primitives.data_transformation.column_parser.Common', ColumnParserPrimitive)

        primitive = index.get_primitive_by_id('d510cb7a-1782-4f51-b44c-58f0236e47c7')

        primitive_description = primitive.metadata.to_json_structure()

        pipeline_run.validate_primitive(primitive_description)


if __name__ == '__main__':
    unittest.main()
