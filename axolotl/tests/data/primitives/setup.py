import os
from setuptools import setup, find_packages

PACKAGE_NAME = 'test_primitives'


def read_package_variable(key):
    """Read the value of a variable from the package without importing."""
    module_path = os.path.join(PACKAGE_NAME, '__init__.py')
    with open(module_path) as module:
        for line in module:
            parts = line.strip().split(' ')
            if parts and parts[0] == key:
                return parts[-1].strip("'")
    raise KeyError("'{0}' not found in '{1}'".format(key, module_path))


setup(
    name=PACKAGE_NAME,
    version=read_package_variable('__version__'),
    description='Test primitives',
    author=read_package_variable('__author__'),
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    install_requires=[
        'd3m',
    ],
    url='https://gitlab.com/datadrivendiscovery/tests-data',
    keywords='d3m_primitive',
    entry_points={
        'd3m.primitives': [
            'regression.monomial.Test = test_primitives.monomial:MonomialPrimitive',
            'operator.increment.Test = test_primitives.increment:IncrementPrimitive',
            'operator.sum.Test = test_primitives.sum:SumPrimitive',
            'data_generation.random.Test = test_primitives.random:RandomPrimitive',
            'operator.primitive_sum.Test = test_primitives.primitive_sum:PrimitiveSumPrimitive',
            'operator.null.TransformerTest = test_primitives.null:NullTransformerPrimitive',
            'operator.null.UnsupervisedLearnerTest = test_primitives.null:NullUnsupervisedLearnerPrimitive',
            'classification.random_classifier.Test = test_primitives.random_classifier:RandomClassifierPrimitive',
            'evaluation.compute_scores.Test = test_primitives.fake_score:FakeScorePrimitive',
        ],
    },
)
