import typing
import unittest
import logging

from d3m import container, exceptions, utils
from d3m.metadata import base as metadata_base, hyperparams, params
from d3m.primitive_interfaces import base, transformer, unsupervised_learning

Inputs = container.List
Outputs = container.List


class Hyperparams(hyperparams.Hyperparams):
    pass


class TestPrimitiveValidation(unittest.TestCase):
    def test_multi_produce_missing_argument(self):
        with self.assertRaisesRegex(exceptions.InvalidPrimitiveCodeError, '\'multi_produce\' method arguments have to be an union of all arguments of all produce methods, but it does not accept all expected arguments'):
            # Silence any validation warnings.
            with utils.silence():
                class TestPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
                    metadata = metadata_base.PrimitiveMetadata({
                        'id': '67568a80-dec2-4597-a10f-39afb13d3b9c',
                        'version': '0.1.0',
                        'name': "Test Primitive",
                        'python_path': 'd3m.primitives.test.TestPrimitive',
                        'algorithm_types': [
                            metadata_base.PrimitiveAlgorithmType.NUMERICAL_METHOD,
                        ],
                        'primitive_family': metadata_base.PrimitiveFamily.OPERATOR,
                    })

                    def produce(self, *, inputs: Inputs, second_inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
                        pass

    def test_fit_multi_produce_missing_argument(self):
        with self.assertRaisesRegex(exceptions.InvalidPrimitiveCodeError, '\'fit_multi_produce\' method arguments have to be an union of all arguments of \'set_training_data\' method and all produce methods, but it does not accept all expected arguments'):
            # Silence any validation warnings.
            with utils.silence():
                class TestPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
                    metadata = metadata_base.PrimitiveMetadata({
                        'id': '67568a80-dec2-4597-a10f-39afb13d3b9c',
                        'version': '0.1.0',
                        'name': "Test Primitive",
                        'python_path': 'd3m.primitives.test.TestPrimitive',
                        'algorithm_types': [
                            metadata_base.PrimitiveAlgorithmType.NUMERICAL_METHOD,
                        ],
                        'primitive_family': metadata_base.PrimitiveFamily.OPERATOR,
                    })

                    def produce(self, *, inputs: Inputs, second_inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
                        pass

                    def multi_produce(self, *, produce_methods: typing.Sequence[str], inputs: Inputs, second_inputs: Inputs, timeout: float = None, iterations: int = None) -> base.MultiCallResult:
                        pass

    def test_multi_produce_extra_argument(self):
        with self.assertRaisesRegex(exceptions.InvalidPrimitiveCodeError, '\'multi_produce\' method arguments have to be an union of all arguments of all produce methods, but it accepts unexpected arguments'):
            # Silence any validation warnings.
            with utils.silence():
                class TestPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
                    metadata = metadata_base.PrimitiveMetadata({
                        'id': '67568a80-dec2-4597-a10f-39afb13d3b9c',
                        'version': '0.1.0',
                        'name': "Test Primitive",
                        'python_path': 'd3m.primitives.test.TestPrimitive',
                        'algorithm_types': [
                            metadata_base.PrimitiveAlgorithmType.NUMERICAL_METHOD,
                        ],
                        'primitive_family': metadata_base.PrimitiveFamily.OPERATOR,
                    })

                    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
                        pass

                    def multi_produce(self, *, produce_methods: typing.Sequence[str], inputs: Inputs, second_inputs: Inputs, timeout: float = None, iterations: int = None) -> base.MultiCallResult:
                        pass

    def test_fit_multi_produce_extra_argument(self):
        with self.assertRaisesRegex(exceptions.InvalidPrimitiveCodeError, '\'fit_multi_produce\' method arguments have to be an union of all arguments of \'set_training_data\' method and all produce methods, but it accepts unexpected arguments'):
            # Silence any validation warnings.
            with utils.silence():
                class TestPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
                    metadata = metadata_base.PrimitiveMetadata({
                        'id': '67568a80-dec2-4597-a10f-39afb13d3b9c',
                        'version': '0.1.0',
                        'name': "Test Primitive",
                        'python_path': 'd3m.primitives.test.TestPrimitive',
                        'algorithm_types': [
                            metadata_base.PrimitiveAlgorithmType.NUMERICAL_METHOD,
                        ],
                        'primitive_family': metadata_base.PrimitiveFamily.OPERATOR,
                    })

                    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
                        pass

                    def fit_multi_produce(self, *, produce_methods: typing.Sequence[str], inputs: Inputs, second_inputs: Inputs, timeout: float = None, iterations: int = None) -> base.MultiCallResult:
                        pass

    def test_produce_using_produce_methods(self):
        with self.assertRaisesRegex(exceptions.InvalidPrimitiveCodeError, 'Produce method cannot use \'produce_methods\' argument'):
            # Silence any validation warnings.
            with utils.silence():
                class TestPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
                    metadata = metadata_base.PrimitiveMetadata({
                        'id': '67568a80-dec2-4597-a10f-39afb13d3b9c',
                        'version': '0.1.0',
                        'name': "Test Primitive",
                        'python_path': 'd3m.primitives.test.TestPrimitive',
                        'algorithm_types': [
                            metadata_base.PrimitiveAlgorithmType.NUMERICAL_METHOD,
                        ],
                        'primitive_family': metadata_base.PrimitiveFamily.OPERATOR,
                    })

                    def produce(self, *, inputs: Inputs, produce_methods: typing.Sequence[str], timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
                        pass

    def test_hyperparams_to_tune(self):
        with self.assertRaisesRegex(exceptions.InvalidMetadataError, 'Hyper-parameter in \'hyperparams_to_tune\' metadata does not exist'):
            # Silence any validation warnings.
            with utils.silence():
                class TestPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
                    metadata = metadata_base.PrimitiveMetadata({
                        'id': '67568a80-dec2-4597-a10f-39afb13d3b9c',
                        'version': '0.1.0',
                        'name': "Test Primitive",
                        'python_path': 'd3m.primitives.test.TestPrimitive',
                        'algorithm_types': [
                            metadata_base.PrimitiveAlgorithmType.NUMERICAL_METHOD,
                        ],
                        'primitive_family': metadata_base.PrimitiveFamily.OPERATOR,
                        'hyperparams_to_tune': [
                            'foobar',
                        ]
                    })

                    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
                        pass

    def test_inputs_across_samples(self):
        with self.assertRaisesRegex(exceptions.InvalidPrimitiveCodeError, 'Method \'.*\' has an argument \'.*\' set as computing across samples, but it does not exist'):
            # Silence any validation warnings.
            with utils.silence():
                class TestPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
                    metadata = metadata_base.PrimitiveMetadata({
                        'id': '67568a80-dec2-4597-a10f-39afb13d3b9c',
                        'version': '0.1.0',
                        'name': "Test Primitive",
                        'python_path': 'd3m.primitives.test.TestPrimitive',
                        'algorithm_types': [
                            metadata_base.PrimitiveAlgorithmType.NUMERICAL_METHOD,
                        ],
                        'primitive_family': metadata_base.PrimitiveFamily.OPERATOR,
                        'hyperparams_to_tune': [
                            'foobar',
                        ]
                    })

                    @base.inputs_across_samples('foobar')
                    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
                        pass

        with self.assertRaisesRegex(exceptions.InvalidPrimitiveCodeError, 'Method \'.*\' has an argument \'.*\' set as computing across samples, but it is not a PIPELINE argument'):
            # Silence any validation warnings.
            with utils.silence():
                class TestPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
                    metadata = metadata_base.PrimitiveMetadata({
                        'id': '67568a80-dec2-4597-a10f-39afb13d3b9c',
                        'version': '0.1.0',
                        'name': "Test Primitive",
                        'python_path': 'd3m.primitives.test.TestPrimitive',
                        'algorithm_types': [
                            metadata_base.PrimitiveAlgorithmType.NUMERICAL_METHOD,
                        ],
                        'primitive_family': metadata_base.PrimitiveFamily.OPERATOR,
                        'hyperparams_to_tune': [
                            'foobar',
                        ]
                    })

                    @base.inputs_across_samples('timeout')
                    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
                        pass

    def test_can_detect_too_many_package_components(self):
        logger = logging.getLogger('d3m.metadata.base')

        # Ensure a warning message is generated for too many package components
        with self.assertLogs(logger=logger, level=logging.DEBUG) as cm:
            metadata_base.PrimitiveMetadata()._validate_namespace_compliance('d3m.primitives.classification.random_forest.SKLearn.toomany', metadata_base.PrimitiveFamily.CLASSIFICATION)

        self.assertEqual(len(cm.records), 1)
        self.assertEqual(cm.records[0].msg,
                         "%(python_path)s: Primitive's Python path does not adhere to d3m.primitives namespace specification. "
                         "Reason: must have 5 segments.")

        # Ensure a warning message is NOT generated for an acceptable number of components
        with self.assertLogs(logger=logger, level=logging.DEBUG) as cm:
            logger.debug("Dummy log")
            metadata_base.PrimitiveMetadata()._validate_namespace_compliance('d3m.primitives.classification.random_forest.SKLearn', metadata_base.PrimitiveFamily.CLASSIFICATION)

        self.assertEqual(len(cm.records), 1)

    def test_with_string_instead_of_enum(self):
        logger = logging.getLogger(metadata_base.__name__)

        # Ensure a warning message is NOT generated for an acceptable number of components
        with self.assertLogs(logger=logger, level=logging.DEBUG) as cm:
            logger.debug("Dummy log")
            metadata_base.PrimitiveMetadata()._validate_namespace_compliance('d3m.primitives.classification.random_forest.SKLearn', metadata_base.PrimitiveFamily.CLASSIFICATION.name)

        self.assertEqual(len(cm.records), 1)

    def test_can_detect_too_few_package_components(self):
        logger = logging.getLogger(metadata_base.__name__)

        # Ensure a warning message is generated for too few package components
        with self.assertLogs(logger=logger, level=logging.DEBUG) as cm:
            metadata_base.PrimitiveMetadata()._validate_namespace_compliance('d3m.primitives.classification.too_few', metadata_base.PrimitiveFamily.CLASSIFICATION)

        self.assertEqual(len(cm.records), 1)
        self.assertEqual(cm.records[0].msg,
                         "%(python_path)s: Primitive's Python path does not adhere to d3m.primitives namespace specification. "
                         "Reason: must have 5 segments.")

        # Ensure a warning message is NOT generated for an acceptable number of components
        with self.assertLogs(logger=logger, level=logging.DEBUG) as cm:
            logger.debug("Dummy log")
            metadata_base.PrimitiveMetadata()._validate_namespace_compliance('d3m.primitives.classification.random_forest.SKLearn', metadata_base.PrimitiveFamily.CLASSIFICATION)

        self.assertEqual(len(cm.records), 1)

    def test_can_detect_bad_primitive_family(self):
        logger = logging.getLogger(metadata_base.__name__)

        # Ensure a warning message is generated for a bad primitive family
        with self.assertLogs(logger=logger, level=logging.DEBUG) as cm:
            metadata_base.PrimitiveMetadata()._validate_namespace_compliance('d3m.primitives.bad_family.random_forest.SKLearn', metadata_base.PrimitiveFamily.CLASSIFICATION)

        self.assertEqual(len(cm.records), 1)
        self.assertEqual(cm.records[0].msg,
                         "%(python_path)s: Primitive's Python path does not adhere to d3m.primitives namespace specification."
                         " Reason: primitive family segment must match primitive's primitive family.")

        # Ensure a warning message is NOT generated for an acceptable primitive family
        with self.assertLogs(logger=logger, level=logging.DEBUG) as cm:
            logger.debug("Dummy log")
            metadata_base.PrimitiveMetadata()._validate_namespace_compliance('d3m.primitives.classification.random_forest.SKLearn', metadata_base.PrimitiveFamily.CLASSIFICATION)

        self.assertEqual(len(cm.records), 1)

    def test_can_detect_bad_primitive_name(self):
        logger = logging.getLogger(metadata_base.__name__)

        # Ensure a warning message is generated for a bad primitive name
        with self.assertLogs(logger=logger, level=logging.DEBUG) as cm:
            metadata_base.PrimitiveMetadata()._validate_namespace_compliance('d3m.primitives.classification.bad_name.SKLearn', metadata_base.PrimitiveFamily.CLASSIFICATION)

        self.assertEqual(len(cm.records), 1)
        self.assertEqual(cm.records[0].msg,
                         "%(python_path)s: Primitive's Python path does not adhere to d3m.primitives namespace specification. "
                         "Reason: must have a known primitive name segment.")

        # Ensure a warning message is NOT generated for an acceptable primitive name
        with self.assertLogs(logger=logger, level=logging.DEBUG) as cm:
            logger.debug("Dummy log")
            metadata_base.PrimitiveMetadata()._validate_namespace_compliance('d3m.primitives.classification.random_forest.SKLearn', metadata_base.PrimitiveFamily.CLASSIFICATION)

        self.assertEqual(len(cm.records), 1)

    def test_can_detect_kind_not_capitalized(self):
        logger = logging.getLogger(metadata_base.__name__)

        # Ensure a warning message is generated for a primitive kind not capitalized properly
        with self.assertLogs(logger=logger, level=logging.DEBUG) as cm:
            metadata_base.PrimitiveMetadata()._validate_namespace_compliance('d3m.primitives.classification.random_forest.sklearn', metadata_base.PrimitiveFamily.CLASSIFICATION)

        self.assertEqual(len(cm.records), 1)
        self.assertEqual(cm.records[0].msg,
                         "%(python_path)s: Primitive's Python path does not adhere to d3m.primitives namespace specification. "
                         "Reason: primitive kind segment must start with upper case.")

        # Ensure a warning message is NOT generated for an acceptable primitive kind
        with self.assertLogs(logger=logger, level=logging.DEBUG) as cm:
            logger.debug("Dummy log")
            metadata_base.PrimitiveMetadata()._validate_namespace_compliance('d3m.primitives.classification.random_forest.SKLearn', metadata_base.PrimitiveFamily.CLASSIFICATION)

        self.assertEqual(len(cm.records), 1)

    def test_will_generate_warning_for_missing_contact(self):
        logger = logging.getLogger(metadata_base.__name__)

        bad_metadata = metadata_base.PrimitiveMetadata({
            'id': 'id',
            'version': '0.1.0',
            'name': "Test Primitive",
            'python_path': 'path',
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.PRINCIPAL_COMPONENT_ANALYSIS,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.FEATURE_SELECTION,
            'installation': [{
                'type': metadata_base.PrimitiveInstallationType.PIP,
                'package': 'foobar',
                'version': '0.1.0',
            }],
            'source': {
                'name': 'Test author',
                # 'contact': 'mailto:test@example.com',
                'uris': 'http://someplace'
            }
        })

        good_metadata = metadata_base.PrimitiveMetadata({
            'id': 'id',
            'version': '0.1.0',
            'name': "Test Primitive",
            'python_path': 'path',
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.PRINCIPAL_COMPONENT_ANALYSIS,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.FEATURE_SELECTION,
            'installation': [{
                'type': metadata_base.PrimitiveInstallationType.PIP,
                'package': 'foobar',
                'version': '0.1.0',
            }],
            'source': {
                'name': 'Test author',
                'contact': 'mailto:test@example.com',
                'uris': 'http://someplace'
            }
        })

        # Ensure a warning message is generated for a primitive with no contact specified in the metadata.source
        with self.assertLogs(logger=logger, level=logging.DEBUG) as cm:
            metadata_base.PrimitiveMetadata()._validate_contact_information(bad_metadata.query())

        self.assertEqual(len(cm.records), 1)
        self.assertEqual(cm.records[0].msg, "%(python_path)s: Contact information such as the email address of the author (e.g., \"mailto:author@example.com\") should be specified in primitive metadata in its \"source.contact\" field.")

        # Ensure a warning message is NOT generated for a primitive with a contact specified in the metadata.source
        with self.assertLogs(logger=logger, level=logging.DEBUG) as cm:
            logger.debug("Dummy log")
            metadata_base.PrimitiveMetadata()._validate_contact_information(good_metadata.query())

        self.assertEqual(len(cm.records), 1)

    def test_will_generate_warning_for_empty_contact(self):
        logger = logging.getLogger(metadata_base.__name__)

        bad_metadata = metadata_base.PrimitiveMetadata({
            'id': 'id',
            'version': '0.1.0',
            'name': "Test Primitive",
            'python_path': 'path',
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.PRINCIPAL_COMPONENT_ANALYSIS,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.FEATURE_SELECTION,
            'installation': [{
                'type': metadata_base.PrimitiveInstallationType.PIP,
                'package': 'foobar',
                'version': '0.1.0',
            }],
            'source': {
                'name': 'Test author',
                'contact': '',
                'uris': ['http://someplace']
            }
        })

        good_metadata = metadata_base.PrimitiveMetadata({
            'id': 'id',
            'version': '0.1.0',
            'name': "Test Primitive",
            'python_path': 'path',
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.PRINCIPAL_COMPONENT_ANALYSIS,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.FEATURE_SELECTION,
            'installation': [{
                'type': metadata_base.PrimitiveInstallationType.PIP,
                'package': 'foobar',
                'version': '0.1.0',
            }],
            'source': {
                'name': 'Test author',
                'contact': 'mailto:test@example.com',
                'uris': ['http://someplace']
            }
        })

        # Ensure a warning message is generated for a primitive with empty contact specified in the metadata.source.
        with self.assertLogs(logger=logger, level=logging.DEBUG) as cm:
            metadata_base.PrimitiveMetadata()._validate_contact_information(bad_metadata.query())

        self.assertEqual(len(cm.records), 1)
        self.assertEqual(cm.records[0].msg, "%(python_path)s: Contact information such as the email address of the author (e.g., \"mailto:author@example.com\") should be specified in primitive metadata in its \"source.contact\" field.")

        # Ensure a warning message is NOT generated when a contact value is specified.
        with self.assertLogs(logger=logger, level=logging.DEBUG) as cm:
            logger.debug("Dummy log")
            metadata_base.PrimitiveMetadata()._validate_contact_information(good_metadata.query())

        self.assertEqual(len(cm.records), 1)

    def test_will_not_generate_missing_contact_warning_when_installation_not_specified(self):
        logger = logging.getLogger(metadata_base.__name__)

        good_metadata = metadata_base.PrimitiveMetadata({
            'id': 'id',
            'version': '0.1.0',
            'name': "Test Primitive",
            'python_path': 'path',
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.PRINCIPAL_COMPONENT_ANALYSIS,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.FEATURE_SELECTION,
            'source': {
                'name': 'Test author',
                'uris': ['http://someplace']
            }
        })

        # Ensure a warning message is NOT generated when a contact value is not specified when installation is also
        # not specified.
        with self.assertLogs(logger=logger, level=logging.DEBUG) as cm:
            logger.debug("Dummy log")
            metadata_base.PrimitiveMetadata()._validate_contact_information(good_metadata.query())

        self.assertEqual(len(cm.records), 1)

    def test_will_generate_warning_for_missing_uris(self):
        logger = logging.getLogger(metadata_base.__name__)

        bad_metadata = metadata_base.PrimitiveMetadata({
            'id': 'id',
            'version': '0.1.0',
            'name': "Test Primitive",
            'python_path': 'path',
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.PRINCIPAL_COMPONENT_ANALYSIS,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.FEATURE_SELECTION,
            'installation': [{
                'type': metadata_base.PrimitiveInstallationType.PIP,
                'package': 'foobar',
                'version': '0.1.0',
            }],
            'source': {
                'name': 'Test author',
                'contact': 'mailto:test@example.com',
            }
        })

        good_metadata = metadata_base.PrimitiveMetadata({
            'id': 'id',
            'version': '0.1.0',
            'name': "Test Primitive",
            'python_path': 'path',
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.PRINCIPAL_COMPONENT_ANALYSIS,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.FEATURE_SELECTION,
            'installation': [{
                'type': metadata_base.PrimitiveInstallationType.PIP,
                'package': 'foobar',
                'version': '0.1.0',
            }],
            'source': {
                'name': 'Test author',
                'contact': 'mailto:test@example.com',
                'uris': ['http://someplace'],
            }
        })

        # Ensure a warning message is generated for a primitive with no uris specified in the metadata.source.
        with self.assertLogs(logger=logger, level=logging.DEBUG) as cm:
            metadata_base.PrimitiveMetadata()._validate_contact_information(bad_metadata.query())

        self.assertEqual(len(cm.records), 1)
        self.assertEqual(cm.records[0].msg, "%(python_path)s: A bug reporting URI should be specified in primitive metadata in its \"source.uris\" field.")

        # Ensure a warning message is NOT generated when uris are specified in the metadata.source.
        with self.assertLogs(logger=logger, level=logging.DEBUG) as cm:
            logger.debug("Dummy log")
            metadata_base.PrimitiveMetadata()._validate_contact_information(good_metadata.query())

        self.assertEqual(len(cm.records), 1)

    def test_will_generate_warning_for_empty_uris(self):
        logger = logging.getLogger(metadata_base.__name__)

        bad_metadata = metadata_base.PrimitiveMetadata({
            'id': 'id',
            'version': '0.1.0',
            'name': "Test Primitive",
            'python_path': 'path',
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.PRINCIPAL_COMPONENT_ANALYSIS,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.FEATURE_SELECTION,
            'installation': [{
                'type': metadata_base.PrimitiveInstallationType.PIP,
                'package': 'foobar',
                'version': '0.1.0',
            }],
            'source': {
                'name': 'Test author',
                'contact': 'mailto:test@example.com',
                'uris': [],
            }
        })

        good_metadata = metadata_base.PrimitiveMetadata({
            'id': 'id',
            'version': '0.1.0',
            'name': "Test Primitive",
            'python_path': 'path',
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.PRINCIPAL_COMPONENT_ANALYSIS,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.FEATURE_SELECTION,
            'installation': [{
                'type': metadata_base.PrimitiveInstallationType.PIP,
                'package': 'foobar',
                'version': '0.1.0',
            }],
            'source': {
                'name': 'Test author',
                'contact': 'mailto:test@example.com',
                'uris': ['http://someplace'],
            }
        })

        # Ensure a warning message is generated for a primitive with empty uris specified in the metadata.source.
        with self.assertLogs(logger=logger, level=logging.DEBUG) as cm:
            metadata_base.PrimitiveMetadata()._validate_contact_information(bad_metadata.query())

        self.assertEqual(len(cm.records), 1)
        self.assertEqual(cm.records[0].msg, "%(python_path)s: A bug reporting URI should be specified in primitive metadata in its \"source.uris\" field.")

        # Ensure a warning message is NOT generated when non empty uris are specified in the metadata.source.
        with self.assertLogs(logger=logger, level=logging.DEBUG) as cm:
            logger.debug("Dummy log")
            metadata_base.PrimitiveMetadata()._validate_contact_information(good_metadata.query())

        self.assertEqual(len(cm.records), 1)

    def test_validation_will_warn_on_missing_source(self):
        logger = logging.getLogger(metadata_base.__name__)

        bad_metadata = metadata_base.PrimitiveMetadata({
            'id': 'id',
            'version': '0.1.0',
            'name': "Test Primitive",
            'python_path': 'path',
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.PRINCIPAL_COMPONENT_ANALYSIS,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.FEATURE_SELECTION,
            'installation': [{
                'type': metadata_base.PrimitiveInstallationType.PIP,
                'package': 'foobar',
                'version': '0.1.0',
            }],
        })

        good_metadata = metadata_base.PrimitiveMetadata({
            'id': 'id',
            'version': '0.1.0',
            'name': "Test Primitive",
            'python_path': 'path',
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.PRINCIPAL_COMPONENT_ANALYSIS,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.FEATURE_SELECTION,
            'installation': [{
                'type': metadata_base.PrimitiveInstallationType.PIP,
                'package': 'foobar',
                'version': '0.1.0',
            }],
            'source': {
                'name': 'Test author',
                'contact': 'mailto:test@example.com',
                'uris': ['http://someplace'],
            }
        })

        # Ensure a warning message is generated for a primitive with no source
        with self.assertLogs(logger=logger, level=logging.DEBUG) as cm:
            metadata_base.PrimitiveMetadata()._validate_contact_information(bad_metadata.query())

        self.assertEqual(len(cm.records), 1)
        self.assertEqual(cm.records[0].msg, "%(python_path)s: No \"source\" field in the primitive metadata. Metadata should contain contact information and bug reporting URI.")

        # Ensure a warning message is NOT generated when source is present
        with self.assertLogs(logger=logger, level=logging.DEBUG) as cm:
            logger.debug("Dummy log")
            metadata_base.PrimitiveMetadata()._validate_contact_information(good_metadata.query())

        self.assertEqual(len(cm.records), 1)

    def test_validation_will_warn_on_missing_description(self):
        logger = logging.getLogger(metadata_base.__name__)

        bad_metadata = metadata_base.PrimitiveMetadata({
            'id': 'id',
            'version': '0.1.0',
            'name': "Test Primitive",
            'python_path': 'path',
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.PRINCIPAL_COMPONENT_ANALYSIS,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.FEATURE_SELECTION,
            'installation': [{
                'type': metadata_base.PrimitiveInstallationType.PIP,
                'package': 'foobar',
                'version': '0.1.0',
            }],
        })

        good_metadata = metadata_base.PrimitiveMetadata({
            'id': 'id',
            'version': '0.1.0',
            'name': "Test Primitive",
            'python_path': 'path',
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.PRINCIPAL_COMPONENT_ANALYSIS,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.FEATURE_SELECTION,
            'installation': [{
                'type': metadata_base.PrimitiveInstallationType.PIP,
                'package': 'foobar',
                'version': '0.1.0',
            }],
            'description': 'primitive description'
        })

        # Ensure a warning message is generated for a primitive with no description
        with self.assertLogs(logger=logger, level=logging.DEBUG) as cm:
            metadata_base.PrimitiveMetadata()._validate_description(bad_metadata.query())

        self.assertEqual(len(cm.records), 1)
        self.assertEqual(cm.records[0].msg, "%(python_path)s: Primitive is not providing a description through its docstring.")

        # Ensure a warning message is NOT generated when description is present
        with self.assertLogs(logger=logger, level=logging.DEBUG) as cm:
            logger.debug("Dummy log")
            metadata_base.PrimitiveMetadata()._validate_description(good_metadata.query())

        self.assertEqual(len(cm.records), 1)

    def test_validation_will_warn_on_empty_description(self):
        logger = logging.getLogger(metadata_base.__name__)

        bad_metadata = metadata_base.PrimitiveMetadata({
            'id': 'id',
            'version': '0.1.0',
            'name': "Test Primitive",
            'python_path': 'path',
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.PRINCIPAL_COMPONENT_ANALYSIS,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.FEATURE_SELECTION,
            'installation': [{
                'type': metadata_base.PrimitiveInstallationType.PIP,
                'package': 'foobar',
                'version': '0.1.0',
            }],
            'description': ''
        })

        good_metadata = metadata_base.PrimitiveMetadata({
            'id': 'id',
            'version': '0.1.0',
            'name': "Test Primitive",
            'python_path': 'path',
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.PRINCIPAL_COMPONENT_ANALYSIS,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.FEATURE_SELECTION,
            'installation': [{
                'type': metadata_base.PrimitiveInstallationType.PIP,
                'package': 'foobar',
                'version': '0.1.0',
            }],
            'description': 'primitive description'
        })

        # Ensure a warning message is generated for a primitive with no description
        with self.assertLogs(logger=logger, level=logging.DEBUG) as cm:
            metadata_base.PrimitiveMetadata()._validate_description(bad_metadata.query())

        self.assertEqual(len(cm.records), 1)
        self.assertEqual(cm.records[0].msg, "%(python_path)s: Primitive is not providing a description through its docstring.")

        # Ensure a warning message is NOT generated when description is present
        with self.assertLogs(logger=logger, level=logging.DEBUG) as cm:
            logger.debug("Dummy log")
            metadata_base.PrimitiveMetadata()._validate_description(good_metadata.query())

        self.assertEqual(len(cm.records), 1)

    def test_validation_will_warn_on_inherited_description(self):
        logger = logging.getLogger(metadata_base.__name__)

        bad_metadata = metadata_base.PrimitiveMetadata({
            'id': 'id',
            'version': '0.1.0',
            'name': "Test Primitive",
            'python_path': 'path',
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.PRINCIPAL_COMPONENT_ANALYSIS,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.FEATURE_SELECTION,
            'installation': [{
                'type': metadata_base.PrimitiveInstallationType.PIP,
                'package': 'foobar',
                'version': '0.1.0',
            }],
            'description': 'A base class for primitives description'
        })

        good_metadata = metadata_base.PrimitiveMetadata({
            'id': 'id',
            'version': '0.1.0',
            'name': "Test Primitive",
            'python_path': 'path',
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.PRINCIPAL_COMPONENT_ANALYSIS,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.FEATURE_SELECTION,
            'installation': [{
                'type': metadata_base.PrimitiveInstallationType.PIP,
                'package': 'foobar',
                'version': '0.1.0',
            }],
            'description': 'primitive description'
        })

        # Ensure a warning message is generated for a primitive with no description
        with self.assertLogs(logger=logger, level=logging.DEBUG) as cm:
            metadata_base.PrimitiveMetadata()._validate_description(bad_metadata.query())

        self.assertEqual(len(cm.records), 1)
        self.assertEqual(cm.records[0].msg, "%(python_path)s: Primitive is not providing a description through its docstring.")

        # Ensure a warning message is NOT generated when description is present
        with self.assertLogs(logger=logger, level=logging.DEBUG) as cm:
            logger.debug("Dummy log")
            metadata_base.PrimitiveMetadata()._validate_description(good_metadata.query())

        self.assertEqual(len(cm.records), 1)

    def test_neural_network_mixin(self):
        class MyNeuralNetworkModuleBase:
            pass

        class Params(params.Params):
            pass

        class MyNeuralNetworkModule(MyNeuralNetworkModuleBase):
            pass

        # Silence any validation warnings.
        with utils.silence():
            class TestPrimitive(
                base.NeuralNetworkModuleMixin[Inputs, Outputs, Params, Hyperparams, MyNeuralNetworkModuleBase],
                unsupervised_learning.UnsupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams],
            ):
                metadata = metadata_base.PrimitiveMetadata({
                    'id': '4164deb6-2418-4c96-9959-3d475dcf9584',
                    'version': '0.1.0',
                    'name': "Test neural network module",
                    'python_path': 'd3m.primitives.layer.super.TestPrimitive',
                    'algorithm_types': [
                        metadata_base.PrimitiveAlgorithmType.CONVOLUTIONAL_NEURAL_NETWORK_LAYER,
                    ],
                    'primitive_family': metadata_base.PrimitiveFamily.LAYER,
                })

                def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
                    raise exceptions.NotSupportedError

                def set_training_data(self, *, inputs: Inputs) -> None:
                    raise exceptions.NotSupportedError

                def fit(self, *, timeout: float = None, iterations: int = None) -> base.CallResult[None]:
                    raise exceptions.NotSupportedError

                def get_params(self) -> Params:
                    return Params()

                def set_params(self, *, params: Params) -> None:
                    pass

                def get_module(self, *, input_module: MyNeuralNetworkModuleBase) -> MyNeuralNetworkModuleBase:
                    return MyNeuralNetworkModule()


if __name__ == '__main__':
    unittest.main()
