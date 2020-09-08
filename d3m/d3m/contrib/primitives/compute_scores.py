import inspect
import os.path
import typing

import pandas  # type: ignore

import d3m
from d3m import container, exceptions, metrics, utils as d3m_utils
from d3m.base import utils as base_utils
from d3m.metadata import base as metadata_base, hyperparams, problem
from d3m.primitive_interfaces import base, transformer

__all__ = ('ComputeScoresPrimitive',)

# Primitives needs an installation section so that digest is computed and available for the primitive.
if d3m.__version__[0].isdigit():
    installation = [{
        'type': metadata_base.PrimitiveInstallationType.PIP,
        'package': 'd3m',
        'version': d3m.__version__,
    }]
else:
    installation = [{
        'type': metadata_base.PrimitiveInstallationType.PIP,
        'package_uri': 'git+https://gitlab.com/datadrivendiscovery/d3m.git@{git_commit}#egg=d3m'.format(
            git_commit=d3m_utils.current_git_commit(os.path.dirname(__file__)),
        ),
    }]

Inputs = container.DataFrame
Outputs = container.DataFrame


class MetricsHyperparams(hyperparams.Hyperparams, set_names=False):
    metric = hyperparams.Enumeration(
        values=[metric.name for metric in problem.PerformanceMetric],
        # Default is ignored.
        # TODO: Remove default. See: https://gitlab.com/datadrivendiscovery/d3m/issues/141
        default='ACCURACY',
    )
    pos_label = hyperparams.Hyperparameter[typing.Union[str, None]](None)
    k = hyperparams.Hyperparameter[typing.Union[int, None]](None)


class AllLabelsHyperparams(hyperparams.Hyperparams, set_names=False):
    # Default is ignored.
    # TODO: Remove default. See: https://gitlab.com/datadrivendiscovery/d3m/issues/141
    column_name = hyperparams.Hyperparameter[str]('')
    labels = hyperparams.Set(
        # Default is ignored.
        # TODO: Remove default. See: https://gitlab.com/datadrivendiscovery/d3m/issues/141
        elements=hyperparams.Hyperparameter[str](''),
        default=(),
    )


class Hyperparams(hyperparams.Hyperparams):
    metrics = hyperparams.Set(
        elements=MetricsHyperparams,
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of metrics to compute.",
    )
    all_labels = hyperparams.Set(
        elements=AllLabelsHyperparams,
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="All labels available in a dataset, per target column. When provided for a target column, it overrides all labels from metadata or data for that target column.",
    )
    add_normalized_scores = hyperparams.UniformBool(
        default=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Add additional column with normalized scores?"
    )


class ComputeScoresPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    A primitive that takes a DataFrame with predictions and a scoring Dataset (test split with
    target values present), and computes scores for given metrics and outputs them as a DataFrame.

    It searches only the dataset entry point resource for target columns
    (which should be marked with ``https://metadata.datadrivendiscovery.org/types/TrueTarget``
    semantic type) in the scoring Dataset.

    Primitive does not align rows between truth DataFrame and predictions DataFrame, it
    is expected that metric code does that if necessary. Similarly, it does not align
    columns order either.

    It uses metadata to construct the truth DataFrame and renames the index column to match
    the standard names ``d3mIndex``. It encodes any float vectors as strings.

    For predictions DataFrame it expects that it is already structured correctly with correct
    column names and it leaves to metric code to validate that truth DataFrame and predictions
    DataFrame match. It does not use or expect metadata on predictions DataFrame. Predictions
    DataFrame should already have float vectors encoded as strings.
    """

    metadata: typing.ClassVar[metadata_base.PrimitiveMetadata] = metadata_base.PrimitiveMetadata(
        {
            'id': '799802fb-2e11-4ab7-9c5e-dda09eb52a70',
            'version': '0.5.0',
            'name': "Compute scores given the metrics to use",
            'python_path': 'd3m.primitives.evaluation.compute_scores.Core',
            'source': {
                'name': d3m.__author__,
                'contact': 'mailto:mitar.d3m@tnode.com',
                'uris': [
                    'https://gitlab.com/datadrivendiscovery/d3m/blob/master/d3m/contrib/primitives/compute_scores.py',
                    'https://gitlab.com/datadrivendiscovery/d3m.git',
                ],
            },
            'installation': installation,
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.ACCURACY_SCORE,
                metadata_base.PrimitiveAlgorithmType.F1_SCORE,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.EVALUATION,
        },
    )

    def produce(  # type: ignore
        self, *, inputs: Inputs, score_dataset: container.Dataset, timeout: float = None,
        iterations: int = None,
    ) -> base.CallResult[Outputs]:
        if not self.hyperparams['metrics']:
            raise ValueError("\"metrics\" hyper-parameter cannot be empty.")

        truth, all_labels = self._get_truth(score_dataset)
        predictions = self._get_predictions(inputs)

        for target_column in self.hyperparams['all_labels']:
            all_labels[target_column['column_name']] = list(target_column['labels'])

        outputs: typing.Dict[str, typing.List] = {
            'metric': [],
            'value': [],
        }

        if self.hyperparams['add_normalized_scores']:
            outputs['normalized'] = []

        for metric_configuration in self.hyperparams['metrics']:
            metric = problem.PerformanceMetric[metric_configuration['metric']]
            metric_class = metric.get_class()

            params = {}

            if 'all_labels' in inspect.signature(metric_class).parameters and all_labels:
                params['all_labels'] = all_labels

            for param_name, param_value in metric_configuration.items():
                if param_name == 'metric':
                    continue
                if param_value is None:
                    continue
                params[param_name] = param_value

            if metric.requires_confidence() and metrics.CONFIDENCE_COLUMN not in predictions.columns:
                raise exceptions.InvalidArgumentValueError(
                    f"Metric {metric.name} requires confidence column in predictions, but it is not available.",
                )
            if metric.requires_rank() and metrics.RANK_COLUMN not in predictions.columns:
                raise exceptions.InvalidArgumentValueError(
                    f"Metric {metric.name} requires rank column in predictions, but it is not available.",
                )

            score = metric_class(**params).score(truth, predictions)

            outputs['metric'].append(metric.name)
            outputs['value'].append(score)

            if self.hyperparams['add_normalized_scores']:
                outputs['normalized'].append(metric.normalize(score))

        # Dictionary key order is preserved in Python 3.6+ which makes column order as we want it.
        results = container.DataFrame(data=outputs, columns=list(outputs.keys()), generate_metadata=True)

        # Not really necessary, but it does not hurt. In theory somebody could list same metric multiple times
        # (maybe with different params), so we use "PrimaryMultiKey" here.
        results.metadata = results.metadata.add_semantic_type(
            (metadata_base.ALL_ELEMENTS, 0),
            'https://metadata.datadrivendiscovery.org/types/PrimaryMultiKey',
        )
        results.metadata = results.metadata.add_semantic_type(
            (metadata_base.ALL_ELEMENTS, 1),
            'https://metadata.datadrivendiscovery.org/types/Score',
        )
        if self.hyperparams['add_normalized_scores']:
            results.metadata = results.metadata.add_semantic_type(
                (metadata_base.ALL_ELEMENTS, 2),
                'https://metadata.datadrivendiscovery.org/types/Score',
            )

        return base.CallResult(results)

    def multi_produce(  # type: ignore
        self, *, produce_methods: typing.Sequence[str], inputs: Inputs,
        score_dataset: container.Dataset, timeout: float = None, iterations: int = None,
    ) -> base.MultiCallResult:
        return self._multi_produce(
            produce_methods=produce_methods, timeout=timeout, iterations=iterations,
            inputs=inputs, score_dataset=score_dataset,
        )

    def fit_multi_produce(  # type: ignore
        self, *, produce_methods: typing.Sequence[str], inputs: Inputs,
        score_dataset: container.Dataset, timeout: float = None, iterations: int = None
    ) -> base.MultiCallResult:
        return self._fit_multi_produce(
            produce_methods=produce_methods, timeout=timeout, iterations=iterations,
            inputs=inputs, score_dataset=score_dataset,
        )

    # TODO: Instead of extracting true targets only from the dataset entry point, first denormalize and then extract true targets.
    def _get_truth(self, score_dataset: container.Dataset) -> typing.Tuple[pandas.DataFrame, typing.Dict[str, typing.Any]]:
        """
        Extracts true targets from the Dataset's entry point, or the only tabular resource.
        It requires that there is only one primary index column, which it makes the first
        column, named ``d3mIndex``. Then true target columns follow.

        We return a regular Pandas DataFrame with column names matching those in the metadata,
        and a dict mapping target columns to all label values in those columns, if available in metadata.
        We convert all columns to strings to match what would be loaded from ``predictions.csv`` file.
        It encodes any float vectors as strings.
        """

        main_resource_id, main_resource = base_utils.get_tabular_resource(score_dataset, None, has_hyperparameter=False)

        # We first copy before modifying in-place.
        main_resource = container.DataFrame(main_resource, copy=True)
        main_resource = self._encode_columns(main_resource)

        dataframe = self._to_dataframe(main_resource)

        indices = list(score_dataset.metadata.get_index_columns(at=(main_resource_id,)))
        targets = list(score_dataset.metadata.list_columns_with_semantic_types(
            ['https://metadata.datadrivendiscovery.org/types/TrueTarget'],
            at=(main_resource_id,),
        ))

        if not indices:
            raise exceptions.InvalidArgumentValueError("No primary index column.")
        elif len(indices) > 1:
            raise exceptions.InvalidArgumentValueError("More than one primary index column.")
        if not targets:
            raise ValueError("No true target columns.")

        dataframe = dataframe.iloc[:, indices + targets]

        dataframe = dataframe.rename({dataframe.columns[0]: metrics.INDEX_COLUMN})

        if metrics.CONFIDENCE_COLUMN in dataframe.columns[1:]:
            raise ValueError("True target column cannot be named \"confidence\". It is a reserved name.")
        if metrics.RANK_COLUMN in dataframe.columns[1:]:
            raise ValueError("True target column cannot be named \"rank\". It is a reserved name.")
        if metrics.INDEX_COLUMN in dataframe.columns[1:]:
            raise ValueError("True target column cannot be named \"d3mIndex\". It is a reserved name.")

        if d3m_utils.has_duplicates(dataframe.columns):
            duplicate_names = list(dataframe.columns)
            for name in set(dataframe.columns):
                duplicate_names.remove(name)
            raise exceptions.InvalidArgumentValueError(
                "True target columns have duplicate names: {duplicate_names}".format(
                    duplicate_names=sorted(set(duplicate_names)),
                ),
            )

        all_labels = {}

        for target_column_name, main_resource_column_index in zip(dataframe.columns[1:], targets):
            try:
                column_labels = score_dataset.metadata.query_column_field(main_resource_column_index, 'all_distinct_values', at=(main_resource_id,))
            except KeyError:
                continue

            all_labels[target_column_name] = [str(label) for label in column_labels]

        return dataframe, all_labels

    def _get_predictions(self, inputs: Inputs) -> pandas.DataFrame:
        """
        It requires that predictions already have the right structure (one ``d3mIndex``
        column, at most one ``confidence`` column, at most one ``rank`` column,
        no duplicate column names).

        We return a regular Pandas DataFrame with column names matching those in the metadata.
        We convert all columns to strings to match what would be loaded from ``predictions.csv`` file.
        Predictions DataFrame should already have float vectors encoded as strings.
        """

        dataframe = self._to_dataframe(inputs)

        if metrics.INDEX_COLUMN not in dataframe.columns:
            raise exceptions.InvalidArgumentValueError("No primary index column.")

        if d3m_utils.has_duplicates(dataframe.columns):
            duplicate_names = list(dataframe.columns)
            for name in set(dataframe.columns):
                duplicate_names.remove(name)
            raise exceptions.InvalidArgumentValueError(
                "Predicted target columns have duplicate names: {duplicate_names}".format(
                    duplicate_names=sorted(set(duplicate_names)),
                ),
            )

        return dataframe

    def _to_dataframe(self, inputs: container.DataFrame) -> pandas.DataFrame:
        # We have to copy, otherwise setting "columns" modifies original DataFrame as well.
        dataframe = pandas.DataFrame(inputs, copy=True)

        column_names = []
        for column_index in range(len(inputs.columns)):
            column_names.append(inputs.metadata.query_column(column_index).get('name', inputs.columns[column_index]))

        # Make sure column names are correct.
        dataframe.columns = column_names

        # Convert all columns to string.
        return dataframe.astype(str)

    @classmethod
    def _encode_columns(cls, inputs: Outputs) -> Outputs:
        """
        Encode numpy arrays of numbers into float vectors.
        """

        outputs = inputs
        target_columns = outputs.metadata.list_columns_with_semantic_types(
            ('https://metadata.datadrivendiscovery.org/types/PredictedTarget',),
        )

        for column_index in target_columns:
            structural_type = outputs.metadata.query_column(column_index).get('structural_type', None)

            if structural_type is None:
                continue

            if not issubclass(structural_type, container.ndarray):
                continue

            new_column = []
            all_strings = True
            for value in outputs.iloc[:, column_index]:
                assert isinstance(value, container.ndarray)

                if value.ndim == 1:
                    new_column.append(','.join(str(v) for v in value))
                else:
                    all_strings = False
                    break

            if not all_strings:
                continue

            outputs_metadata = outputs.metadata
            outputs.iloc[:, column_index] = new_column
            outputs.metadata = outputs_metadata.update_column(column_index, {
                'structural_type': str,
                'dimension': metadata_base.NO_VALUE,
            })
            outputs.metadata = outputs.metadata.remove(
                (metadata_base.ALL_ELEMENTS, column_index, metadata_base.ALL_ELEMENTS),
                recursive=True,
            )

        return outputs
