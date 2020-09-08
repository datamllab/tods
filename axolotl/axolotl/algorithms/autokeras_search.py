import logging
import numpy as np

import autokeras as ak
from d3m import exceptions, index, container
from d3m.metadata import base as metadata_base

from axolotl.algorithms.autokeras_integration import keras2pipeline
from axolotl.algorithms.base import PipelineSearchBase
from axolotl.utils.pipeline import PipelineResult

logger = logging.getLogger(__name__)


class AutoKerasSearch(PipelineSearchBase):

    def __init__(self, problem_description, backend,
                 max_trials=10000, directory='.', epochs=1, batch_size=32, validation_split=0.2):
        super(AutoKerasSearch, self).__init__(problem_description, backend, ranking_function=None)

        self.clf = ak.ImageClassifier(max_trials=max_trials, seed=self.random_seed, directory=directory)
        self.tuner = self.clf.tuner
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split

    def search_fit(self, input_data, time_limit=300, *, expose_values=False):
        dataframe = self.get_dataframe(input_data)
        y = self.get_y(dataframe)
        x = self.get_x(dataframe)

        self.clf.fit(x=x, y=y, epochs=self.epochs, batch_size=self.batch_size,
                     validation_split=self.validation_split)
        keras_model = self.clf.export_model()
        best_pipeline = keras2pipeline(keras_model, batch_size=self.batch_size)

        fitted_pipeline_result = self.backend.fit_pipeline(
            problem_description=self.problem_description, pipeline=best_pipeline,
            input_data=input_data, expose_outputs=expose_values
        )

        if fitted_pipeline_result.error is not None:
            logging.error('No solution founded')
            pipeline_result = PipelineResult(pipeline=best_pipeline)
            pipeline_result.error = RuntimeError("No solution found")
            return pipeline_result

        self.best_fitted_pipeline_id = fitted_pipeline_result.fitted_pipeline_id
        return fitted_pipeline_result

    def mark_columns(self, dataset):
        problem_inputs = self.problem_description['inputs']
        for problem_input in problem_inputs:
            for target in problem_input.get('targets', []):
                if target['resource_id'] not in dataset:
                    raise exceptions.NotFoundError(
                        "Error marking target column: dataset does not contain resource with resource ID '{resource_id}'.".format(
                            resource_id=target['resource_id'],
                        ),
                    )
                if not isinstance(dataset[target['resource_id']], container.DataFrame):
                    raise TypeError(
                        "Error marking target column: resource '{resource_id}' is not a DataFrame.".format(
                            resource_id=target['resource_id'],
                        ),
                    )
                if not 0 <= target['column_index'] < dataset[target['resource_id']].shape[1]:
                    raise ValueError(
                        "Error marking target column: resource '{resource_id}' does not have a column with index '{column_index}'.".format(
                            resource_id=target['resource_id'],
                            column_index=target['column_index'],
                        ),
                    )

                dataset.metadata = dataset.metadata.add_semantic_type(
                    (target['resource_id'], metadata_base.ALL_ELEMENTS, target['column_index']),
                    'https://metadata.datadrivendiscovery.org/types/Target',
                )
                dataset.metadata = dataset.metadata.add_semantic_type(
                    (target['resource_id'], metadata_base.ALL_ELEMENTS, target['column_index']),
                    'https://metadata.datadrivendiscovery.org/types/TrueTarget',
                )
                # If column is marked as a target, it cannot be attribute as well.
                # This allows one to define in problem description otherwise attribute columns as targets.
                # See: https://gitlab.com/datadrivendiscovery/d3m/issues/265
                dataset.metadata = dataset.metadata.remove_semantic_type(
                    (target['resource_id'], metadata_base.ALL_ELEMENTS, target['column_index']),
                    'https://metadata.datadrivendiscovery.org/types/Attribute',
                )
        return dataset

    def get_dataframe(self, input_data):
        # denormalize
        denormalize = index.get_primitive('d3m.primitives.data_transformation.denormalize.Common')
        hyperparams_class = denormalize.metadata.get_hyperparams()
        primitive = denormalize(hyperparams=hyperparams_class.defaults())
        dataset = primitive.produce(inputs=input_data[0]).value

        # Add Target column into dataset
        dataset = self.mark_columns(dataset)

        # dataset to dataframe
        dataset_dataframe = index.get_primitive('d3m.primitives.data_transformation.dataset_to_dataframe.Common')
        hyperparams_class = dataset_dataframe.metadata.get_hyperparams()
        primitive = dataset_dataframe(hyperparams=hyperparams_class.defaults())
        dataframe = primitive.produce(inputs=dataset).value

        return dataframe

    def get_y(self, dataframe):
        # extract targets
        get_columns_semantic = index.get_primitive(
            'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common')
        hyperparams_class = get_columns_semantic.metadata.get_hyperparams()
        primitive = get_columns_semantic(
            hyperparams=hyperparams_class.defaults().replace(
                {
                    'semantic_types': (
                        'https://metadata.datadrivendiscovery.org/types/TrueTarget',
                        'https://metadata.datadrivendiscovery.org/types/Target',
                        'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',
                        'https://metadata.datadrivendiscovery.org/types/PredictedTarget'
                    )
                }
            )
        )
        targets = primitive.produce(inputs=dataframe).value
        y = np.array(targets, dtype=np.int64)
        return y

    def get_x(self, dataframe):
        # reading images
        image_reader = index.get_primitive('d3m.primitives.data_preprocessing.image_reader.Common')
        hyperparams_class = image_reader.metadata.get_hyperparams()
        primitive = image_reader(hyperparams=hyperparams_class.defaults().replace(
            {'return_result': 'replace'})
        )
        columns_to_use = primitive._get_columns(dataframe.metadata)
        column_index = columns_to_use[0]
        temp = [
            primitive._read_filename(column_index, dataframe.metadata.query((row_index, column_index)), value)
            for row_index, value in enumerate(dataframe.iloc[:, column_index])
        ]
        x = np.array(temp, dtype=np.float64)
        return x
