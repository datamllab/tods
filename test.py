import pandas as pd

from tods import schemas as schemas_utils
from tods import generate_dataset, evaluate_pipeline

import os

from d3m import index
from d3m.metadata.base import ArgumentType
from d3m.metadata.pipeline import Pipeline, PrimitiveStep
from axolotl.backend.simple import SimpleRunner
from tods import generate_dataset, generate_problem
# from tods.searcher import BruteForceSearch

from tods import generate_dataset, load_pipeline, evaluate_pipeline



table_path = '../../yahoo_sub_5.csv'
target_index = 6 # what column is the target
metric = 'F1_MACRO' # F1 on both label 0 and 1

# Read data and generate dataset
df = pd.read_csv(table_path)
dataset = generate_dataset(df, target_index)

print(dataset)

# print('here')

# # Load the default pipeline
# pipeline = schemas_utils.load_default_pipeline()

# print('here2')

# # Run the pipeline
# pipeline_result = evaluate_pipeline(dataset, pipeline, metric)

# print('here3')
# print(pipeline_result)


pipeline_description = Pipeline()
pipeline_description.add_input(name='inputs')

# Step 0: dataset_to_dataframe
step_0 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.dataset_to_dataframe'))
step_0.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
step_0.add_output('produce')
pipeline_description.add_step(step_0)

# Step 1: column_parser
step_1 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.column_parser'))
step_1.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
step_1.add_output('produce')
pipeline_description.add_step(step_1)

# Step 2: extract_columns_by_semantic_types(attributes)
step_2 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.extract_columns_by_semantic_types'))
step_2.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
step_2.add_output('produce')
step_2.add_hyperparameter(name='semantic_types', argument_type=ArgumentType.VALUE,
							  data=['https://metadata.datadrivendiscovery.org/types/Attribute'])
pipeline_description.add_step(step_2)

# print('step 3')
# # Step 3: extract_columns_by_semantic_types(targets)
# step_3 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.extract_columns_by_semantic_types'))
# step_3.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
# step_3.add_output('produce')
# step_3.add_hyperparameter(name='semantic_types', argument_type=ArgumentType.VALUE,
# 							data=['https://metadata.datadrivendiscovery.org/types/TrueTarget'])
# pipeline_description.add_step(step_3)

attributes = 'steps.2.produce'
targets = 'steps.3.produce'

print('step 4')




# step_3 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.time_interval_transform'))
# step_3.add_hyperparameter(name="time_interval", argument_type=ArgumentType.VALUE, data = 'T')
# step_3.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.2.produce')
# step_3.add_output('produce')
# pipeline_description.add_step(step_3)


# Step 3: holt smoothing
step_3 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.timeseries_processing.transformation.simple_exponential_smoothing'))
step_3.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.2.produce')
# step_3.add_hyperparameter(name="exclude_columns", argument_type=ArgumentType.VALUE, data = (2, 3))
# step_3.add_hyperparameter(name="use_semantic_types", argument_type=ArgumentType.VALUE, data = True)
step_3.add_output('produce')
pipeline_description.add_step(step_3)

#works:
# moving_average_transform
# holt_winters_exponential_smoothing
# simple_exponential_smoothing
# decomposition.time_series_seasonality_trend_decomposition
# subsequence_segmentation

# not
# holt_smoothing
# axiswise_scaler
# standard_scaler
# power_transformer
# quantile_transformer






# Step 4: processing
step_4 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.feature_analysis.statistical_h_mean'))
step_4.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.3.produce')
step_4.add_output('produce')
pipeline_description.add_step(step_4)

# Step 5: algorithm`
step_5 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.detection_algorithm.pyod_loda'))
step_5.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.4.produce')
step_5.add_output('produce')
pipeline_description.add_step(step_5)

# Step 6: Predictions
step_6 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.construct_predictions'))
step_6.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.5.produce')
step_6.add_argument(name='reference', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
step_6.add_output('produce')
pipeline_description.add_step(step_6)


pipeline_description.add_output(name='output predictions', data_reference='steps.6.produce')

# Output to json
data = pipeline_description.to_json()
with open('autoencoder_pipeline.json', 'w') as f:
    f.write(data)
    # print(data)


DEFAULT_PIPELINE_DIR = os.path.join('autoencoder_pipeline.json')

from axolotl.utils import pipeline as pipeline_utils
pipeline = pipeline_utils.load_pipeline(DEFAULT_PIPELINE_DIR)

pipeline_result = evaluate_pipeline(dataset, pipeline, metric)
print(pipeline_result)
print(pipeline_result.scores.value[0])

# raise pipeline_result.error[0]
# 
# {"id": "c9ff3096-5d52-4226-b9b9-52c81443d524", "schema": "https://metadata.datadrivendiscovery.org/schemas/v0/pipeline.json", "created": "2021-08-24T04:03:52.594989Z", "inputs": [{"name": "inputs"}], "outputs": [{"data": "steps.6.produce", "name": "output predictions"}], "steps": [{"type": "PRIMITIVE", "primitive": {"id": "c78138d9-9377-31dc-aee8-83d9df049c60", "version": "0.3.0", "python_path": "d3m.primitives.tods.data_processing.dataset_to_dataframe", "name": "Extract a DataFrame from a Dataset"}, "arguments": {"inputs": {"type": "CONTAINER", "data": "inputs.0"}}, "outputs": [{"id": "produce"}]}, {"type": "PRIMITIVE", "primitive": {"id": "81235c29-aeb9-3828-911a-1b25319b6998", "version": "0.6.0", "python_path": "d3m.primitives.tods.data_processing.column_parser", "name": "Parses strings into their types"}, "arguments": {"inputs": {"type": "CONTAINER", "data": "steps.0.produce"}}, "outputs": [{"id": "produce"}]}, {"type": "PRIMITIVE", "primitive": {"id": "a996cd89-ddf0-367f-8e7f-8c013cbc2891", "version": "0.4.0", "python_path": "d3m.primitives.tods.data_processing.extract_columns_by_semantic_types", "name": "Extracts columns by semantic type"}, "arguments": {"inputs": {"type": "CONTAINER", "data": "steps.1.produce"}}, "outputs": [{"id": "produce"}], "hyperparams": {"semantic_types": {"type": "VALUE", "data": ["https://metadata.datadrivendiscovery.org/types/Attribute"]}}}, {"type": "PRIMITIVE", "primitive": {"id": "b72dab84-5bdf-3bdd-88f4-1f959c384a63", "version": "0.0.1", "python_path": "d3m.primitives.tods.timeseries_processing.transformation.holt_smoothing", "name": "statsmodels.preprocessing.HoltSmoothing"}, "arguments": {"inputs": {"type": "CONTAINER", "data": "steps.2.produce"}}, "outputs": [{"id": "produce"}], "hyperparams": {"endog": {"type": "VALUE", "data": 2}}}, {"type": "PRIMITIVE", "primitive": {"id": "4793224a-d71b-33c3-b39c-cc849262fc5d", "version": "0.1.0", "python_path": "d3m.primitives.tods.feature_analysis.statistical_h_mean", "name": "Time Series Decompostional"}, "arguments": {"inputs": {"type": "CONTAINER", "data": "steps.3.produce"}}, "outputs": [{"id": "produce"}], "hyperparams": {"window_size": {"type": "VALUE", "data": 10}}}, {"type": "PRIMITIVE", "primitive": {"id": "0303d5ef-5d6a-3880-85ec-f519e8effc55", "version": "0.0.1", "python_path": "d3m.primitives.tods.detection_algorithm.pyod_loda", "name": "TODS.anomaly_detection_primitives.LODAPrimitive"}, "arguments": {"inputs": {"type": "CONTAINER", "data": "steps.4.produce"}}, "outputs": [{"id": "produce"}], "hyperparams": {"n_bins": {"type": "VALUE", "data": 10}}}, {"type": "PRIMITIVE", "primitive": {"id": "2530840a-07d4-3874-b7d8-9eb5e4ae2bf3", "version": "0.3.0", "python_path": "d3m.primitives.tods.data_processing.construct_predictions", "name": "Construct pipeline predictions output"}, "arguments": {"inputs": {"type": "CONTAINER", "data": "steps.5.produce"}, "reference": {"type": "CONTAINER", "data": "steps.1.produce"}}, "outputs": [{"id": "produce"}]}], "digest": "d91ab567b50e7dd8315cb625c1e35d90f4369dd72bd1547530a09c4b5da0ccd4"}