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
temp = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.dataset_to_dataframe'))
temp.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
temp.add_output('produce')
pipeline_description.add_step(temp)

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

print('step 3')
# Step 3: extract_columns_by_semantic_types(targets)
step_3 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.extract_columns_by_semantic_types'))
step_3.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
step_3.add_output('produce')
step_3.add_hyperparameter(name='semantic_types', argument_type=ArgumentType.VALUE,
							data=['https://metadata.datadrivendiscovery.org/types/TrueTarget'])
pipeline_description.add_step(step_3)

attributes = 'steps.2.produce'
targets = 'steps.3.produce'

print('step 4')




# step_4 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.timestamp_validation'))
# step_4.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.2.produce')
# step_4.add_output('produce')
# pipeline_description.add_step(step_4)

import sys
this = sys.modules[__name__] # this is now your current namespace

setattr(this, 'step_%s' % 4, PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.feature_analysis.statistical_h_mean')))

step_4.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.2.produce')
step_4.add_output('produce')
pipeline_description.add_step(step_4)








# Step 4: processing
step_5 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.feature_analysis.statistical_maximum'))
step_5.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.4.produce')
step_5.add_output('produce')
pipeline_description.add_step(step_5)

# Step 5: algorithm`
step_6 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.detection_algorithm.pyod_ae'))
step_6.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.5.produce')
step_6.add_output('produce')
pipeline_description.add_step(step_6)

# Step 6: Predictions
step_7 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.construct_predictions'))
step_7.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.6.produce')
step_7.add_argument(name='reference', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
step_7.add_output('produce')
pipeline_description.add_step(step_7)


pipeline_description.add_output(name='output predictions', data_reference='steps.7.produce')

# Output to json
data = pipeline_description.to_json()
with open('autoencoder_pipeline.json', 'w') as f:
    f.write(data)
    print(data)


DEFAULT_PIPELINE_DIR = os.path.join('autoencoder_pipeline.json')

from axolotl.utils import pipeline as pipeline_utils
pipeline = pipeline_utils.load_pipeline(DEFAULT_PIPELINE_DIR)

pipeline_result = evaluate_pipeline(dataset, pipeline, metric)

print(pipeline_result)
print('-------------------------')
print(pipeline_result.scores.value[0])

# raise pipeline_result.error

