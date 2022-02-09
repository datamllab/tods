import argparse

from d3m import index
from d3m.metadata.base import ArgumentType
from d3m.metadata.pipeline import Pipeline, PrimitiveStep


# Creating pipeline
pipeline_description = Pipeline()
pipeline_description.add_input(name='inputs')

#Step 0: Denormalise
step_0 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.common.denormalize'))
step_0.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
step_0.add_output('produce')
pipeline_description.add_step(step_0)

#Step 1: Convert the dataset to a DataFrame
step_1 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.dataset_to_dataframe'))
step_1.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
step_1.add_output('produce')
pipeline_description.add_step(step_1)

#Step 2: Read the csvs corresponding to the paths in the Dataframe in the form of arrays
step_2 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.common.csv_reader'))
step_2.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
step_2.add_output('produce')
step_2.add_hyperparameter(name = 'use_columns', argument_type=ArgumentType.VALUE, data = [0,1])
step_2.add_hyperparameter(name = 'return_result', argument_type=ArgumentType.VALUE, data = 'replace')
pipeline_description.add_step(step_2)

#Step 3: Column Parser
step_3 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.column_parser'))
step_3.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.2.produce')
step_3.add_output('produce')
step_3.add_hyperparameter(name='parse_semantic_types', argument_type=ArgumentType.VALUE,
                                  data=['http://schema.org/Boolean','http://schema.org/Integer', 'http://schema.org/Float', 'https://metadata.datadrivendiscovery.org/types/FloatVector',])
pipeline_description.add_step(step_3)

# Step 4: extract_columns_by_semantic_types(attributes)
step_4 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.extract_columns_by_semantic_types'))
step_4.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.3.produce')
step_4.add_output('produce')
step_4.add_hyperparameter(name='semantic_types', argument_type=ArgumentType.VALUE,
							  data=['https://metadata.datadrivendiscovery.org/types/Attribute'])
pipeline_description.add_step(step_4)

# Step 5: extract_columns_by_semantic_types(targets)
step_5 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.extract_columns_by_semantic_types'))
step_5.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.3.produce')
step_5.add_output('produce')
step_5.add_hyperparameter(name='semantic_types', argument_type=ArgumentType.VALUE,
							data=['https://metadata.datadrivendiscovery.org/types/TrueTarget'])
pipeline_description.add_step(step_5)

attributes = 'steps.4.produce'
targets = 'steps.5.produce'

# Step 6: processing
step_6 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.feature_analysis.statistical_maximum'))
step_6.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=attributes)
step_6.add_output('produce')
pipeline_description.add_step(step_6)

# Step 7: algorithm
#step_7 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.detection_algorithm.pyod_ae'))
step_7 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.detection_algorithm.pyod_ocsvm'))
step_7.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.6.produce')
step_7.add_output('produce_score')
pipeline_description.add_step(step_7)

# Step 8: Predictions
#step_8 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.construct_predictions'))
step_8 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.detection_algorithm.system_wise_detection'))
step_8.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.7.produce_score')
#step_8.add_argument(name='reference', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
step_8.add_output('produce')
pipeline_description.add_step(step_8)

step_9 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.construct_predictions'))
step_9.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.8.produce')
step_9.add_argument(name='reference', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
step_9.add_output('produce')
pipeline_description.add_step(step_9)

# Final Output
pipeline_description.add_output(name='output predictions', data_reference='steps.9.produce')

# Output to json
data = pipeline_description.to_json()
with open('system_pipeline.json', 'w') as f:
    f.write(data)
    print(data)
