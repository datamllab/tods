from d3m import index
from d3m.metadata.base import ArgumentType
from d3m.metadata.pipeline import Pipeline, PrimitiveStep
from d3m.metadata import hyperparams
import copy

# -> dataset_to_dataframe -> column_parser -> extract_columns_by_semantic_types(attributes) -> imputer -> random_forest
#                                             extract_columns_by_semantic_types(targets)    ->            ^

# Creating pipeline
pipeline_description = Pipeline()
pipeline_description.add_input(name='inputs')

# Step 0: dataset_to_dataframe
primitive_0 = index.get_primitive('d3m.primitives.tods.data_processing.dataset_to_dataframe')
step_0 = PrimitiveStep(primitive=primitive_0)
step_0.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
step_0.add_output('produce')
pipeline_description.add_step(step_0)

# # Step 1: column_parser
primitive_1 = index.get_primitive('d3m.primitives.data_transformation.column_parser.Common')
step_1 = PrimitiveStep(primitive=primitive_1)
step_1.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
step_1.add_output('produce')
pipeline_description.add_step(step_1)


# Step 2: extract_columns_by_semantic_types(attributes)
step_2 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common'))
step_2.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
step_2.add_output('produce')
step_2.add_hyperparameter(name='semantic_types', argument_type=ArgumentType.VALUE, data=['https://metadata.datadrivendiscovery.org/types/Attribute'])
pipeline_description.add_step(step_2)


# Step 3: extract_columns_by_semantic_types(targets)
step_3 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common'))
step_3.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
step_3.add_output('produce')
step_3.add_hyperparameter(name='semantic_types', argument_type=ArgumentType.VALUE,
							data=['https://metadata.datadrivendiscovery.org/types/TrueTarget'])
pipeline_description.add_step(step_3)

attributes = 'steps.2.produce'
targets = 'steps.3.produce'

# Step 4: test primitive
primitive_4 = index.get_primitive('d3m.primitives.tods.detection_algorithm.pyod_loda')
step_4 = PrimitiveStep(primitive=primitive_4)
step_4.add_hyperparameter(name='contamination', argument_type=ArgumentType.VALUE, data=0.1)
step_4.add_hyperparameter(name='return_result', argument_type=ArgumentType.VALUE, data='new')
step_4.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=attributes)
step_4.add_output('produce')
pipeline_description.add_step(step_4)

# Step 5: Predictions
step_5 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.construct_predictions.Common'))
step_5.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.4.produce')
step_5.add_argument(name='reference', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
step_5.add_output('produce')
pipeline_description.add_step(step_5)

# Final Output
pipeline_description.add_output(name='output predictions', data_reference='steps.5.produce')

# Output to json
data = pipeline_description.to_json()
with open('example_pipeline.json', 'w') as f:
    f.write(data)
    print(data)

