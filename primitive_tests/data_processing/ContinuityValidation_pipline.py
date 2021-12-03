from d3m import index
from d3m.metadata.base import ArgumentType
from d3m.metadata.pipeline import Pipeline, PrimitiveStep

# Creating pipeline
pipeline_description = Pipeline()
pipeline_description.add_input(name='inputs')

# Step 0: dataset_to_dataframe
step_0 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.dataset_to_dataframe'))
step_0.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data='inputs.0')
step_0.add_output('produce')
pipeline_description.add_step(step_0)

# Step 1: column_parser
step_1 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.column_parser'))
step_1.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data='steps.0.produce')
step_1.add_output('produce')
pipeline_description.add_step(step_1)

# Step 3: ContinuityValidation
step_2 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.continuity_validation'))
step_2.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data='steps.1.produce')
step_2.add_output('produce')
step_2.add_hyperparameter(name = 'continuity_option', argument_type=ArgumentType.VALUE, data = 'imputation')
step_2.add_hyperparameter(name = 'interval', argument_type=ArgumentType.VALUE, data = 0.3)
# Or:
# step_2.add_hyperparameter(name = 'continuity_option', argument_type=ArgumentType.VALUE, data = 'ablation')
pipeline_description.add_step(step_2)

# Final Output
pipeline_description.add_output(name='output predictions', data_reference='steps.2.produce')

# Output to JSON
data = pipeline_description.to_json()
with open('example_pipeline.json', 'w') as f:
    f.write(data)
    print(data)

