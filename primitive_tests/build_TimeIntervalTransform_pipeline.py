from d3m import index
from d3m.metadata.base import ArgumentType
from d3m.metadata.pipeline import Pipeline, PrimitiveStep

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

# # Step 1: dataframe transformation
# primitive_1 = index.get_primitive('d3m.primitives.data_transformation.SKPowerTransformer')
# primitive_1 = index.get_primitive('d3m.primitives.data_transformation.SKStandardization')
# primitive_1 = index.get_primitive('d3m.primitives.data_transformation.SKQuantileTransformer')

#Step 1: column_parser
step_1 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.column_parser'))
step_1.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
step_1.add_output('produce')
pipeline_description.add_step(step_1)

primitive_2 = index.get_primitive('d3m.primitives.tods.data_processing.time_interval_transform')
step_2 = PrimitiveStep(primitive=primitive_2)
step_2.add_hyperparameter(name="time_interval", argument_type=ArgumentType.VALUE, data = '5T')
step_2.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
step_2.add_output('produce')
pipeline_description.add_step(step_2)
#
# # Step 2: column_parser
# step_2 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.column_parser'))
# step_2.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
# step_2.add_output('produce')
# pipeline_description.add_step(step_2)
#
#
# # Step 3: extract_columns_by_semantic_types(attributes)
# step_3 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.extract_columns_by_semantic_types'))
# step_3.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.2.produce')
# step_3.add_output('produce')
# step_3.add_hyperparameter(name='semantic_types', argument_type=ArgumentType.VALUE,
#                                   data=['https://metadata.datadrivendiscovery.org/types/Attribute'])
# pipeline_description.add_step(step_3)
#
# # Step 4: extract_columns_by_semantic_types(targets)
# step_4 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.extract_columns_by_semantic_types'))
# step_4.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
# step_4.add_output('produce')
# step_4.add_hyperparameter(name='semantic_types', argument_type=ArgumentType.VALUE,
#                                   data=['https://metadata.datadrivendiscovery.org/types/TrueTarget'])
# pipeline_description.add_step(step_4)
#
# attributes = 'steps.3.produce'
# targets = 'steps.4.produce'
#
# # Step 5: imputer
# step_5 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_cleaning.imputer.SKlearn'))
# step_5.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=attributes)
# step_5.add_output('produce')
# pipeline_description.add_step(step_5)
#
# # Step 6: random_forest
# step_6 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.regression.random_forest.SKlearn'))
# step_6.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.5.produce')
# step_6.add_argument(name='outputs', argument_type=ArgumentType.CONTAINER, data_reference=targets)
# step_6.add_output('produce')
# pipeline_description.add_step(step_6)

# Final Output
pipeline_description.add_output(name='output predictions', data_reference='steps.1.produce')

# Output to YAML
yaml = pipeline_description.to_yaml()
with open('pipeline.yml', 'w') as f:
    f.write(yaml)
print(yaml)

# Or you can output json
#data = pipline_description.to_json()
