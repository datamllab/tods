from d3m import index
from d3m.metadata.base import ArgumentType, Context
from d3m.metadata.pipeline import Pipeline, PrimitiveStep

# Creating pipeline
pipeline_description = Pipeline()
pipeline_description.add_input(name='inputs')

# Step 0: dataset_to_dataframe
step_0 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.dataset_to_dataframe.Common'))
step_0.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
step_0.add_output('produce')
pipeline_description.add_step(step_0)

# Step 1: Simple profiler primitive 
step_1 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.schema_discovery.profiler.Common'))
step_1.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
step_1.add_output('produce')
pipeline_description.add_step(step_1)

# Step 2: Extract columns by structural type
step_2 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.extract_columns_by_structural_types.Common'))
step_2.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
step_2.add_output('produce')
pipeline_description.add_step(step_2)

# Step 3: column_parser
step_3 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.column_parser.Common'))
step_3.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.2.produce')
step_3.add_output('produce')
pipeline_description.add_step(step_3)

# Step 4 one hot encode
step_4 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_preprocessing.one_hot_encoder.PandasCommon'))
step_4.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.3.produce')
step_4.add_hyperparameter(name='use_columns', argument_type=ArgumentType.VALUE, data = [2,5])
step_4.add_output('produce')
pipeline_description.add_step(step_4)

# Step 5 remove text
step_5 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.remove_columns.Common'))
step_5.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.4.produce')
step_5.add_hyperparameter(name='columns', argument_type=ArgumentType.VALUE, data = [25])
step_5.add_output('produce')
pipeline_description.add_step(step_5)

# Step 6 extract attributes
step_6 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common'))
step_6.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.5.produce')
step_6.add_hyperparameter(name="semantic_types", argument_type=ArgumentType.VALUE, data=["https://metadata.datadrivendiscovery.org/types/Attribute"],)
step_6.add_output('produce')
pipeline_description.add_step(step_6)

# Step 7 extract target
step_7 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common'))
step_7.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.5.produce')
step_7.add_hyperparameter(name="semantic_types", argument_type=ArgumentType.VALUE, data=["https://metadata.datadrivendiscovery.org/types/Target"],)
step_7.add_output('produce')
pipeline_description.add_step(step_7)

# Step 8: classifier 
step_8 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.classification.random_forest.Common'))
step_8.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.6.produce')
step_8.add_argument(name='outputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.7.produce')
step_8.add_output('produce')
pipeline_description.add_step(step_8)

# Step 9: construct output
step_9 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.construct_predictions.Common'))
step_9.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.8.produce')
step_9.add_argument(name='reference', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
step_9.add_output('produce')
pipeline_description.add_step(step_9)

# Final Output
pipeline_description.add_output(name='output predictions', data_reference='steps.9.produce')

# Output json pipeline
blob = pipeline_description.to_json()
filename = blob[8:44] + '.json'
with open(filename, 'w') as outfile:
    outfile.write(blob)

