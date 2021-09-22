from tods import generate_dataset, load_pipeline, evaluate_pipeline
import pandas as pd
table_path = 'datasets/anomaly/raw_data/yahoo_sub_5.csv'
df = pd.read_csv(table_path)
dataset = generate_dataset(df, 6)
# pipeline = schemas_utils.load_default_pipeline()
pipeline = load_pipeline('example_pipeline.json')
# pipeline = load_pipeline('autoencoder_pipeline.json')

# print(pipeline)

pipeline_result = evaluate_pipeline(dataset, pipeline, 'F1_MACRO')

print(pipeline_result)
# print(pipeline_result.outputs[0]['outputs.0'])
# print(type(pipeline_result.outputs[0]['outputs.0']))

# print(pipeline_result.outputs[0]['outputs.0'].iloc[1289])
# print(pipeline_result.outputs[0]['outputs.0'].iloc[1290])
# print(pipeline_result.outputs[0]['outputs.0'].iloc[1291])