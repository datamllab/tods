import pandas as pd

from axolotl.backend.simple import SimpleRunner

from tods import evaluate_pipeline,generate_dataset, generate_problem,save_fitted_pipeline,load_fitted_pipeline,produce_fitted_pipeline, fit_pipeline
from tods.searcher import BruteForceSearch,RaySearcher

import json
# Some information
#table_path = 'datasets/NAB/realTweets/labeled_Twitter_volume_GOOG.csv' # The path of the dataset
#target_index = 2 # what column is the target



if __name__ == '__main__':
    table_path = 'datasets/anomaly/raw_data/yahoo_sub_5.csv'
    search_space_path = "tods/searcher/example_search_space.json"
    target_index = 6 # what column is the target
    #table_path = 'datasets/NAB/realTweets/labeled_Twitter_volume_IBM.csv' # The path of the dataset
    time_limit = 30 # How many seconds you wanna search

    #metric = 'F1' # F1 on label 1
    metric = 'F1_MACRO' # F1 on both label 0 and 1

    # Read data and generate dataset and problem
    df = pd.read_csv(table_path)

    dataset = generate_dataset(df[0:800], target_index=target_index)

    # 09f00e92-7e02-4a1d-9ff7-d5ab875c9439
    # Dataset(id='23899d86-da17-4ec9-ac99-11296b0a1c7b', name='23899d86-da17-4ec9-ac99-11296b0a1c7b')
    # Dataset(id='36863c48-3882-4d23-8cff-c8b4454ad7a5', name='36863c48-3882-4d23-8cff-c8b4454ad7a5')
    print(dataset.metadata)
    print(type(dataset))

    # [{'selector': [], 'metadata': {'schema': 'https://metadata.datadrivendiscovery.org/schemas/v0/container.json', 'structural_type': 'd3m.container.dataset.Dataset', 'id': 'c175dfd1-1d73-44f1-a74e-29d4f9becbe7', 'name': 'c175dfd1-1d73-44f1-a74e-29d4f9becbe7', 'digest': '9a3bd414-af98-4005-bc05-2b7744b56e85', 'dimension': {'name': 'resources', 'semantic_types': ['https://metadata.datadrivendiscovery.org/types/DatasetResource'], 'length': 1}}}, {'selector': ['learningData'], 'metadata': {'structural_type': 'd3m.container.pandas.DataFrame', 'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Table', 'https://metadata.datadrivendiscovery.org/types/DatasetEntryPoint'], 'dimension': {'name': 'rows', 'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'], 'length': 1400}}}, {'selector': ['learningData', '__ALL_ELEMENTS__'], 'metadata': {'dimension': {'name': 'columns', 'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'], 'length': 8}}}, {'selector': ['learningData', '__ALL_ELEMENTS__', 0], 'metadata': {'name': 'd3mIndex', 'structural_type': 'numpy.int64', 'semantic_types': ['http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/PrimaryKey']}}, {'selector': ['learningData', '__ALL_ELEMENTS__', 1], 'metadata': {'name': 'timestamp', 'structural_type': 'str', 'semantic_types': ['http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/Attribute']}}, {'selector': ['learningData', '__ALL_ELEMENTS__', 2], 'metadata': {'name': 'value_0', 'structural_type': 'str', 'semantic_types': ['http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/Attribute']}}, {'selector': ['learningData', '__ALL_ELEMENTS__', 3], 'metadata': {'name': 'value_1', 'structural_type': 'str', 'semantic_types': ['http://schema.org/Float', 'https://metadata.datadrivendiscovery.org/types/Attribute']}}, {'selector': ['learningData', '__ALL_ELEMENTS__', 4], 'metadata': {'name': 'value_2', 'structural_type': 'str', 'semantic_types': ['http://schema.org/Float', 'https://metadata.datadrivendiscovery.org/types/Attribute']}}, {'selector': ['learningData', '__ALL_ELEMENTS__', 5], 'metadata': {'name': 'value_3', 'structural_type': 'str', 'semantic_types': ['http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/Attribute']}}, {'selector': ['learningData', '__ALL_ELEMENTS__', 6], 'metadata': {'name': 'value_4', 'structural_type': 'str', 'semantic_types': ['http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/Attribute']}}, {'selector': ['learningData', '__ALL_ELEMENTS__', 7], 'metadata': {'name': 'anomaly', 'structural_type': 'str', 'semantic_types': ['http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/SuggestedTarget']}}]
    train_problem_description = generate_problem(dataset, metric)

    
    # load the fitted pipeline based on id
    loaded = load_fitted_pipeline('12bc5643-f473-4687-b67c-5b8cc44c7eb4')
    
    print(loaded['dataset_metadata'].to_json_structure())

    # use the loaded fitted pipelines
    # result = produce_fitted_pipeline(dataset, loaded)
    result = produce_fitted_pipeline(dataset,loaded)

    print('*' * 52)
    print("Load model result: ")
    print('-' * 52)       
    print(result)
    print('*' * 52)
