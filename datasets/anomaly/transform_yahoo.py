# TODO: Wrap it as a class and connect it to GUI
# A script to transform anomaly data to d3m format
import pandas as pd
import numpy as np
import os
import json

##############################
# Some information for the dataset to be transformed
# Designed for time series data
name = 'yahoo_sub_5'
src_path = './raw_data/yahoo_sub_5.csv'
label_name = 'anomaly'
timestamp_name = 'timestamp'
value_names = ['value_{}'.format(i) for i in range(5)]
ratio = 0.9 # Ratio of training data, the rest is for testing

###############################



dst_root = './' + name
dirs = ['./', 'SCORE', 'TEST', 'TRAIN']
maps = {'./': None, 'SCORE': 'TEST', 'TEST': 'TEST', 'TRAIN': 'TRAIN'}

# Create the corresponding directories
for d in dirs:
    if maps[d] is not None:
        dataset_name = 'dataset_' + maps[d]
        problem_name = 'problem_' + maps[d]
    else:
        dataset_name = name + '_dataset'
        problem_name = name + '_problem'
    tables_dir = os.path.join(dst_root, d, dataset_name, 'tables')
    if not os.path.exists(tables_dir):
        os.makedirs(tables_dir)
    problem_dir = os.path.join(dst_root, d, problem_name)
    if not os.path.exists(problem_dir):
        os.makedirs(problem_dir)

# Process data
_df = pd.DataFrame()
df = pd.read_csv(src_path)
_df['d3mIndex'] = df.index
_df['timestamp'] = df[timestamp_name]
for value_name in value_names:
    _df[value_name] = df[value_name]
_df['ground_truth'] = df[label_name]
df = _df
cols = df.columns.tolist()

# Save all the data
df.to_csv(os.path.join(dst_root, name+'_dataset', 'tables', 'learningData.csv'), index=False)

# Save training and testing data
train_df, test_df = df[:int(df.shape[0]*ratio)], df[int(df.shape[0]*ratio):]

train_df.to_csv(os.path.join(dst_root, 'TRAIN', 'dataset_TRAIN', 'tables', 'learningData.csv'), index=False)
test_df.to_csv(os.path.join(dst_root, 'TEST', 'dataset_TEST', 'tables', 'learningData.csv'), index=False)
test_df.to_csv(os.path.join(dst_root, 'SCORE', 'dataset_TEST', 'tables', 'learningData.csv'), index=False)

# Data splits
row_0 = train_df.shape[0]
row_1 = train_df.shape[0]
row = row_0 + row_1
df = pd.DataFrame(np.array([[i for i in range(row)], ['TRAIN' for _ in range(row_0)] + ['TEST' for _ in range(row_1)], [0 for _ in range(row)], [0 for _ in range(row)]]).transpose(), columns = ['d3mIndex', 'type', 'repeat', 'fold'])

# Save data splits for all data
train_df.to_csv(os.path.join(dst_root, name+'_problem', 'dataSplits.csv'), index=False)

# Save training and testing splits
train_df, test_df = df[:row_0], df[row_0:]
train_df.to_csv(os.path.join(dst_root, 'TRAIN', 'problem_TRAIN', 'dataSplits.csv'), index=False)
test_df.to_csv(os.path.join(dst_root, 'TEST', 'problem_TEST', 'dataSplits.csv'), index=False)
test_df.to_csv(os.path.join(dst_root, 'SCORE', 'problem_TEST', 'dataSplits.csv'), index=False)


# Dataset JSON files
# Load template
with open('template/datasetDoc.json') as json_file:
    data = json.load(json_file)
columns = []
for i in range(len(cols)):
    c = {}
    c['colIndex'] = i
    c['colName'] = cols[i]
    if i == 0:
        c['colType'] = 'integer'
        c['role'] = ['index']
    elif i == 1:
        c['colType'] = 'integer'
        c['role'] = ['attribute']
    elif i == len(cols)-1:
        c['colType'] = 'integer'
        c['role'] = ['suggestedTarget']
    else:
        c['colType'] = 'real'
        c['role'] = ['attribute']
    columns.append(c)
data['dataResources'][0]['columns'] = columns
data['dataResources'][0]['columnsCount'] = len(cols)
 
data['about']['datasetID'] = name + '_dataset'
data['about']['datasetName'] = name
with open(os.path.join(dst_root, name+'_dataset', 'datasetDoc.json'), 'w') as outfile:
    json.dump(data, outfile, indent=4)

data['about']['datasetID'] = name +'_dataset_TRAIN'
data['about']['datasetName'] = "NULL"
with open(os.path.join(dst_root, 'TRAIN', 'dataset_TRAIN', 'datasetDoc.json'), 'w') as outfile:
    json.dump(data, outfile, indent=4)

data['about']['datasetID'] = name + '_dataset_TEST'
data['about']['datasetName'] = 'NULL'
with open(os.path.join(dst_root, 'TEST', 'dataset_TEST', 'datasetDoc.json'), 'w') as outfile:
    json.dump(data, outfile, indent=4)

data['about']['datasetID'] = name + '_dataset_TEST'
data['about']['datasetName'] = 'NULL'
with open(os.path.join(dst_root, 'SCORE', 'dataset_TEST', 'datasetDoc.json'), 'w') as outfile:
    json.dump(data, outfile, indent=4)

# Problem JSON files
# Load template
with open('template/problemDoc.json') as json_file:
    data = json.load(json_file)

data['about']['problemID'] = name+'_problem'
data['about']['problemName'] = name+'_problem'
data['about']['problemDescription'] = 'Anomaly detection'
data['about']['taskKeywords'] = ['classification', 'binary', 'tabular']
data['inputs']['data'][0]['datasetID'] = name + '_dataset'
data['inputs']['data'][0]['targets'][0]['colIndex'] = len(cols)-1
data['inputs']['data'][0]['targets'][0]['colName'] = cols[-1]
data['inputs']['dataSplits']['datasetViewMaps']['train'][0]['from'] = name+'_dataset'
data['inputs']['dataSplits']['datasetViewMaps']['test'][0]['from'] = name+'_dataset'
data['inputs']['dataSplits']['datasetViewMaps']['score'][0]['from'] = name+'_dataset'
data['inputs']['dataSplits']['datasetViewMaps']['train'][0]['to'] = name+'_dataset_TRAIN'
data['inputs']['dataSplits']['datasetViewMaps']['test'][0]['to'] = name+'_dataset_TEST'
data['inputs']['dataSplits']['datasetViewMaps']['score'][0]['to'] = name+'_dataset_SCORE'

with open(os.path.join(dst_root, name+'_problem', 'problemDoc.json'), 'w') as outfile:
    json.dump(data, outfile, indent=4)

with open(os.path.join(dst_root, 'TRAIN', 'problem_TRAIN', 'problemDoc.json'), 'w') as outfile:
    json.dump(data, outfile, indent=4)

with open(os.path.join(dst_root, 'TEST', 'problem_TEST', 'problemDoc.json'), 'w') as outfile:
    json.dump(data, outfile, indent=4)

with open(os.path.join(dst_root, 'SCORE', 'problem_TEST', 'problemDoc.json'), 'w') as outfile:
    json.dump(data, outfile, indent=4)

# Make an empty targets.csv
with open(os.path.join(dst_root, 'SCORE', 'targets.csv'), 'w') as outfile:
    outfile.write('')




