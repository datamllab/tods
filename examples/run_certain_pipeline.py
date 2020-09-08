
import os

results_dir = './'
pipeline_run_yml_dir = './'

pipeline_yml_name = './pipeline.yml' # './pipeline_yml/pipeline_10.yml'
pipline_yml_index = pipeline_yml_name[11:-4]

python_command = 'python3 -m d3m runtime fit-produce -p ' + pipeline_yml_name \
                     + ' -r ./datasets/anomaly/yahoo_sub_5/TRAIN/problem_TRAIN/problemDoc.json' \
                     + ' -i ./datasets/anomaly/yahoo_sub_5/TRAIN/dataset_TRAIN/datasetDoc.json' \
                     + ' -t ./datasets/anomaly/yahoo_sub_5/TEST/dataset_TEST/datasetDoc.json -o ' \
                     + results_dir + 'result.csv' \
                     + ' -O ' \
                     + pipeline_run_yml_dir + 'pipeline_run' + '.yml'

print(python_command)
os.system(python_command)
# 'python3 -m d3m runtime fit-produce -p pipeline.yml
# -r ../datasets/anomaly/kpi/TRAIN/problem_TRAIN/problemDoc.json
# -i ../datasets/anomaly/kpi/TRAIN/dataset_TRAIN/datasetDoc.json
# -t ../datasets/anomaly/kpi/TEST/dataset_TEST/datasetDoc.json
# -o results.csv -O pipeline_run.yml'

# python3 -m d3m runtime fit-produce -p pipeline.yml
# -r ../datasets/anomaly/yahoo_sub_5/TRAIN/problem_TRAIN/problemDoc.json
# -i ../datasets/anomaly/yahoo_sub_5/TRAIN/dataset_TRAIN/datasetDoc.json
# -t ../datasets/anomaly/yahoo_sub_5/TEST/dataset_TEST/datasetDoc.json
# -o result.csv -O pipeline_run.yml