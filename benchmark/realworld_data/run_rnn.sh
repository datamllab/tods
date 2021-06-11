#!/bin/bash

data="swan_sf creditcard web_attack water_quality"

for d in $data
do
	rnn_pipelines=$(ls pipelines/RNN_LSTM/$d)
	for p in $rnn_pipelines
	do
		python run_pipeline.py --pipeline_path pipelines/RNN_LSTM/$d/$p --data_path ./data/$d.csv
	done
done
