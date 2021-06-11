#!/bin/bash

data="creditcard web_attack water_quality swan_sf"

simple_pipelines=$(ls pipelines/simple)
subseq_pipelines=$(ls pipelines/subseq)

for d in $data
do
	for p in $subseq_pipelines
	do
		python run_pipeline.py --pipeline_path pipelines/subseq/$p --data_path ./data/$d.csv
	done
	for p in $simple_pipelines
	do
		python run_pipeline.py --pipeline_path pipelines/simple/$p --data_path ./data/$d.csv
	done
done

for d in $data
do
	rnn_pipelines=$(ls pipelines/RNN_LSTM/$d)
	for p in $rnn_pipelines
	do
		python run_pipeline.py --pipeline_path pipelines/RNN_LSTM/$d/$p --data_path ./data/$d.csv
	done
done

for d in $data
do
	ae_pipelines=$(ls pipelines/AE/$d)
	for p in $ae_pipelines
	do
		tsp python run_pipeline.py --pipeline_path pipelines/AE/$d/$p --data_path ./data/$d.csv
	done
done
