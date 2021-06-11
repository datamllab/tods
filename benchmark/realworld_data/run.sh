#!/bin/bash

#data="web_attack water_quality"
data="creditcard"

simple_pipelines=$(ls pipelines/simple | grep mogaal)
subseq_pipelines=$(ls pipelines/subseq)

for d in $data
do
	for p in $subseq_pipelines
	do
		tsp python run_pipeline.py --pipeline_path pipelines/subseq/$p --data_path ./data/$d.csv
	done
	#for p in $simple_pipelines
	#do
	#	tsp python run_pipeline.py --pipeline_path pipelines/simple/$p --data_path ./data/$d.csv
	#done
done
