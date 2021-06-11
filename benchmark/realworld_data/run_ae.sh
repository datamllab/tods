#!/bin/bash

data="swan_sf creditcard web_attack water_quality"


for d in $data
do
	ae_pipelines=$(ls pipelines/AE/$d)
	for p in $ae_pipelines
	do
		tsp python run_pipeline.py --pipeline_path pipelines/AE/$d/$p --data_path ./data/$d.csv
	done
done
