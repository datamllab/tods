#!/bin/bash

if [ ! -d "./pipelines/simple" ]; then
	mkdir -p ./pipelines/simple
fi
if [ ! -d "./pipelines/subseq" ]; then
	mkdir -p ./pipelines/subseq
fi
python pipeline_construction/pipeline_construction_simple.py 
python pipeline_construction/pipeline_construction_subseq.py 

data="swan_sf creditcard web_attack water_quality"
for d in $data
do 
	if [ ! -d "./pipelines/AE/$d" ]; then
		mkdir -p ./pipelines/AE/$d
	fi
	if [ ! -d "./pipelines/RNN_LSTM/$d" ]; then
		mkdir -p ./pipelines/RNN_LSTM/$d
	fi

	python pipeline_construction/neural/build_AE_pipeline.py "./data/"$d".csv"
	python pipeline_construction/neural/build_RNNLSTM_pipeline.py "./data/"$d".csv"
done


