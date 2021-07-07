#!/bin/bash

pipelines=$(ls Pipeline)


for p in $pipelines
do
    python runner/run_pipeline_uni.py --pipeline_path $p
done

for p in $pipelines
do
    python runner/run_pipeline_multi.py --pipeline_path $p
done


