#!/bin/bash

pipelines=$(ls Pipeline)


for p in $pipelines
do
    python runner/run_pipeline_uni_types.py --pipeline_path $p
done



