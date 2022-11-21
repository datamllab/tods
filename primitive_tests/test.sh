#!/bin/bash

#modules="data_processing timeseries_processing feature_analysis detection_algorithm reinforcement"
modules="timeseries_processing"
#modules="data_processing"
#test_scripts=$(ls primitive_tests | grep -v -f tested_file.txt)

for module in $modules
do
	test_scripts=$(ls $module | grep -v -f tested_file.txt)
	#test_scripts=$(ls $module)
	for file in $test_scripts
	do
		for f in $tested_file
		do
			echo $f
		done
		echo $file

		# Test pipeline building
		#python primitive_tests/$file > tmp.txt 2>>tmp.txt
		python $module/$file > tmp.txt 2>>tmp.txt
		error=$(cat tmp.txt | grep 'Error' | wc -l) 
		echo "    #Pipeline Building Errors:" $error
		if [ "$error" -gt "0" ]
		then
			cat tmp.txt
			#rm tmp.txt
			break
		fi
		# Test on KPI dataset
		#python3 -m d3m runtime fit-produce -p pipeline.yml -r datasets/anomaly/kpi/TRAIN/problem_TRAIN/problemDoc.json -i datasets/anomaly/kpi/TRAIN/dataset_TRAIN/datasetDoc.json -t datasets/anomaly/kpi/TEST/dataset_TEST/datasetDoc.json -o results.csv -O pipeline_run.yml
		#python3 -m d3m runtime fit-produce -p pipeline.yml -r datasets/anomaly/kpi/TRAIN/problem_TRAIN/problemDoc.json -i datasets/anomaly/kpi/TRAIN/dataset_TRAIN/datasetDoc.json -t datasets/anomaly/kpi/TEST/dataset_TEST/datasetDoc.json -o results.csv 2>>tmp.txt

		# Test on Yahoo dataset
		#python3 -m d3m runtime fit-produce -p pipeline.yml -r datasets/anomaly/yahoo_sub_5/TRAIN/problem_TRAIN/problemDoc.json -i datasets/anomaly/yahoo_sub_5/TRAIN/dataset_TRAIN/datasetDoc.json -t datasets/anomaly/yahoo_sub_5/TEST/dataset_TEST/datasetDoc.json -o results.csv -O pipeline_run.yml
		python3 -m d3m runtime fit-produce -p example_pipeline.json -r ../datasets/anomaly/yahoo_sub_5/TRAIN/problem_TRAIN/problemDoc.json -i ../datasets/anomaly/yahoo_sub_5/TRAIN/dataset_TRAIN/datasetDoc.json -t ../datasets/anomaly/yahoo_sub_5/TEST/dataset_TEST/datasetDoc.json -o results.csv 2> tmp.txt
		error=$(cat tmp.txt | grep 'Error' | wc -l) 
		echo "    #Pipeline Running Errors:" $error
		if [ "$error" -gt "0" ]
		then
			cat tmp.txt
			#rm tmp.txt
			break
		fi
		echo $file >> tested_file.txt
	done
done
