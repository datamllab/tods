#!/bin/bash
test_scripts=$(ls tods/utils/test_main)
for file in $test_scripts
do
	for f in $tested_file
	do
		echo $f
		echo $file
	done
	echo $file

	# Test pipeline building
	python tods/utils/test_main/$file > tods/utils/skinterface/tmp.txt 2>>tods/utils/skinterface/tmp.txt
	error=$(cat tmp.txt | grep 'Error' | wc -l) 
	echo "\t#Pipeline Building Errors:" $error
	if [ "$error" -gt "0" ]
	then
		cat tods/utils/skinterface/tmp.txt
		#rm tmp.txt
		break
	fi
	echo $file >> tods/utils/skinterface/tested_file.txt
done

# do
#     for f in *.py; do python "$f"; done
# done