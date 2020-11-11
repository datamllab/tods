# !/bin/bash

files=$(ls primitive_tests)
for f in $files
do
	f_path="./primitive_tests/"$f
	save_path="./new_tests/"$f
	cat $f_path | sed 's/d3m.primitives.data_transformation.dataset_to_dataframe.Common/d3m.primitives.tods.data_processing.dataset_to_dataframe/g'| sed 's/d3m.primitives.data_transformation.column_parser.Common/d3m.primitives.tods.data_processing.column_parser/g' | sed 's/d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common/d3m.primitives.tods.data_processing.extract_columns_by_semantic_types/g' | sed 's/d3m.primitives.data_transformation.construct_predictions.Common/d3m.primitives.tods.data_processing.construct_predictions/g' > $save_path
done
