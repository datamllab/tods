## v0.8.0

* Removed multi-targets support in `classification.light_gbm.Common` and fixed
  categorical attributes handling.
  [!118](https://gitlab.com/datadrivendiscovery/common-primitives/merge_requests/118)
* Unified date parsing across primitives.
  Added `raise_error` hyper-parameter to `data_preprocessing.datetime_range_filter.Common`.
  This bumped the version of the primitive.
  [!117](https://gitlab.com/datadrivendiscovery/common-primitives/merge_requests/117)
* `evaluation.kfold_time_series_split.Common` now parses the datetime column
  before sorting. `fuzzy_time_parsing` hyper-parameter was added to the primitive.
  This bumped the version of the primitive.
  [!110](https://gitlab.com/datadrivendiscovery/common-primitives/merge_requests/110)
* Added option `equal` to hyper-parameter `match_logic` of primitive
  `data_transformation.extract_columns_by_semantic_types.Common` to support set equality
  when determining columns to extract. This bumped the version of the primitive.
  [!116](https://gitlab.com/datadrivendiscovery/common-primitives/merge_requests/116)
* Fixed `data_preprocessing.one_hot_encoder.MakerCommon` to work with the
  latest core package.
* `data_cleaning.tabular_extractor.Common` has been fixed to work with the
  latest version of sklearn.
  [!113](https://gitlab.com/datadrivendiscovery/common-primitives/merge_requests/113)
* ISI side of `data_augmentation.datamart_augmentation.Common` and
  `data_augmentation.datamart_download.Common` has been updated.
  [!108](https://gitlab.com/datadrivendiscovery/common-primitives/merge_requests/108)
* Improved how pipelines and pipeline runs for all primitives are managed.
  Many more pipelines and pipeline runs were added.
* `evaluation.kfold_timeseries_split.Common` has been renamed to `evaluation.kfold_time_series_split.Common`.
* Fixed `data_preprocessing.dataset_sample.Common` on empty input.
  [!95](https://gitlab.com/datadrivendiscovery/common-primitives/merge_requests/95)
* `data_preprocessing.datetime_range_filter.Common` does not assume local timezone
  when parsing dates.
  [#115](https://gitlab.com/datadrivendiscovery/common-primitives/issues/115)
* Added `fuzzy_time_parsing` hyper-parameter to `data_transformation.column_parser.Common`.
  This bumped the version of the primitive.
* Fixed `data_transformation.column_parser.Common` to work correctly with `python-dateutil==2.8.1`.
  [#119](https://gitlab.com/datadrivendiscovery/common-primitives/issues/119).
* Refactored `data_preprocessing.one_hot_encoder.MakerCommon` to address some issues.
  [#66](https://gitlab.com/datadrivendiscovery/common-primitives/issues/66)
  [#75](https://gitlab.com/datadrivendiscovery/common-primitives/issues/75)
  [!96](https://gitlab.com/datadrivendiscovery/common-primitives/merge_requests/96)
* Added support for handling of numeric columns to `data_preprocessing.regex_filter.Common` and `data_preprocessing.term_filter.Common`.
  [!101](https://gitlab.com/datadrivendiscovery/common-primitives/merge_requests/101)
  [!104](https://gitlab.com/datadrivendiscovery/common-primitives/merge_requests/104)
* Fixed exception in `produce` method in `data_transformation.datetime_field_compose.Common` caused by using incorrect type for dataframe indexer.
  [!102](https://gitlab.com/datadrivendiscovery/common-primitives/merge_requests/102)
* Added primitives:
  * `data_transformation.grouping_field_compose.Common`

## v0.7.0

* Renamed primitives:
  * `data_transformation.add_semantic_types.DataFrameCommon` to `data_transformation.add_semantic_types.Common`
  * `data_transformation.remove_semantic_types.DataFrameCommon` to `data_transformation.remove_semantic_types.Common`
  * `data_transformation.replace_semantic_types.DataFrameCommon` to `data_transformation.replace_semantic_types.Common`
  * `operator.column_map.DataFrameCommon` to `operator.column_map.Common`
  * `regression.xgboost_gbtree.DataFrameCommon` to `regression.xgboost_gbtree.Common`
  * `classification.light_gbm.DataFrameCommon` to `classification.light_gbm.Common`
  * `classification.xgboost_gbtree.DataFrameCommon` to `classification.xgboost_gbtree.Common`
  * `classification.xgboost_dart.DataFrameCommon` to `classification.xgboost_dart.Common`
  * `classification.random_forest.DataFrameCommon` to `classification.random_forest.Common`
  * `data_transformation.extract_columns.DataFrameCommon` to `data_transformation.extract_columns.Common`
  * `data_transformation.extract_columns_by_semantic_types.DataFrameCommon` to `data_transformation.extract_columns_by_semantic_types.Common`
  * `data_transformation.extract_columns_by_structural_types.DataFrameCommon` to `data_transformation.extract_columns_by_structural_types.Common`
  * `data_transformation.cut_audio.DataFrameCommon` to `data_transformation.cut_audio.Common`
  * `data_transformation.column_parser.DataFrameCommon` to `data_transformation.column_parser.Common`
  * `data_transformation.remove_columns.DataFrameCommon` to `data_transformation.remove_columns.Common`
  * `data_transformation.remove_duplicate_columns.DataFrameCommon` to `data_transformation.remove_duplicate_columns.Common`
  * `data_transformation.horizontal_concat.DataFrameConcat` to `data_transformation.horizontal_concat.DataFrameCommon`
  * `data_transformation.construct_predictions.DataFrameCommon` to `data_transformation.construct_predictions.Common`
  * `data_transformation.datetime_field_compose.DataFrameCommon` to `data_transformation.datetime_field_compose.Common`
  * `data_preprocessing.label_encoder.DataFrameCommon` to `data_preprocessing.label_encoder.Common`
  * `data_preprocessing.label_decoder.DataFrameCommon` to `data_preprocessing.label_decoder.Common`
  * `data_preprocessing.image_reader.DataFrameCommon` to `data_preprocessing.image_reader.Common`
  * `data_preprocessing.text_reader.DataFrameCommon` to `data_preprocessing.text_reader.Common`
  * `data_preprocessing.video_reader.DataFrameCommon` to `data_preprocessing.video_reader.Common`
  * `data_preprocessing.csv_reader.DataFrameCommon` to `data_preprocessing.csv_reader.Common`
  * `data_preprocessing.audio_reader.DataFrameCommon` to `data_preprocessing.audio_reader.Common`
  * `data_preprocessing.regex_filter.DataFrameCommon` to `data_preprocessing.regex_filter.Common`
  * `data_preprocessing.term_filter.DataFrameCommon` to `data_preprocessing.term_filter.Common`
  * `data_preprocessing.numeric_range_filter.DataFrameCommon` to `data_preprocessing.numeric_range_filter.Common`
  * `data_preprocessing.datetime_range_filter.DataFrameCommon` to `data_preprocessing.datetime_range_filter.Common`

## v0.6.0

* Added `match_logic`, `negate`, and `add_index_columns` hyper-parameters
  to `data_transformation.extract_columns_by_structural_types.DataFrameCommon`
  and `data_transformation.extract_columns_by_semantic_types.DataFrameCommon`
  primitives.
* `feature_extraction.sparse_pca.Common` has been removed and is now available as part of realML.
  [!89](https://gitlab.com/datadrivendiscovery/common-primitives/merge_requests/89)
* Added new primitives:
  * `data_preprocessing.datetime_range_filter.DataFrameCommon`
  * `data_transformation.datetime_field_compose.DataFrameCommon`
  * `d3m.primitives.data_preprocessing.flatten.DataFrameCommon`
  * `data_augmentation.datamart_augmentation.Common`
  * `data_augmentation.datamart_download.Common`
  * `data_preprocessing.dataset_sample.Common`

    [#53](https://gitlab.com/datadrivendiscovery/common-primitives/issues/53)
    [!86](https://gitlab.com/datadrivendiscovery/common-primitives/merge_requests/86)
    [!87](https://gitlab.com/datadrivendiscovery/common-primitives/merge_requests/87)
    [!85](https://gitlab.com/datadrivendiscovery/common-primitives/merge_requests/85)
    [!63](https://gitlab.com/datadrivendiscovery/common-primitives/merge_requests/63)
    [!92](https://gitlab.com/datadrivendiscovery/common-primitives/merge_requests/92)
    [!93](https://gitlab.com/datadrivendiscovery/common-primitives/merge_requests/93)
    [!81](https://gitlab.com/datadrivendiscovery/common-primitives/merge_requests/81)

* Fixed `fit` method to return correct value for `operator.column_map.DataFrameCommon`,
  `operator.dataset_map.DataFrameCommon`, and `schema_discovery.profiler.Common`.
* Some not maintained primitives have been disabled. If you are using them, consider adopting them.
  * `classification.bayesian_logistic_regression.Common`
  * `regression.convolutional_neural_net.TorchCommon`
  * `operator.diagonal_mvn.Common`
  * `regression.feed_forward_neural_net.TorchCommon`
  * `data_preprocessing.image_reader.Common`
  * `clustering.k_means.Common`
  * `regression.linear_regression.Common`
  * `regression.loss.TorchCommon`
  * `feature_extraction.pca.Common`
* `data_transformation.update_semantic_types.DatasetCommon` has been removed.
  Use `data_transformation.add_semantic_types.DataFrameCommon`,
  `data_transformation.remove_semantic_types.DataFrameCommon`,
  or `data_transformation.replace_semantic_types.DataFrameCommon` together with
  `operator.dataset_map.DataFrameCommon` primitive to obtain previous functionality.
  [#83](https://gitlab.com/datadrivendiscovery/common-primitives/issues/83)
* `data_transformation.remove_columns.DatasetCommon` has been removed.
  Use `data_transformation.remove_columns.DataFrameCommon` together with
  `operator.dataset_map.DataFrameCommon` primitive to obtain previous functionality.
  [#83](https://gitlab.com/datadrivendiscovery/common-primitives/issues/83)
* Some primitives which operate on Dataset have been converted to operate
  on DataFrame and renamed. Use them together with `operator.dataset_map.DataFrameCommon`
  primitive to obtain previous functionality.
  * `data_preprocessing.regex_filter.DatasetCommon` to `data_preprocessing.regex_filter.DataFrameCommon`
  * `data_preprocessing.term_filter.DatasetCommon` to `data_preprocessing.term_filter.DataFrameCommon`
  * `data_preprocessing.numeric_range_filter.DatasetCommon` to `data_preprocessing.numeric_range_filter.DataFrameCommon`

    [#83](https://gitlab.com/datadrivendiscovery/common-primitives/issues/83)
    [!84](https://gitlab.com/datadrivendiscovery/common-primitives/merge_requests/84)

* `schema_discovery.profiler.Common` has been improved:
  * More options added to `detect_semantic_types`.
  * Added new `remove_unknown_type` hyper-parameter.

## v0.5.0

* `evaluation.compute_scores.Common` primitive has been moved to the core
  package and renamed to `evaluation.compute_scores.Core`.
* `metafeature_extraction.compute_metafeatures.Common` has been renamed to
  `metalearning.metafeature_extractor.Common`
* `evaluation.compute_scores.Common` has now a `add_normalized_scores` hyper-parameter
  to control adding also a column with normalized scores to the output, which is now
  added by default.
* `data_preprocessing.text_reader.DataFrameCommon` primitive has been fixed.
* `data_transformation.rename_duplicate_name.DataFrameCommon` primitive was
  fixed to handle all types of column names.
  [#73](https://gitlab.com/datadrivendiscovery/common-primitives/issues/73)
  [!65](https://gitlab.com/datadrivendiscovery/common-primitives/merge_requests/65)
* Added new primitives:
  * `data_cleaning.tabular_extractor.Common`
  * `data_preprocessing.one_hot_encoder.PandasCommon`
  * `schema_discovery.profiler.Common`
  * `data_transformation.ravel.DataFrameRowCommon`
  * `operator.column_map.DataFrameCommon`
  * `operator.dataset_map.DataFrameCommon`
  * `data_transformation.normalize_column_references.Common`
  * `data_transformation.normalize_graphs.Common`
  * `feature_extraction.sparse_pca.Common`
  * `evaluation.kfold_timeseries_split.Common`

    [#57](https://gitlab.com/datadrivendiscovery/common-primitives/issues/57)
    [!42](https://gitlab.com/datadrivendiscovery/common-primitives/merge_requests/42)
    [!44](https://gitlab.com/datadrivendiscovery/common-primitives/merge_requests/44)
    [!47](https://gitlab.com/datadrivendiscovery/common-primitives/merge_requests/47)
    [!71](https://gitlab.com/datadrivendiscovery/common-primitives/merge_requests/71)
    [!73](https://gitlab.com/datadrivendiscovery/common-primitives/merge_requests/73)
    [!77](https://gitlab.com/datadrivendiscovery/common-primitives/merge_requests/77)
    [!66](https://gitlab.com/datadrivendiscovery/common-primitives/merge_requests/66)
    [!67](https://gitlab.com/datadrivendiscovery/common-primitives/merge_requests/67)

* Added hyper-parameter `error_on_no_columns` to `classification.random_forest.DataFrameCommon`.
* Common primitives have been updated to latest changes in d3m core package.
* Many utility functions from `utils.py` have been moved to the d3m core package.

## v0.4.0

* Renamed `data_preprocessing.one_hot_encoder.Common` to
  `data_preprocessing.one_hot_encoder.MakerCommon` and reimplement it.
  [!54](https://gitlab.com/datadrivendiscovery/common-primitives/merge_requests/54)
* Added new primitives:
  * `classification.xgboost_gbtree.DataFrameCommon`
  * `classification.xgboost_dart.DataFrameCommon`
  * `regression.xgboost_gbtree.DataFrameCommon`
  * `classification.light_gbm.DataFrameCommon`
  * `data_transformation.rename_duplicate_name.DataFrameCommon`

    [!45](https://gitlab.com/datadrivendiscovery/common-primitives/merge_requests/45)
    [!46](https://gitlab.com/datadrivendiscovery/common-primitives/merge_requests/46)
    [!49](https://gitlab.com/datadrivendiscovery/common-primitives/merge_requests/49)

* Made sure `utils.select_columns` works also when given a tuple of columns, and not a list.
  [!58](https://gitlab.com/datadrivendiscovery/common-primitives/merge_requests/58)
* `classification.random_forest.DataFrameCommon` updated so that produced columns have
  names matching column names during fitting. Moreover, `produce_feature_importances`
  return a `DataFrame` with each column being one feature and having one row with
  importances.
  [!59](https://gitlab.com/datadrivendiscovery/common-primitives/merge_requests/59)
* `regression.feed_forward_neural_net.TorchCommon` updated to support
  selection of columns using semantic types.
  [!57](https://gitlab.com/datadrivendiscovery/common-primitives/merge_requests/57)

## v0.3.0

* Made `evaluation.redact_columns.Common` primitive more general so that it can
  redact any columns based on their semantic type and not just targets.
* Renamed primitives:
  * `datasets.Denormalize` to `data_transformation.denormalize.Common`
  * `datasets.DatasetToDataFrame` to `data_transformation.dataset_to_dataframe.Common`
  * `evaluation.ComputeScores` to `evaluation.compute_scores.Common`
  * `evaluation.RedactTargets` to `evaluation.redact_columns.Common`
  * `evaluation.KFoldDatasetSplit` to `evaluation.kfold_dataset_split.Common`
  * `evaluation.TrainScoreDatasetSplit` to `evaluation.train_score_dataset_split.Common`
  * `evaluation.NoSplitDatasetSplit` to `evaluation.no_split_dataset_split.Common`
  * `evaluation.FixedSplitDatasetSplit` to `evaluation.fixed_split_dataset_split.Commmon`
  * `classifier.RandomForest` to `classification.random_forest.DataFrameCommon`
  * `metadata.ComputeMetafeatures` to `metafeature_extraction.compute_metafeatures.Common`
  * `audio.CutAudio` to `data_transformation.cut_audio.DataFrameCommon`
  * `data.ListToNDArray` to `data_transformation.list_to_ndarray.Common`
  * `data.StackNDArrayColumn` to `data_transformation.stack_ndarray_column.Common`
  * `data.AddSemanticTypes` to `data_transformation.add_semantic_types.DataFrameCommon`
  * `data.RemoveSemanticTypes` to `data_transformation.remove_semantic_types.DataFrameCommon`
  * `data.ConstructPredictions` to `data_transformation.construct_predictions.DataFrameCommon`
  * `data.ColumnParser` to `data_transformation.column_parser.DataFrameCommon`
  * `data.CastToType` to `data_transformation.cast_to_type.Common`
  * `data.ExtractColumns` to `data_transformation.extract_columns.DataFrameCommon`
  * `data.ExtractColumnsBySemanticTypes` to `data_transformation.extract_columns_by_semantic_types.DataFrameCommon`
  * `data.ExtractColumnsByStructuralTypes` to `data_transformation.extract_columns_by_structural_types.DataFrameCommon`
  * `data.RemoveColumns` to `data_transformation.remove_columns.DataFrameCommon`
  * `data.RemoveDuplicateColumns` to `data_transformation.remove_duplicate_columns.DataFrameCommon`
  * `data.HorizontalConcat` to `data_transformation.horizontal_concat.DataFrameConcat`
  * `data.DataFrameToNDArray` to `data_transformation.dataframe_to_ndarray.Common`
  * `data.NDArrayToDataFrame` to `data_transformation.ndarray_to_dataframe.Common`
  * `data.DataFrameToList` to `data_transformation.dataframe_to_list.Common`
  * `data.ListToDataFrame` to `data_transformation.list_to_dataframe.Common`
  * `data.NDArrayToList` to `data_transformation.ndarray_to_list.Common`
  * `data.ReplaceSemanticTypes` to `data_transformation.replace_semantic_types.DataFrameCommon`
  * `data.UnseenLabelEncoder` to `data_preprocessing.label_encoder.DataFrameCommon`
  * `data.UnseenLabelDecoder` to `data_preprocessing.label_decoder.DataFrameCommon`
  * `data.ImageReader` to `data_preprocessing.image_reader.DataFrameCommon`
  * `data.TextReader` to `data_preprocessing.text_reader.DataFrameCommon`
  * `data.VideoReader` to `data_preprocessing.video_reader.DataFrameCommon`
  * `data.CSVReader` to `data_preprocessing.csv_reader.DataFrameCommon`
  * `data.AudioReader` to `data_preprocessing.audio_reader.DataFrameCommon`
  * `datasets.UpdateSemanticTypes` to `data_transformation.update_semantic_types.DatasetCommon`
  * `datasets.RemoveColumns` to `data_transformation.remove_columns.DatasetCommon`
  * `datasets.RegexFilter` to `data_preprocessing.regex_filter.DatasetCommon`
  * `datasets.TermFilter` to `data_preprocessing.term_filter.DatasetCommon`
  * `datasets.NumericRangeFilter` to `data_preprocessing.numeric_range_filter.DatasetCommon`
  * `common_primitives.BayesianLogisticRegression` to `classification.bayesian_logistic_regression.Common`
  * `common_primitives.ConvolutionalNeuralNet` to `regression.convolutional_neural_net.TorchCommon`
  * `common_primitives.DiagonalMVN` to `operator.diagonal_mvn.Common`
  * `common_primitives.FeedForwardNeuralNet` to `regression.feed_forward_neural_net.TorchCommon`
  * `common_primitives.ImageReader` to `data_preprocessing.image_reader.Common`
  * `common_primitives.KMeans` to `clustering.kmeans.Common`
  * `common_primitives.LinearRegression` to `regression.linear_regression.Common`
  * `common_primitives.Loss` to `regression.loss.TorchCommon`
  * `common_primitives.PCA` to `feature_extraction.pca.Common`
  * `common_primitives.OneHotMaker` to `data_preprocessing.one_hot_encoder.Common`
* Fixed pickling issue of `classifier.RandomFores`.
  [#47](https://gitlab.com/datadrivendiscovery/common-primitives/issues/47)
  [!48](https://gitlab.com/datadrivendiscovery/common-primitives/merge_requests/48)
* `data.ColumnParser` primitive has now additional hyper-parameter `replace_index_columns`
  which controls whether index columns are still replaced when otherwise appending returned
  parsed columns or not.
* Made `data.RemoveDuplicateColumns` fit and remember duplicate columns during training.
  [#45](https://gitlab.com/datadrivendiscovery/common-primitives/issues/45)
* Added `match_logic` hyper-parameter to the `data.ReplaceSemanticTypes` primitive
  which allows one to control how multiple specified semantic types match.
* Added new primitives:
  * `metadata.ComputeMetafeatures`
  * `datasets.RegexFilter`
  * `datasets.TermFilter`
  * `datasets.NumericRangeFilter`
  * `evaluation.NoSplitDatasetSplit`
  * `evaluation.FixedSplitDatasetSplit`
* Column parser fixed to parse columns with `http://schema.org/DateTime` semantic type.
* Simplified logic (and made it more predictable) of `combine_columns` utility function when
  using `new` `return_result` and `add_index_columns` set to true. Now if output already contains
  any index column, input index columns are not added. And if there are no index columns,
  all input index columns are added at the beginning.
* Fixed `_can_use_inputs_column` in `classifier.RandomForest`. Added check of structural type, so
  only columns with numerical structural types are processed.
* Correctly set column names in `evaluation.ComputeScores` primitive's output.
* Cast indices and columns to match predicted columns' dtypes.
  [#33](https://gitlab.com/datadrivendiscovery/common-primitives/issues/33)
* `datasets.DatasetToDataFrame` primitive does not try to generate metadata automatically
  because this is not really needed (metadata can just be copied from the dataset). This
  speeds up the primitive.
  [#34](https://gitlab.com/datadrivendiscovery/common-primitives/issues/34)
* Made it uniform that whenever we are generating lists of all column names
  we try first to get the name from the metadata and fallback to one in DataFrame.
  Instead of using a column index in the latter case.
* Made splitting primitives, `classifier.RandomForest` and `data.UnseenLabelEncoder`
  be picklable even unfitted.
* Fixed entry point for `audio.CutAudio` primitive.

## v0.2.0

* Made those primitives operate on semantic types and support different ways to return results.
* Added or updated many primitives:
  * `data.ExtractColumns`
  * `data.ExtractColumnsBySemanticTypes`
  * `data.ExtractColumnsByStructuralTypes`
  * `data.RemoveColumns`
  * `data.RemoveDuplicateColumns`
  * `data.HorizontalConcat`
  * `data.CastToType`
  * `data.ColumnParser`
  * `data.ConstructPredictions`
  * `data.DataFrameToNDArray`
  * `data.NDArrayToDataFrame`
  * `data.DataFrameToList`
  * `data.ListToDataFrame`
  * `data.NDArrayToList`
  * `data.ListToNDArray`
  * `data.StackNDArrayColumn`
  * `data.AddSemanticTypes`
  * `data.RemoveSemanticTypes`
  * `data.ReplaceSemanticTypes`
  * `data.UnseenLabelEncoder`
  * `data.UnseenLabelDecoder`
  * `data.ImageReader`
  * `data.TextReader`
  * `data.VideoReader`
  * `data.CSVReader`
  * `data.AudioReader`
  * `datasets.Denormalize`
  * `datasets.DatasetToDataFrame`
  * `datasets.UpdateSemanticTypes`
  * `datasets.RemoveColumns`
  * `evaluation.RedactTargets`
  * `evaluation.ComputeScores`
  * `evaluation.KFoldDatasetSplit`
  * `evaluation.TrainScoreDatasetSplit`
  * `audio.CutAudio`
  * `classifier.RandomForest`
* Starting list enabled primitives in the [`entry_points.ini`](./entry_points.ini) file.
* Created `devel` branch which contains primitives coded against the
  future release of the `d3m` core package (its `devel` branch).
  `master` branch of this repository is made against the latest stable
  release of the `d3m` core package.
* Dropped support for Python 2.7 and require Python 3.6.
* Renamed repository and package to `common-primitives` and `common_primitives`,
  respectively.
* Repository migrated to gitlab.com and made public.

## v0.1.1

* Made common primitives work on Python 2.7.

## v0.1.0

* Initial set of common primitives.
