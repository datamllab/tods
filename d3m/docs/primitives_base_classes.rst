High-level primitives base classes
==================================

High-level primitives base classes provides tools to the developers
to easily create new primitives by abstracting some unnecessary and
repetitive work.

Primitives base classes
-----------------------

``FileReaderPrimitiveBase``:  A primitive base class for reading files referenced in columns.

``DatasetSplitPrimitiveBase``: A base class for primitives which fit on a
``Dataset`` object to produce splits of that ``Dataset`` when producing.

``TabularSplitPrimitiveBase``: A primitive base class for splitting tabular datasets.


Examples
--------

Examples of primitives using these base classes can be found `in
this
repository <https://gitlab.com/datadrivendiscovery/common-primitives/-/tree/master/common_primitives>`__:

-  `DataFrameImageReaderPrimitive <https://gitlab.com/datadrivendiscovery/common-primitives/-/blob/master/common_primitives/dataframe_image_reader.py>`__
    A primitive which reads columns referencing image files.
-  `FixedSplitDatasetSplitPrimitive <https://gitlab.com/datadrivendiscovery/common-primitives/-/blob/master/common_primitives/fixed_split.py>`__
   A primitive which splits a tabular Dataset in a way that uses for the test
   (score) split a fixed list of primary index values or row indices of the main
   resource to be used. All other rows are added used for the train split.
-  `KFoldDatasetSplitPrimitive <https://gitlab.com/datadrivendiscovery/common-primitives/-/blob/master/common_primitives/kfold_split.py>`__
   A primitive which splits a tabular Dataset for k-fold cross-validation.
-  `KFoldTimeSeriesSplitPrimitive <https://gitlab.com/datadrivendiscovery/common-primitives/-/blob/master/common_primitives/kfold_split_timeseries.py>`__
   A primitive which splits a tabular time-series Dataset for k-fold cross-validation.
-  `NoSplitDatasetSplitPrimitive <https://gitlab.com/datadrivendiscovery/common-primitives/-/blob/master/common_primitives/no_split.py>`__
   A primitive which splits a tabular Dataset in a way that for all splits it
   produces the same (full) Dataset.
-  `TrainScoreDatasetSplitPrimitive <https://gitlab.com/datadrivendiscovery/common-primitives/-/blob/master/common_primitives/train_score_split.py>`__
   A primitive which splits a tabular Dataset into random train and score subsets.
