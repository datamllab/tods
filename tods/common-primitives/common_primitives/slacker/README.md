This package contains Slacker modules code copied as they are from existing D3M datasets' solutions,
with the following changes:
* `base` module import modified to be a relative import.
* Commented out any prints.
* In `DataFrameCategoricalEncoder` made sure regular dicts are stored in `code_maps` and not `defaultdict`.
  This makes pickling possible.
* Made code use `SimpleImputer` instead of `Imputer` for compatibility with newer sklearn.
* Made default value for `missing_values` of `DenseMixedStrategyImputer` be `np.nan` because
  `SimpleImputer` does not process string `NaN` as a special value anymore.
* Updated call to `OneHotEncoder` to not use `categorical_features`.
* Replaced all calles of `as_matrix` with calls to `values`.

Some solutions contain slightly modified versions, but these files here match the most common ones.
