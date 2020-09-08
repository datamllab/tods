import abc
import itertools
import typing

import numpy  # type: ignore
import pandas  # type: ignore
from sklearn import metrics, preprocessing  # type: ignore

from d3m import container, exceptions, utils
from d3m.metadata import problem

__ALL__ = ('class_map',)

INDEX_COLUMN = 'd3mIndex'
CONFIDENCE_COLUMN = 'confidence'
RANK_COLUMN = 'rank'
EMPTY_VALUES = {numpy.nan, float('NaN'), ""}

Truth = typing.TypeVar('Truth', bound=container.DataFrame)
Predictions = typing.TypeVar('Predictions', bound=container.DataFrame)
AllLabels = typing.TypeVar('AllLabels', bound=typing.Mapping[str, typing.Sequence])


class Metric(metaclass=utils.AbstractMetaclass):
    @abc.abstractmethod
    def score(self, truth: Truth, predictions: Predictions) -> typing.Any:
        raise NotImplementedError

    @classmethod
    def align(cls, truth: Truth, predictions: Predictions) -> Predictions:
        """
        Aligns columns and rows in ``predictions`` to match those in ``truth``.

        It requires that all index values in ``truth`` are present in ``predictions``
        and only those. It requires that any column name in ``truth`` is also
        present in ``predictions``. Any additional columns present in ``predictions``
        are pushed to the right.

        Parameters
        ----------
        truth:
            Truth DataFrame.
        predictions:
            Predictions DataFrame.

        Returns
        -------
        Predictions with aligned rows.
        """

        truth_columns_set = set(truth.columns)
        predictions_columns_set = set(predictions.columns)

        if len(truth_columns_set) != len(truth.columns):
            raise exceptions.InvalidArgumentValueError("Duplicate column names in predictions.")
        if len(predictions_columns_set) != len(predictions.columns):
            raise exceptions.InvalidArgumentValueError("Duplicate column names in predictions.")

        columns_diff = truth_columns_set - predictions_columns_set
        if columns_diff:
            raise exceptions.InvalidArgumentValueError(f"Not all columns which exist in truth exist in predictions: {sorted(columns_diff)}")

        if INDEX_COLUMN not in truth.columns:
            raise exceptions.InvalidArgumentValueError(f"Index column '{INDEX_COLUMN}' is missing in truth.")
        if INDEX_COLUMN not in predictions.columns:
            raise exceptions.InvalidArgumentValueError(f"Index column '{INDEX_COLUMN}' is missing in predictions.")

        extra_predictions_columns = [column for column in predictions.columns if column not in truth_columns_set]

        # Reorder columns.
        predictions = predictions.reindex(columns=list(truth.columns) + extra_predictions_columns)

        truth_index_set = set(truth.loc[:, INDEX_COLUMN])
        predictions_index_set = set(predictions.loc[:, INDEX_COLUMN])

        if truth_index_set != predictions_index_set:
            raise exceptions.InvalidArgumentValueError(f"Predictions and truth do not have the same set of index values.")

        truth_index_map: typing.Dict = {}
        last_index = None
        for i, index in enumerate(truth.loc[:, INDEX_COLUMN]):
            if index in truth_index_map:
                if last_index != index:
                    raise exceptions.InvalidArgumentValueError(f"Truth does not have all rows with same index value grouped together.")
            else:
                truth_index_map[index] = i
                last_index = index

        predictions_index_order = []
        for index in predictions.loc[:, INDEX_COLUMN]:
            predictions_index_order.append(truth_index_map[index])

        # Reorder rows.
        # TODO: How to not use a special column name?
        #       Currently it will fail if "__row_order__" already exists. We could set "allow_duplicates", but that would just hide
        #       the fact that we have a duplicated column. How can we then control over which one we really sort and which one we drop?
        predictions.insert(0, '__row_order__', predictions_index_order)
        predictions.sort_values(['__row_order__'], axis=0, inplace=True, kind='mergesort')
        predictions.drop('__row_order__', axis=1, inplace=True)
        predictions.reset_index(drop=True, inplace=True)

        return predictions

    @classmethod
    def get_target_columns(cls, dataframe: pandas.DataFrame) -> pandas.DataFrame:
        """
        Returns only target columns present in ``dataframe``.
        """

        columns = list(dataframe.columns)

        index_columns = columns.count(INDEX_COLUMN)
        if index_columns < 1:
            raise exceptions.InvalidArgumentValueError(f"Index column '{INDEX_COLUMN}' is missing in predictions.")
        elif index_columns > 1:
            raise exceptions.InvalidArgumentValueError(f"Predictions contain multiple index columns '{INDEX_COLUMN}': {index_columns}")

        dataframe = dataframe.drop(columns=[INDEX_COLUMN])

        confidence_columns = columns.count(CONFIDENCE_COLUMN)
        if confidence_columns > 1:
            raise exceptions.InvalidArgumentValueError(f"Predictions contain multiple confidence columns '{CONFIDENCE_COLUMN}': {confidence_columns}")
        elif confidence_columns:
            dataframe = dataframe.drop(columns=[CONFIDENCE_COLUMN])

        rank_columns = columns.count(RANK_COLUMN)
        if rank_columns > 1:
            raise exceptions.InvalidArgumentValueError(f"Predictions contain multiple rank columns '{RANK_COLUMN}': {rank_columns}")
        elif rank_columns:
            dataframe = dataframe.drop(columns=[RANK_COLUMN])

        if not len(dataframe.columns):
            raise exceptions.InvalidArgumentValueError(f"Predictions do not contain any target columns.")

        return dataframe

    @classmethod
    def get_index_column(cls, dataframe: pandas.DataFrame) -> pandas.DataFrame:
        """
        Returns only index column present in ``dataframe``.
        """

        columns = list(dataframe.columns)

        index_columns = columns.count(INDEX_COLUMN)
        if index_columns < 1:
            raise exceptions.InvalidArgumentValueError(f"Index column '{INDEX_COLUMN}' is missing in predictions.")
        elif index_columns > 1:
            raise exceptions.InvalidArgumentValueError(f"Predictions contain multiple index columns '{INDEX_COLUMN}': {index_columns}")

        return dataframe.loc[:, [INDEX_COLUMN]]

    @classmethod
    def get_confidence_column(cls, dataframe: pandas.DataFrame) -> pandas.DataFrame:
        """
        Returns only confidence column present in ``dataframe``.
        """

        columns = list(dataframe.columns)

        confidence_columns = columns.count(CONFIDENCE_COLUMN)
        if confidence_columns < 1:
            raise exceptions.InvalidArgumentValueError(f"Confidence column '{CONFIDENCE_COLUMN}' is missing in predictions.")
        elif confidence_columns > 1:
            raise exceptions.InvalidArgumentValueError(f"Predictions contain multiple confidence columns '{CONFIDENCE_COLUMN}': {confidence_columns}")

        return dataframe.loc[:, [CONFIDENCE_COLUMN]]

    @classmethod
    def get_rank_column(cls, dataframe: pandas.DataFrame) -> pandas.DataFrame:
        """
        Returns only rank column present in ``dataframe``.
        """

        columns = list(dataframe.columns)

        rank_columns = columns.count(RANK_COLUMN)
        if rank_columns < 1:
            raise exceptions.InvalidArgumentValueError(f"Rank column '{RANK_COLUMN}' is missing in predictions.")
        elif rank_columns > 1:
            raise exceptions.InvalidArgumentValueError(f"Predictions contain multiple rank columns '{RANK_COLUMN}': {rank_columns}")

        return dataframe.loc[:, [RANK_COLUMN]]

    @classmethod
    def vectorize_columns(cls, dataframe: pandas.DataFrame) -> pandas.DataFrame:
        """
        For every non-index column, convert all values in rows belonging to the
        same index to one row with value being a tuple of values. The order of values
        in a tuple follows the order of original rows and is preserved between columns.
        """

        columns_set = set(dataframe.columns)

        if len(columns_set) != len(dataframe.columns):
            raise exceptions.InvalidArgumentValueError("Duplicate column names.")

        if INDEX_COLUMN not in dataframe.columns:
            raise exceptions.InvalidArgumentValueError(f"Index column '{INDEX_COLUMN}' is missing.")

        columns_without_index = [column_name for column_name in dataframe.columns if column_name != INDEX_COLUMN]

        rows = {}
        for index_value in dataframe.loc[:, INDEX_COLUMN].unique():
            rows[index_value] = {
                # When we have multiple columns, some of them might not have values for all rows,
                # and there are more rows because some other column needs them. In such case
                # the column with less values should put an empty value in those extra rows
                # (generally an empty string).
                column_name: tuple(v for v in dataframe.loc[dataframe[INDEX_COLUMN] == index_value, column_name] if not cls.is_empty_value(v))
                for column_name in columns_without_index
            }

        output = pandas.DataFrame.from_dict(rows, orient='index', columns=columns_without_index)
        output.index.set_names([INDEX_COLUMN], inplace=True)
        output.reset_index(inplace=True)

        return output

    @classmethod
    def is_empty_value(cls, v: typing.Any) -> bool:
        return v in EMPTY_VALUES or (isinstance(v, (float, numpy.float64, numpy.float32)) and numpy.isnan(v))

    @classmethod
    def one_hot_encode_target(cls, series: pandas.Series, all_labels: typing.Sequence) -> pandas.DataFrame:
        """
        Returns one-hot-encoded dataframe where the columns are the labels of the target column,
        which is provided as a series of tuples, where each tuple contains all labels of a
        given sample.
        """

        mlb = preprocessing.MultiLabelBinarizer(all_labels)
        encoded = mlb.fit_transform(series)

        return encoded

    @classmethod
    def one_hot_encode_confidence(cls, series: pandas.Series, all_labels: typing.Sequence) -> pandas.DataFrame:
        """
        Returns one-hot-encoded dataframe where the columns are the labels of the confidence column,
        which is provided as a series of tuples, where each tuple contains confidence for all labels
        of a given sample, ordered in order specified by ``labels``.

        Returned dataframe has instead of 0 or 1, a confidence value itself.
        """

        encoded = series.apply(pandas.Series)
        encoded.columns = all_labels

        return encoded


class _AllAsMultiLabelBase(Metric):
    def __init__(self, all_labels: AllLabels = None) -> None:
        self.all_labels = all_labels

    def encode_targets(self, truth: Truth, predictions: Predictions) -> typing.Sequence[typing.Tuple[pandas.DataFrame, pandas.DataFrame, typing.Sequence]]:
        truth_vectorized = self.vectorize_columns(truth)
        predictions_vectorized = self.vectorize_columns(predictions)

        predictions_vectorized = self.align(truth_vectorized, predictions_vectorized)

        truth_targets = self.get_target_columns(truth_vectorized)
        predictions_targets = self.get_target_columns(predictions_vectorized)

        if len(truth_targets.columns) != len(predictions_targets.columns):
            raise exceptions.InvalidArgumentValueError(f"The number of target columns in truth ({len(truth_targets.columns)}) and predictions ({len(predictions_targets.columns)}) do not match.")

        truth_targets_columns_set = set(truth_targets.columns)

        # This holds from checks in "align".
        assert truth_targets_columns_set == set(predictions_targets.columns), (truth_targets.columns, predictions_targets.columns)

        result = []
        for column in truth_targets.columns:
            # We know that column names are unique because we check in "align".
            truth_target = truth_targets[column]
            predictions_target = predictions_targets[column]

            truth_target_values_set = set(itertools.chain.from_iterable(truth_target))
            predictions_target_values_set = set(itertools.chain.from_iterable(predictions_target))

            # If all labels were provided.
            if self.all_labels is not None and column in self.all_labels:
                all_labels_set = set(self.all_labels[column])

                extra_truth_target_values_set = truth_target_values_set - all_labels_set
                if extra_truth_target_values_set:
                    raise exceptions.InvalidArgumentValueError(f"Truth contains extra labels: {sorted(extra_truth_target_values_set)}")

                extra_predictions_target_values_set = predictions_target_values_set - all_labels_set
                if extra_predictions_target_values_set:
                    raise exceptions.InvalidArgumentValueError(f"Predictions contain extra labels: {sorted(extra_predictions_target_values_set)}")

            # Otherwise we infer all labels from available data.
            else:
                all_labels_set = truth_target_values_set | predictions_target_values_set

            all_labels = sorted(all_labels_set)

            truth_target_encoded = self.one_hot_encode_target(truth_target, all_labels)
            predictions_target_encoded = self.one_hot_encode_target(predictions_target, all_labels)

            result.append((truth_target_encoded, predictions_target_encoded, all_labels))

        return result

    def score(self, truth: Truth, predictions: Predictions) -> float:
        # We encode all as multi-label.
        encoded_targets = self.encode_targets(truth, predictions)

        if not encoded_targets:
            raise exceptions.InvalidArgumentValueError("No target column.")

        scores = []
        for truth_target_encoded, predictions_target_encoded, labels in encoded_targets:
            scores.append(self.score_one(truth_target_encoded, predictions_target_encoded, labels))

        return float(numpy.mean(scores))

    @abc.abstractmethod
    def score_one(self, truth_target_encoded: pandas.DataFrame, predictions_target_encoded: pandas.DataFrame, all_labels: typing.Sequence) -> float:
        raise NotImplementedError


class _MultiTaskBase(Metric):
    def score(self, truth: Truth, predictions: Predictions) -> float:
        predictions = self.align(truth, predictions)

        truth_targets = self.get_target_columns(truth)
        predictions_targets = self.get_target_columns(predictions)

        if len(truth_targets.columns) != len(predictions_targets.columns):
            raise exceptions.InvalidArgumentValueError(f"The number of target columns in truth ({len(truth_targets.columns)}) and predictions ({len(predictions_targets.columns)}) do not match.")

        if not len(truth_targets.columns):
            raise exceptions.InvalidArgumentValueError("No target column.")

        # This holds from checks in "align".
        assert set(truth_targets.columns) == set(predictions_targets.columns), (truth_targets.columns, predictions_targets.columns)

        scores = []
        for column in truth_targets.columns:
            # We know that column names are unique because we check in "align".
            truth_target = truth_targets[column]
            predictions_target = predictions_targets[column]

            scores.append(self.score_one(truth_target, predictions_target))

        return float(numpy.mean(scores))

    @abc.abstractmethod
    def score_one(self, truth_target: pandas.Series, predictions_target: pandas.Series) -> float:
        raise NotImplementedError


class AccuracyMetric(_AllAsMultiLabelBase):
    """
    Supports binary, multi-class, multi-label, and multi-task predictions.
    """

    def score_one(self, truth_target_encoded: pandas.DataFrame, predictions_target_encoded: pandas.DataFrame, all_labels: typing.Sequence) -> float:
        return metrics.accuracy_score(truth_target_encoded, predictions_target_encoded)


class PrecisionMetric(_MultiTaskBase):
    """
    Supports binary and multi-task predictions.
    """

    def __init__(self, pos_label: str) -> None:
        self.pos_label = pos_label

    def score_one(self, truth_target: pandas.Series, predictions_target: pandas.Series) -> float:
        # We do not have to pass labels because we are using binary average.
        return metrics.precision_score(truth_target, predictions_target, pos_label=self.pos_label, average='binary')


class RecallMetric(_MultiTaskBase):
    """
    Supports binary and multi-task predictions.
    """

    def __init__(self, pos_label: str) -> None:
        self.pos_label = pos_label

    def score_one(self, truth_target: pandas.Series, predictions_target: pandas.Series) -> float:
        # We do not have to pass labels because we are using binary average.
        return metrics.recall_score(truth_target, predictions_target, pos_label=self.pos_label, average='binary')


class F1Metric(_MultiTaskBase):
    """
    Supports binary and multi-task predictions.
    """

    def __init__(self, pos_label: str) -> None:
        self.pos_label = pos_label

    def score_one(self, truth_target: pandas.Series, predictions_target: pandas.Series) -> float:
        # We do not have to pass labels because we are using binary average.
        return metrics.f1_score(truth_target, predictions_target, pos_label=self.pos_label, average='binary')


class F1MicroMetric(_AllAsMultiLabelBase):
    """
    Supports multi-class, multi-label, and multi-task predictions.
    """

    def score_one(self, truth_target_encoded: pandas.DataFrame, predictions_target_encoded: pandas.DataFrame, all_labels: typing.Sequence) -> float:
        # We use multi-label F1 score to compute for multi-class target as well.
        # We want to use all labels, so we do not pass labels on.
        return metrics.f1_score(truth_target_encoded, predictions_target_encoded, average='micro')


class F1MacroMetric(_AllAsMultiLabelBase):
    """
    Supports multi-class, multi-label, and multi-task predictions.
    """

    def score_one(self, truth_target_encoded: pandas.DataFrame, predictions_target_encoded: pandas.DataFrame, all_labels: typing.Sequence) -> float:
        # We use multi-label F1 score to compute for multi-class target as well.
        # We want to use all labels, so we do not pass labels on.
        return metrics.f1_score(truth_target_encoded, predictions_target_encoded, average='macro')


class MeanSquareErrorMetric(Metric):
    """
    Supports univariate and multivariate.
    """

    def score(self, truth: Truth, predictions: Predictions) -> float:
        predictions = self.align(truth, predictions)

        truth_targets = self.get_target_columns(truth)
        predictions_targets = self.get_target_columns(predictions)

        return metrics.mean_squared_error(truth_targets, predictions_targets, multioutput='uniform_average')


class RootMeanSquareErrorMetric(Metric):
    """
    Supports univariate and multivariate.
    """

    def score(self, truth: Truth, predictions: Predictions) -> float:
        predictions = self.align(truth, predictions)

        truth_targets = self.get_target_columns(truth)
        predictions_targets = self.get_target_columns(predictions)

        mean_squared_error = metrics.mean_squared_error(truth_targets, predictions_targets, multioutput='raw_values')

        return float(numpy.mean(numpy.sqrt(mean_squared_error)))


class MeanAbsoluteErrorMetric(Metric):
    """
    Supports univariate and multivariate.
    """

    def score(self, truth: Truth, predictions: Predictions) -> float:
        predictions = self.align(truth, predictions)

        truth_targets = self.get_target_columns(truth)
        predictions_targets = self.get_target_columns(predictions)

        return metrics.mean_absolute_error(truth_targets, predictions_targets, multioutput='uniform_average')


class RSquaredMetric(Metric):
    """
    Supports univariate and multivariate.
    """

    def score(self, truth: Truth, predictions: Predictions) -> float:
        predictions = self.align(truth, predictions)

        truth_targets = self.get_target_columns(truth)
        predictions_targets = self.get_target_columns(predictions)

        return metrics.r2_score(truth_targets, predictions_targets, multioutput='uniform_average')


class NormalizeMutualInformationMetric(Metric):
    def score(self, truth: Truth, predictions: Predictions) -> float:
        predictions = self.align(truth, predictions)

        truth_targets = self.get_target_columns(truth)
        predictions_targets = self.get_target_columns(predictions)

        if len(truth_targets.columns) != len(predictions_targets.columns):
            raise exceptions.InvalidArgumentValueError(f"The number of target columns in truth ({len(truth_targets.columns)}) and predictions ({len(predictions_targets.columns)}) do not match.")

        if len(truth_targets.columns) != 1:
            raise exceptions.InvalidArgumentValueError("Only one target column is supported.")

        return metrics.normalized_mutual_info_score(truth_targets.iloc[:, 0].ravel(), predictions_targets.iloc[:, 0].ravel(), average_method='geometric')


class JaccardSimilarityScoreMetric(_MultiTaskBase):
    """
    Supports binary and multi-task predictions.
    """

    def __init__(self, pos_label: str) -> None:
        self.pos_label = pos_label

    def score_one(self, truth_target: pandas.Series, predictions_target: pandas.Series) -> float:
        # We do not have to pass labels because we are using binary average.
        return metrics.jaccard_score(truth_target, predictions_target, pos_label=self.pos_label, average='binary')


class PrecisionAtTopKMetric(Metric):
    def __init__(self, k: int) -> None:
        self.k = k

    def score(self, truth: Truth, predictions: Predictions) -> float:
        predictions = self.align(truth, predictions)

        truth_targets = self.get_target_columns(truth)
        predictions_targets = self.get_target_columns(predictions)

        if len(truth_targets.columns) != len(predictions_targets.columns):
            raise exceptions.InvalidArgumentValueError(f"The number of target columns in truth ({len(truth_targets.columns)}) and predictions ({len(predictions_targets.columns)}) do not match.")

        if len(truth_targets.columns) != 1:
            raise exceptions.InvalidArgumentValueError("Only one target column is supported.")

        truth_targets = truth_targets.values.ravel().astype(int)
        predictions_targets = predictions_targets.values.ravel().astype(int)

        truth_targets = numpy.argsort(truth_targets)[::-1]
        predictions_targets = numpy.argsort(predictions_targets)[::-1]

        truth_targets = truth_targets[0:self.k]
        predictions_targets = predictions_targets[0:self.k]

        return numpy.float(len(numpy.intersect1d(truth_targets, predictions_targets))) / self.k


class ObjectDetectionAveragePrecisionMetric(Metric):
    def _convert_bounding_polygon_to_box_coords(self, bounding_polygon: typing.List) -> typing.List:
        # box_coords = [x_min, y_min, x_max, y_max]
        if len(bounding_polygon) != 8:
            raise exceptions.NotSupportedError("Polygon must contain eight vertices for this metric.")

        if bounding_polygon[0] != bounding_polygon[2] or bounding_polygon[4] != bounding_polygon[6]:
            raise exceptions.NotSupportedError("X coordinates in bounding box do not match.")

        if bounding_polygon[1] != bounding_polygon[7] or bounding_polygon[3] != bounding_polygon[5]:
            raise exceptions.NotSupportedError("Y coordinates in bounding box do not match.")

        box_coords = [bounding_polygon[0], bounding_polygon[1],
                      bounding_polygon[4], bounding_polygon[5]]
        return box_coords

    def _group_gt_boxes_by_image_name(self, gt_boxes: typing.List) -> typing.Dict:
        gt_dict: typing.Dict = {}

        for box in gt_boxes:
            image_name = box[0]
            bounding_polygon = box[1:]
            bbox = self._convert_bounding_polygon_to_box_coords(bounding_polygon)

            if image_name not in gt_dict.keys():
                gt_dict[image_name] = []

            gt_dict[image_name].append({'bbox': bbox})

        return gt_dict

    def _voc_ap(self, rec: numpy.ndarray, prec: numpy.ndarray) -> float:
        # First append sentinel values at the end.
        mrec = numpy.concatenate(([0.], rec, [1.]))
        mpre = numpy.concatenate(([0.], prec, [0.]))

        # Compute the precision envelope.
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = numpy.maximum(mpre[i - 1], mpre[i])

        # To calculate area under PR curve, look for points
        # where X axis (recall) changes value.
        i = numpy.where(mrec[1:] != mrec[:-1])[0]

        # And sum (\Delta recall) * prec.
        ap = numpy.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

        return float(ap)

    def _object_detection_average_precision(self, y_true: typing.List, y_pred: typing.List) -> float:
        """
        This function takes a list of ground truth bounding polygons (rectangles in this case)
        and a list of detected bounding polygons (also rectangles) for a given class and
        computes the average precision of the detections with respect to the ground truth polygons.
        Parameters:
        -----------
        y_true: list
         List of ground truth polygons. Each polygon is represented as a list of
         vertices, starting in the upper-left corner going counter-clockwise.
         Since in this case, the polygons are rectangles, they will have the
         following format:
            [image_name, x_min, y_min, x_min, y_max, x_max, y_max, x_max, y_min].
        y_pred: list
         List of bounding box polygons with their corresponding confidence scores. Each
         polygon is represented as a list of vertices, starting in the upper-left corner
         going counter-clockwise. Since in this case, the polygons are rectangles, they
         will have the following format:
            [image_name, x_min, y_min, x_min, y_max, x_max, y_max, x_max, y_min, confidence_score].
        Returns:
        --------
        ap: float
         Average precision between detected polygons (rectangles) and the ground truth polylgons (rectangles).
         (it is also the area under the precision-recall curve).
        Example 1:
        >> predictions_list_1 = [['img_00001.png', 110, 110, 110, 210, 210, 210, 210, 110, 0.6],
                                 ['img_00002.png', 5, 10, 5, 20, 20, 20, 20, 10, 0.9],
                                 ['img_00002.png', 120, 130, 120, 200, 200, 200, 200, 130, 0.6]]
        >> ground_truth_list_1 = [['img_00001.png', 100, 100, 100, 200, 200, 200, 200, 100],
                                  ['img_00002.png', 10, 10, 10, 20, 20, 20, 20, 10],
                                  ['img_00002.png', 70, 80, 70, 150, 140, 150, 140, 80]]
        >> ap_1 = object_detection_average_precision(ground_truth_list_1, predictions_list_1)
        >> print(ap_1)
        0.667
        Example 2:
        >> predictions_list_2 = [['img_00285.png', 330, 463, 330, 505, 387, 505, 387, 463, 0.0739],
                                 ['img_00285.png', 420, 433, 420, 498, 451, 498, 451, 433, 0.0910],
                                 ['img_00285.png', 328, 465, 328, 540, 403, 540, 403, 465, 0.1008],
                                 ['img_00285.png', 480, 477, 480, 522, 508, 522, 508, 477, 0.1012],
                                 ['img_00285.png', 357, 460, 357, 537, 417, 537, 417, 460, 0.1058],
                                 ['img_00285.png', 356, 456, 356, 521, 391, 521, 391, 456, 0.0843],
                                 ['img_00225.png', 345, 460, 345, 547, 415, 547, 415, 460, 0.0539],
                                 ['img_00225.png', 381, 362, 381, 513, 455, 513, 455, 362, 0.0542],
                                 ['img_00225.png', 382, 366, 382, 422, 416, 422, 416, 366, 0.0559],
                                 ['img_00225.png', 730, 463, 730, 583, 763, 583, 763, 463, 0.0588]]
        >> ground_truth_list_2 = [['img_00285.png', 480, 457, 480, 529, 515, 529, 515, 457],
                                  ['img_00285.png', 480, 457, 480, 529, 515, 529, 515, 457],
                                  ['img_00225.png', 522, 540, 522, 660, 576, 660, 576, 540],
                                  ['img_00225.png', 739, 460, 739, 545, 768, 545, 768, 460]]
        >> ap_2 = object_detection_average_precision(ground_truth_list_2, predictions_list_2)
        >> print(ap_2)
        0.125
        Example 3:
        >> predictions_list_3 = [['img_00001.png', 110, 110, 110, 210, 210, 210, 210, 110, 0.6],
                                 ['img_00002.png', 120, 130, 120, 200, 200, 200, 200, 130, 0.6],
                                 ['img_00002.png', 5, 8, 5, 16, 15, 16, 15, 8, 0.9],
                                 ['img_00002.png', 11, 12, 11, 18, 21, 18, 21, 12, 0.9]]
        >> ground_truth_list_3 = [['img_00001.png', 100, 100, 100, 200, 200, 200, 200, 100],
                                  ['img_00002.png', 10, 10, 10, 20, 20, 20, 20, 10],
                                  ['img_00002.png', 70, 80, 70, 150, 140, 150, 140, 80]]
        >> ap_3 = object_detection_average_precision(ground_truth_list_3, predictions_list_3)
        >> print(ap_3)
        0.444
        Example 4:
        (Same as example 3 except the last two box predictions in img_00002.png are switched)
        >> predictions_list_4 = [['img_00001.png', 110, 110, 110, 210, 210, 210, 210, 110, 0.6],
                                 ['img_00002.png', 120, 130, 120, 200, 200, 200, 200, 130, 0.6],
                                 ['img_00002.png', 11, 12, 11, 18, 21, 18, 21, 12, 0.9],
                                 ['img_00002.png', 5, 8, 5, 16, 15, 16, 15, 8, 0.9]]
        >> ground_truth_list_4 = [['img_00001.png', 100, 100, 100, 200, 200, 200, 200, 100],
                                  ['img_00002.png', 10, 10, 10, 20, 20, 20, 20, 10],
                                  ['img_00002.png', 70, 80, 70, 150, 140, 150, 140, 80]]
        >> ap_4 = object_detection_average_precision(ground_truth_list_4, predictions_list_4)
        >> print(ap_4)
        0.444
        """

        ovthresh = 0.5

        # y_true = typing.cast(Truth, unvectorize(y_true))
        # y_pred = typing.cast(Predictions, unvectorize(y_pred))

        # Load ground truth.
        gt_dict = self._group_gt_boxes_by_image_name(y_true)

        # Extract gt objects for this class.
        recs = {}
        npos = 0

        imagenames = sorted(gt_dict.keys())
        for imagename in imagenames:
            Rlist = [obj for obj in gt_dict[imagename]]
            bbox = numpy.array([x['bbox'] for x in Rlist])
            det = [False] * len(Rlist)
            npos = npos + len(Rlist)
            recs[imagename] = {'bbox': bbox, 'det': det}

        # Load detections.
        det_length = len(y_pred[0])

        # Check that all boxes are the same size.
        for det in y_pred:
            assert len(det) == det_length, 'Not all boxes have the same dimensions.'

        image_ids = [x[0] for x in y_pred]
        BP = numpy.array([[float(z) for z in x[1:-1]] for x in y_pred])
        BB = numpy.array([self._convert_bounding_polygon_to_box_coords(x) for x in BP])

        confidence = numpy.array([float(x[-1]) for x in y_pred])
        boxes_w_confidences_list = numpy.hstack((BB, -1 * confidence[:, None]))
        boxes_w_confidences = numpy.empty(
            (boxes_w_confidences_list.shape[0],),
            dtype=[
                ('x_min', float), ('y_min', float),
                ('x_max', float), ('y_max', float),
                ('confidence', float),
            ],
        )
        boxes_w_confidences[:] = [tuple(i) for i in boxes_w_confidences_list]

        # Sort by confidence.
        sorted_ind = numpy.argsort(
            boxes_w_confidences, kind='mergesort',
            order=('confidence', 'x_min', 'y_min', 'x_max', 'y_max'))
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # Go down y_pred and mark TPs and FPs.
        nd = len(image_ids)
        tp = numpy.zeros(nd)
        fp = numpy.zeros(nd)
        for d in range(nd):
            R = recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -numpy.inf
            BBGT = R['bbox'].astype(float)

            if BBGT.size > 0:
                # Compute overlaps.
                # Intersection.
                ixmin = numpy.maximum(BBGT[:, 0], bb[0])
                iymin = numpy.maximum(BBGT[:, 1], bb[1])
                ixmax = numpy.minimum(BBGT[:, 2], bb[2])
                iymax = numpy.minimum(BBGT[:, 3], bb[3])
                iw = numpy.maximum(ixmax - ixmin + 1., 0.)
                ih = numpy.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                # Union.
                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                       (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                       (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

                overlaps = inters / uni
                ovmax = numpy.max(overlaps)
                jmax = numpy.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
            else:
                fp[d] = 1.

        # Compute precision recall.
        fp = numpy.cumsum(fp)
        tp = numpy.cumsum(tp)
        rec = tp / float(npos)
        # Avoid divide by zero in case the first detection matches a difficult ground truth.
        prec = tp / numpy.maximum(tp + fp, numpy.finfo(numpy.float64).eps)
        ap = self._voc_ap(rec, prec)

        return ap

    def score(self, truth: Truth, predictions: Predictions) -> float:
        predictions = self.align(truth, predictions)

        truth_index = self.get_index_column(truth)
        truth_targets = self.get_target_columns(truth)

        if len(truth_targets.columns) != 1:
            raise NotImplementedError("Support for multiple target columns is not yet implemented.")

        truth_list = []
        for i, (index, target) in enumerate(pandas.concat([truth_index, truth_targets], axis=1).itertuples(index=False, name=None)):
            truth_list.append([index] + [float(v) for v in target.split(',')])

        predictions_index = self.get_index_column(predictions)
        predictions_targets = self.get_target_columns(predictions)
        predictions_confidence = self.get_confidence_column(predictions)

        if len(predictions_targets.columns) != 1:
            raise NotImplementedError("Support for multiple target columns is not yet implemented.")

        predictions_list = []
        for i, (index, target, confidence) in enumerate(pandas.concat([predictions_index, predictions_targets, predictions_confidence], axis=1).itertuples(index=False, name=None)):
            predictions_list.append([index] + [float(v) for v in target.split(',')] + [float(confidence)])

        return self._object_detection_average_precision(truth_list, predictions_list)


class HammingLossMetric(_AllAsMultiLabelBase):
    """
    Hamming loss gives the percentage of wrong labels to the total number of labels.
    Lower the hamming loss, better is the performance of the method used.

    Supports multi-label and multi-task predictions.
    """

    def score_one(self, truth_target_encoded: pandas.DataFrame, predictions_target_encoded: pandas.DataFrame, all_labels: typing.Sequence) -> float:
        # We do not have to pass labels because they are not needed and passing them is deprecated.
        return metrics.hamming_loss(truth_target_encoded, predictions_target_encoded)


class _RocAucBase(Metric):
    def __init__(self, all_labels: AllLabels = None) -> None:
        self.all_labels = all_labels

    def encode_confidence(self, truth: Truth, predictions: Predictions) -> typing.Tuple[pandas.DataFrame, pandas.DataFrame]:
        truth_vectorized = self.vectorize_columns(truth)
        predictions_vectorized = self.vectorize_columns(predictions)

        predictions_vectorized = self.align(truth_vectorized, predictions_vectorized)

        truth_targets = self.get_target_columns(truth_vectorized)
        predictions_targets = self.get_target_columns(predictions_vectorized)
        predictions_confidence = self.get_confidence_column(predictions_vectorized).iloc[:, 0]

        if len(truth_targets.columns) != 1:
            raise exceptions.InvalidArgumentValueError(f"Invalid number of target columns in truth: {len(truth_targets.columns)}")
        if len(predictions_targets.columns) != 1:
            raise exceptions.InvalidArgumentValueError(f"Invalid number of target columns in predictions: {len(predictions_targets.columns)}")

        truth_targets_columns_set = set(truth_targets.columns)

        # This holds from checks in "align".
        assert truth_targets_columns_set == set(predictions_targets.columns), (truth_targets.columns, predictions_targets.columns)

        target_column_name = truth_targets.columns[0]
        truth_target = truth_targets.iloc[:, 0]
        predictions_target = predictions_targets.iloc[:, 0]

        truth_target_values_set = set(itertools.chain.from_iterable(truth_target))
        predictions_target_values_set = set(itertools.chain.from_iterable(predictions_target))

        # If all labels were provided.
        if self.all_labels is not None and target_column_name in self.all_labels:
            all_labels_set = set(self.all_labels[target_column_name])

            extra_truth_target_values_set = truth_target_values_set - all_labels_set
            if extra_truth_target_values_set:
                raise exceptions.InvalidArgumentValueError(f"Truth contains extra labels: {sorted(extra_truth_target_values_set)}")

            extra_predictions_target_values_set = predictions_target_values_set - all_labels_set
            if extra_predictions_target_values_set:
                raise exceptions.InvalidArgumentValueError(f"Predictions contain extra labels: {sorted(extra_predictions_target_values_set)}")

        # Otherwise we infer labels from available data.
        else:
            all_labels_set = truth_target_values_set | predictions_target_values_set

        all_labels = sorted(all_labels_set)

        truth_target_encoded = self.one_hot_encode_target(truth_target, all_labels)

        for i, prediction_targets in enumerate(predictions_target):
            prediction_targets_set = set(prediction_targets)
            prediction_targets_list = list(prediction_targets)
            confidences = predictions_confidence[i]

            if len(prediction_targets_set) != len(prediction_targets_list):
                raise exceptions.InvalidArgumentValueError(
                    f"Duplicate target values ({prediction_targets_list}) for sample '{predictions.loc[i, INDEX_COLUMN]}'."
                )
            if len(prediction_targets) != len(confidences):
                raise exceptions.InvalidArgumentValueError(
                    f"The number of target values ({len(prediction_targets)}) does not match the number of confidence values ({len(confidences)}) for sample '{predictions.loc[i, INDEX_COLUMN]}'."
                )

            assert not (prediction_targets_set - all_labels_set), (prediction_targets_set, all_labels_set)

            # We have to order confidences to match labels order.
            # If any label is missing in confidences, we add it with confidence 0.
            if all_labels != prediction_targets_list:
                confidences_map = {label: confidence for label, confidence in zip(prediction_targets, confidences)}
                predictions_confidence[i] = tuple(confidences_map.get(label, 0.0) for label in all_labels)

            # Check that all confidences can be converted to float and that they sum to 1.
            sum_confidences = sum(float(confidence) for confidence in predictions_confidence[i])
            if not numpy.isclose(sum_confidences, 1.0):
                raise exceptions.InvalidArgumentValueError(
                    f"Confidences do not sum to 1.0 for sample '{predictions.loc[i, INDEX_COLUMN]}', but {sum_confidences}."
                )

        predictions_confidence_encoded = self.one_hot_encode_confidence(predictions_confidence, all_labels)

        return truth_target_encoded, predictions_confidence_encoded


class RocAucMetric(_RocAucBase):
    """
    Supports binary predictions.
    """

    def score(self, truth: Truth, predictions: Predictions) -> float:
        truth_target_encoded, predictions_confidence_encoded = self.encode_confidence(truth, predictions)

        # We use multi-label ROC AUC to compute for binary target as well.
        scores = metrics.roc_auc_score(truth_target_encoded, predictions_confidence_encoded, average=None)

        if len(scores) != 2:
            raise exceptions.InvalidArgumentValueError("Predictions are not binary.")

        assert numpy.isclose(scores[0], scores[1]), scores

        return scores[0]


class RocAucMicroMetric(_RocAucBase):
    """
    Supports multi-class and multi-label predictions.
    """

    def score(self, truth: Truth, predictions: Predictions) -> float:
        truth_target_encoded, predictions_confidence_encoded = self.encode_confidence(truth, predictions)

        # We use multi-label ROC AUC to compute for multi-class target as well.
        return metrics.roc_auc_score(truth_target_encoded, predictions_confidence_encoded, average='micro')


class RocAucMacroMetric(_RocAucBase):
    """
    Supports multi-class and multi-label predictions.
    """

    def score(self, truth: Truth, predictions: Predictions) -> float:
        truth_target_encoded, predictions_confidence_encoded = self.encode_confidence(truth, predictions)

        # We use multi-label ROC AUC to compute for multi-class target as well.
        return metrics.roc_auc_score(truth_target_encoded, predictions_confidence_encoded, average='macro')


class _RankMetricBase(Metric):
    MAX_RANK = 500

    @classmethod
    def get_merged_truth_predictions(cls, truth: Truth, predictions: Predictions) -> pandas.DataFrame:
        predictions = cls.align(truth, predictions)

        truth_index = cls.get_index_column(truth)
        truth_targets = cls.get_target_columns(truth)

        if len(truth_targets.columns) != 1:
            raise exceptions.InvalidArgumentValueError("Only one target column is supported.")

        truth = pandas.concat([truth_index, truth_targets], axis=1)

        predictions_index = cls.get_index_column(predictions)
        predictions_targets = cls.get_target_columns(predictions)
        predictions_rank = cls.get_rank_column(predictions)

        if len(predictions_targets.columns) != 1:
            raise exceptions.InvalidArgumentValueError("Only one target column is supported.")

        predictions = pandas.concat([predictions_index, predictions_targets, predictions_rank], axis=1)

        merged_truth_predictions = pandas.merge(truth, predictions, how='inner', on=truth.columns.values.tolist())

        # edge-case: none of the true tuples appear in the predictions.
        if merged_truth_predictions.empty:
            return merged_truth_predictions

        # edge-case: some of the tuples does not appear in the predictions. In this case we give missing true tuples a MAX_RANK of 500.
        if merged_truth_predictions.shape[0] != truth.shape[0]:
            outer_merged_truth_predictions = pandas.merge(truth, predictions, how='outer', on=truth.columns.values.tolist())
            non_represented = outer_merged_truth_predictions[outer_merged_truth_predictions[RANK_COLUMN].isnull()]
            non_represented = non_represented.fillna(cls.MAX_RANK)
            merged_truth_predictions = pandas.concat([merged_truth_predictions, non_represented], axis=0)

        return merged_truth_predictions


class MeanReciprocalRankMetric(_RankMetricBase):
    """
    This computes the mean of the reciprocal of elements of a vector of rankings. This metric is used for linkPrediction problems.
    Consider the example:
        learningData:
            d3mIndex    subject object      relationship (target)
            0           James   John        father
            1           John    Patricia    sister
            2           Robert  Thomas      brother
            ...
            ...

        truth:
            d3mIndex    relationship
            0           father
            1           sister
            2           brother

        predictions:
            d3mIndex    relationships   rank
            0           brother         1
            0           cousin          2
            0           mother          3
            0           father          4 *
            0           grandfather     5
            1           sister          1 *
            1           mother          2
            1           aunt            3
            2           father          1
            2           brother         2 *
            2           sister          3
            2           grandfather     4
            2           aunt            5

        Note that ranks (of truth relationships in the predictions) = [4,1,2]
        MRR = np.sum(1/ranks)/len(ranks)
        MRR = 0.58333
    """

    def score(self, truth: Truth, predictions: Predictions) -> float:
        merged_truth_predictions = self.get_merged_truth_predictions(truth, predictions)

        # edge-case: none of the true tuples appear in the predictions. This should return a score of 0.0.
        if merged_truth_predictions.empty:
            return 0.0

        ranks = merged_truth_predictions[RANK_COLUMN].astype(float)
        return numpy.sum(1 / ranks) / len(ranks)


class HitsAtKMetric(_RankMetricBase):
    """
    The computes how many elements of a vector of ranks make it to the top 'k' positions.
    Consider the example:
        learningData:
            d3mIndex    subject object      relationship (target)
            0           James   John        father
            1           John    Patricia    sister
            2           Robert  Thomas      brother
            ...
            ...

        truth:
            d3mIndex    relationship
            0           father
            1           sister
            2           brother

        predictions:
            d3mIndex    relationships   rank
            0           brother         1
            0           cousin          2
            0           mother          3
            0           father          4 *
            0           grandfather     5
            1           sister          1 *
            1           mother          2
            1           aunt            3
            2           father          1
            2           brother         2 *
            2           sister          3
            2           grandfather     4
            2           aunt            5

        Note that ranks (of truth relationships in the predictions) = [4,1,2]
        Hits@3 = 2/3 = 0.666666
        Hits@1 = 1/3 = 0.3333333
        Hits@5 = 3/3 = 1.0
    """

    def __init__(self, k: int) -> None:
        self.k = k

    def score(self, truth: Truth, predictions: Predictions) -> float:
        merged_truth_predictions = self.get_merged_truth_predictions(truth, predictions)

        # edge-case: none of the true tuples appear in the predictions. This should return a score of 0.0.
        if merged_truth_predictions.empty:
            return 0.0

        ranks = merged_truth_predictions[RANK_COLUMN].astype(float)
        return numpy.sum(ranks <= self.k) / len(ranks)


class_map: typing.Dict[problem.PerformanceMetricBase, Metric] = {
    problem.PerformanceMetric.ACCURACY: AccuracyMetric,
    problem.PerformanceMetric.PRECISION: PrecisionMetric,
    problem.PerformanceMetric.RECALL: RecallMetric,
    problem.PerformanceMetric.F1: F1Metric,
    problem.PerformanceMetric.F1_MICRO: F1MicroMetric,
    problem.PerformanceMetric.F1_MACRO: F1MacroMetric,
    problem.PerformanceMetric.MEAN_SQUARED_ERROR: MeanSquareErrorMetric,
    problem.PerformanceMetric.ROOT_MEAN_SQUARED_ERROR: RootMeanSquareErrorMetric,
    problem.PerformanceMetric.MEAN_ABSOLUTE_ERROR: MeanAbsoluteErrorMetric,
    problem.PerformanceMetric.R_SQUARED: RSquaredMetric,
    problem.PerformanceMetric.NORMALIZED_MUTUAL_INFORMATION: NormalizeMutualInformationMetric,
    problem.PerformanceMetric.JACCARD_SIMILARITY_SCORE: JaccardSimilarityScoreMetric,
    problem.PerformanceMetric.PRECISION_AT_TOP_K: PrecisionAtTopKMetric,
    problem.PerformanceMetric.OBJECT_DETECTION_AVERAGE_PRECISION: ObjectDetectionAveragePrecisionMetric,
    problem.PerformanceMetric.HAMMING_LOSS: HammingLossMetric,
    problem.PerformanceMetric.ROC_AUC: RocAucMetric,
    problem.PerformanceMetric.ROC_AUC_MICRO: RocAucMicroMetric,
    problem.PerformanceMetric.ROC_AUC_MACRO: RocAucMacroMetric,
    problem.PerformanceMetric.MEAN_RECIPROCAL_RANK: MeanReciprocalRankMetric,
    problem.PerformanceMetric.HITS_AT_K: HitsAtKMetric,
}
