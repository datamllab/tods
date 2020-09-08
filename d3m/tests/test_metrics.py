import io
import unittest

import pandas
import sklearn
from distutils.version import LooseVersion

from d3m import exceptions, metrics
from d3m.metadata import problem


class TestMetrics(unittest.TestCase):
    def _read_csv(self, csv):
        return pandas.read_csv(
            io.StringIO(csv),
            # We do not want to do any conversion of values at this point.
            # This should be done by primitives later on.
            dtype=str,
            # We always expect one row header.
            header=0,
            # We want empty strings and not NaNs.
            na_filter=False,
            encoding='utf8',
        )

    def test_alignment(self):
        truth = self._read_csv("""
d3mIndex,class_label
1,a
2,b
3,c
4,d
        """)

        predictions = self._read_csv("""
d3mIndex,class_label,confidence
2,b,0.4
4,d,0.5
3,c,0.6
1,a,0.1
        """)

        self.assertEqual(metrics.Metric.align(truth, predictions).values.tolist(), [['1', 'a', '0.1'], ['2', 'b', '0.4'], ['3', 'c', '0.6'], ['4', 'd', '0.5']])

        predictions = self._read_csv("""
d3mIndex,confidence,class_label
1,0.1,a
2,0.4,b
4,0.5,d
3,0.6,c
        """)

        self.assertEqual(metrics.Metric.align(truth, predictions).values.tolist(), [['1', 'a', '0.1'], ['2', 'b', '0.4'], ['3', 'c', '0.6'], ['4', 'd', '0.5']])

        predictions = self._read_csv("""
confidence,class_label,d3mIndex
0.1,a,1
0.4,b,2
0.5,d,4
0.6,c,3
        """)

        self.assertEqual(metrics.Metric.align(truth, predictions).values.tolist(), [['1', 'a', '0.1'], ['2', 'b', '0.4'], ['3', 'c', '0.6'], ['4', 'd', '0.5']])

        predictions = self._read_csv("""
d3mIndex
1
2
4
3
        """)

        with self.assertRaises(exceptions.InvalidArgumentValueError):
            metrics.Metric.align(truth, predictions)

        predictions = self._read_csv("""
d3mIndex,class_label,confidence
1,a,0.1
2,b,0.4
3,c,0.6
        """)

        with self.assertRaises(exceptions.InvalidArgumentValueError):
            metrics.Metric.align(truth, predictions)

        truth = self._read_csv("""
d3mIndex,class_label
1,a1
1,a2
2,b
3,c1
3,c2
3,c3
4,d1
4,d2
        """)

        predictions = self._read_csv("""
d3mIndex,class_label
2,b
4,d
3,c
1,a
        """)

        self.assertEqual(metrics.Metric.align(truth, predictions).values.tolist(), [['1', 'a'], ['2', 'b'], ['3', 'c'], ['4', 'd']])

        predictions = self._read_csv("""
d3mIndex,class_label
4,d
2,b1
2,b2
2,b3
2,b4
2,b5
2,b6
3,c
1,a
        """)

        self.assertEqual(metrics.Metric.align(truth, predictions).values.tolist(), [['1', 'a'], ['2', 'b1'], ['2', 'b2'], ['2', 'b3'], ['2', 'b4'], ['2', 'b5'], ['2', 'b6'], ['3', 'c'], ['4', 'd']])

        truth = self._read_csv("""
d3mIndex,class_label
1,a1
1,a2
3,c1
2,b
3,c2
3,c3
4,d1
4,d2
        """)

        with self.assertRaises(exceptions.InvalidArgumentValueError):
            metrics.Metric.align(truth, predictions)

    def test_labels(self):
        pred_df = pandas.DataFrame(columns=['d3mIndex', 'class'], dtype=object)
        pred_df['d3mIndex'] = pandas.Series([0, 1, 2, 3, 4])
        pred_df['class'] = pandas.Series(['a', 'b', 'a', 'b', 'b'])

        ground_truth_df = pandas.DataFrame(columns=['d3mIndex', 'class'], dtype=object)
        ground_truth_df['d3mIndex'] = pandas.Series([0, 1, 2, 3, 4])
        ground_truth_df['class'] = pandas.Series(['a', 'b', 'a', 'b', 'a'])

        precision_metric = metrics.PrecisionMetric(pos_label='a')
        self.assertEqual(precision_metric.score(ground_truth_df, pred_df), 1.0)

        precision_metric = metrics.PrecisionMetric(pos_label='b')
        self.assertAlmostEqual(precision_metric.score(ground_truth_df, pred_df), 0.6666666666666666)

    def test_hamming_loss(self):
        # Testcase 1: MultiLabel, typical

        y_true = self._read_csv("""
d3mIndex,class_label
3,happy-pleased
3,relaxing-calm
7,amazed-suprised
7,happy-pleased
13,quiet-still
13,sad-lonely
        """)

        y_pred = self._read_csv("""
d3mIndex,class_label
3,happy-pleased
3,sad-lonely
7,amazed-suprised
7,happy-pleased
13,quiet-still
13,happy-pleased
        """)

        self.assertAlmostEqual(metrics.HammingLossMetric().score(y_true, y_pred), 0.26666666666666666)

        # Testcase 2: MultiLabel, Zero loss

        y_true = self._read_csv("""
d3mIndex,class_label
3,happy-pleased
3,relaxing-calm
7,amazed-suprised
7,happy-pleased
13,quiet-still
13,sad-lonely
        """)

        y_pred = self._read_csv("""
d3mIndex,class_label
3,happy-pleased
3,relaxing-calm
7,amazed-suprised
7,happy-pleased
13,quiet-still
13,sad-lonely
        """)

        self.assertAlmostEqual(metrics.HammingLossMetric().score(y_true, y_pred), 0.0)

        # Testcase 3: MultiLabel, Complete loss

        y_true = self._read_csv("""
d3mIndex,class_label
3,happy-pleased
3,relaxing-calm
7,amazed-suprised
7,happy-pleased
13,quiet-still
13,sad-lonely
        """)

        y_pred = self._read_csv("""
d3mIndex,class_label
3,ecstatic
3,sad-lonely
3,quiet-still
3,amazed-suprised
7,ecstatic
7,sad-lonely
7,relaxing-calm
7,quiet-still
13,ecstatic
13,happy-pleased
13,relaxing-calm
13,amazed-suprised
        """)

        self.assertAlmostEqual(metrics.HammingLossMetric().score(y_true, y_pred), 1.0)

        # Testcase 4: Multiclass, case 1
        # Multiclass is not really supported or reasonable to use, but we still test it to test also edge cases.

        y_true = self._read_csv("""
d3mIndex,species
2,versicolor
16,virginica
17,setosa
22,versicolor
30,versicolor
31,virginica
26,versicolor
33,versicolor
1,versicolor
37,virginica
        """)

        y_pred = self._read_csv("""
d3mIndex,species
1,setosa
2,versicolor
22,versicolor
26,virginica
30,versicolor
31,virginica
33,versicolor
17,setosa
37,virginica
16,virginica
        """)

        self.assertAlmostEqual(metrics.HammingLossMetric().score(y_true, y_pred), 0.1333333)

         # Testcase 5: Multiclass, case 2
        # Multiclass is not really supported or reasonable to use, but we still test it to test also edge cases.

        y_true = self._read_csv("""
d3mIndex,species
1,versicolor
2,versicolor
16,virginica
17,setosa
22,versicolor
26,versicolor
30,versicolor
31,virginica
33,versicolor
37,virginica
        """)

        y_pred = self._read_csv("""
d3mIndex,species
1,versicolor
2,versicolor
16,virginica
17,setosa
22,versicolor
26,versicolor
30,versicolor
31,virginica
33,versicolor
37,virginica
        """)

        self.assertAlmostEqual(metrics.HammingLossMetric().score(y_true, y_pred), 0.0)

        # Testcase 6: Multiclass, case 3
        # Multiclass is not really supported or reasonable to use, but we still test it to test also edge cases.

        y_true = self._read_csv("""
d3mIndex,species
1,versicolor
2,versicolor
16,versicolor
17,virginica
22,versicolor
26,versicolor
30,versicolor
31,virginica
33,versicolor
37,virginica
        """)

        y_pred = self._read_csv("""
d3mIndex,species
1,setosa
2,setosa
16,setosa
17,setosa
22,setosa
26,setosa
30,setosa
31,setosa
33,setosa
37,setosa
        """)

        self.assertAlmostEqual(metrics.HammingLossMetric().score(y_true, y_pred), 0.66666666)

    def test_root_mean_squared_error(self):
        y_true = self._read_csv("""
d3mIndex,value
1,3
2,-1.0
17,7
16,2
        """)

        # regression univariate, regression multivariate, forecasting, collaborative filtering
        y_pred = self._read_csv("""
d3mIndex,value
1,2.1
2,0.0
16,2
17,8
        """)

        self.assertAlmostEqual(metrics.RootMeanSquareErrorMetric().score(y_true, y_pred), 0.8381527307120105)

        y_true = self._read_csv("""
d3mIndex,value1,value2
1,0.5,1
2,-1,1
16,7,-6
        """)

        y_pred = self._read_csv("""
d3mIndex,value1,value2
1,0,2
2,-1,2
16,8,-5
        """)

        self.assertAlmostEqual(metrics.RootMeanSquareErrorMetric().score(y_true, y_pred), 0.8227486121839513)

    def test_precision_at_top_k(self):
        # Forecasting test
        ground_truth_list_1 = self._read_csv("""
d3mIndex,value
1,1
6,6
2,10
4,5
5,12
7,2
8,18
3,7
9,4
10,8
        """)
        predictions_list_1 = self._read_csv("""
d3mIndex,value
1,0
10,11
2,2
4,6
5,14
6,9
7,3
8,17
9,10
3,8
        """)
        self.assertAlmostEqual(metrics.PrecisionAtTopKMetric(k=5).score(ground_truth_list_1, predictions_list_1), 0.6)

    def test_object_detection_average_precision(self):
        # Object Dectection test
        predictions_list_1 = self._read_csv("""
d3mIndex,box,confidence
1,"110,110,110,210,210,210,210,110",0.6
2,"5,10,5,20,20,20,20,10",0.9
2,"120,130,120,200,200,200,200,130",0.6
        """)

        ground_truth_list_1 = self._read_csv("""
d3mIndex,box
1,"100,100,100,200,200,200,200,100"
2,"10,10,10,20,20,20,20,10"
2,"70,80,70,150,140,150,140,80"
        """)

        self.assertAlmostEqual(metrics.ObjectDetectionAveragePrecisionMetric().score(ground_truth_list_1, predictions_list_1), 0.6666666666666666)

        predictions_list_2 = self._read_csv("""
d3mIndex,box,confidence
285,"330,463,330,505,387,505,387,463",0.0739
285,"420,433,420,498,451,498,451,433",0.0910
285,"328,465,328,540,403,540,403,465",0.1008
285,"480,477,480,522,508,522,508,477",0.1012
285,"357,460,357,537,417,537,417,460",0.1058
285,"356,456,356,521,391,521,391,456",0.0843
225,"345,460,345,547,415,547,415,460",0.0539
225,"381,362,381,513,455,513,455,362",0.0542
225,"382,366,382,422,416,422,416,366",0.0559
225,"730,463,730,583,763,583,763,463",0.0588
        """)

        ground_truth_list_2 = self._read_csv("""
d3mIndex,box
285,"480,457,480,529,515,529,515,457"
285,"480,457,480,529,515,529,515,457"
225,"522,540,522,660,576,660,576,540"
225,"739,460,739,545,768,545,768,460"
        """)

        self.assertAlmostEqual(metrics.ObjectDetectionAveragePrecisionMetric().score(ground_truth_list_2, predictions_list_2), 0.125)

        predictions_list_3 = self._read_csv("""
d3mIndex,box,confidence
1,"110,110,110,210,210,210,210,110",0.6
2,"120,130,120,200,200,200,200,130",0.6
2,"5,8,5,16,15,16,15,8",0.9
2,"11,12,11,18,21,18,21,12",0.9
        """)

        ground_truth_list_3 = self._read_csv("""
d3mIndex,box
1,"100,100,100,200,200,200,200,100"
2,"10,10,10,20,20,20,20,10"
2,"70,80,70,150,140,150,140,80"
        """)

        self.assertAlmostEqual(metrics.ObjectDetectionAveragePrecisionMetric().score(ground_truth_list_3, predictions_list_3), 0.4444444444444444)

        predictions_list_4 = self._read_csv("""
d3mIndex,box,confidence
1,"110,110,110,210,210,210,210,110",0.6
2,"120,130,120,200,200,200,200,130",0.6
2,"11,12,11,18,21,18,21,12",0.9
2,"5,8,5,16,15,16,15,8",0.9
        """)

        ground_truth_list_4 = self._read_csv("""
d3mIndex,box
1,"100,100,100,200,200,200,200,100"
2,"10,10,10,20,20,20,20,10"
2,"70,80,70,150,140,150,140,80"
        """)

        self.assertAlmostEqual(metrics.ObjectDetectionAveragePrecisionMetric().score(ground_truth_list_4, predictions_list_4), 0.4444444444444444)

    def test_normalized_mutual_info_score(self):
        # Community Detection Test
        predictions_list_1 = self._read_csv("""
d3mIndex,Class
0,2
1,2
2,1
3,1
        """)

        ground_truth_list_1 = self._read_csv("""
d3mIndex,Class
0,1
1,1
2,1
3,1
        """)

        self.assertAlmostEqual(metrics.NormalizeMutualInformationMetric().score(ground_truth_list_1, predictions_list_1), 0.5)

    def test_f1_score(self):
        # MultiTask MultiClass Classification
        y_true = self._read_csv("""
d3mIndex,value1,value2
1,1,1
2,3,2
16,4,1
        """)

        y_pred = self._read_csv("""
d3mIndex,value1,value2
1,1,2
2,3,1
16,4,2
        """)

        self.assertAlmostEqual(metrics.F1MacroMetric().score(y_true, y_pred), 0.5)
        self.assertAlmostEqual(metrics.F1MicroMetric().score(y_true, y_pred), 0.5)

        # MultiClass Classification Test
        y_true = self._read_csv("""
d3mIndex,class_label
1,0
2,1
3,2
4,3
        """)

        y_pred = self._read_csv("""
d3mIndex,class_label
1,0
2,2
3,1
4,3
        """)

        self.assertAlmostEqual(metrics.F1MacroMetric().score(y_true, y_pred), 0.5)
        self.assertAlmostEqual(metrics.F1MicroMetric().score(y_true, y_pred), 0.5)

        # MultiTask Binary Classification
        y_true = self._read_csv("""
d3mIndex,value1,value2
1,1,1
2,0,0
16,0,1
        """)

        y_pred = self._read_csv("""
d3mIndex,value1,value2
1,1,1
2,0,1
16,0,0
        """)

        self.assertAlmostEqual(metrics.F1Metric(pos_label='1').score(y_true, y_pred), 0.75)

        # MultiLabel Classification Test
        y_true = self._read_csv("""
d3mIndex,class_label
1,3
1,1
2,2
3,3
        """)

        y_pred = self._read_csv("""
d3mIndex,class_label
1,1
1,2
1,3
2,1
3,3
        """)

        self.assertEqual(metrics.F1MacroMetric().score(y_true, y_pred), 0.5555555555555555)
        self.assertAlmostEqual(metrics.F1MicroMetric().score(y_true, y_pred), 0.6666666666666665)

        # MultiTask MultiLabel Classification Test
        y_true = self._read_csv("""
d3mIndex,value1,value2
1,3,1
1,1,
2,2,0
3,3,1
3,3,3
        """)

        y_pred = self._read_csv("""
d3mIndex,value1,value2
1,1,1
1,2,
1,3,
2,1,3
2,,3
3,3,0
        """)

        self.assertEqual(metrics.F1MacroMetric().score(y_true, y_pred), 0.38888888888888884)
        self.assertAlmostEqual(metrics.F1MicroMetric().score(y_true, y_pred), 0.47619047619047616)

    def test_all_labels(self):
        y_true = self._read_csv("""
d3mIndex,class_label
3,happy-pleased
3,relaxing-calm
7,amazed-suprised
7,happy-pleased
13,quiet-still
13,sad-lonely
        """)

        y_pred = self._read_csv("""
d3mIndex,class_label
3,happy-pleased
3,sad-lonely
7,amazed-suprised
7,happy-pleased
13,quiet-still
13,happy-pleased
        """)

        self.assertAlmostEqual(metrics.HammingLossMetric(all_labels={'class_label': ['happy-pleased', 'relaxing-calm', 'amazed-suprised', 'quiet-still', 'sad-lonely', 'foobar']}).score(y_true, y_pred), 0.2222222222222222)

        with self.assertRaisesRegex(exceptions.InvalidArgumentValueError, 'Truth contains extra labels'):
            self.assertAlmostEqual(metrics.HammingLossMetric(all_labels={'class_label': ['happy-pleased', 'relaxing-calm', 'amazed-suprised']}).score(y_true, y_pred), 0.2222222222222222)

    def test_duplicate_columns(self):
        y_true = self._read_csv("""
d3mIndex,value1,value2
1,1,1
16,4,1
2,3,2
        """)

        y_pred = self._read_csv("""
d3mIndex,value1,value2
1,1,2
2,3,1
16,4,2
        """)

        y_true.columns = ('d3mIndex', 'value1', 'value1')
        y_pred.columns = ('d3mIndex', 'value1', 'value1')

        with self.assertRaises(exceptions.InvalidArgumentValueError):
            (metrics.F1MicroMetric().score(y_true, y_pred), 0.5)

    def test_precision(self):
        # Binary Classification Test
        y_true = self._read_csv("""
d3mIndex,class_label
1,pos
2,pos
3,neg
4,neg
5,pos
        """)

        y_pred = self._read_csv("""
d3mIndex,class_label
1,pos
2,pos
3,neg
4,neg
5,neg
        """)
        self.assertEqual(metrics.PrecisionMetric("pos").score(y_true, y_pred), 1.0)

        y_pred_2 = self._read_csv("""
d3mIndex,class_label
1,pos
2,pos
3,pos
4,pos
5,neg
        """)

        self.assertEqual(metrics.PrecisionMetric("pos").score(y_true, y_pred_2), 0.5)

        y_pred_3 = self._read_csv("""
d3mIndex,class_label
1,neg
2,neg
3,pos
4,pos
5,neg
        """)

        self.assertEqual(metrics.PrecisionMetric("pos").score(y_true, y_pred_3), 0.0)

    def test_accuracy(self):
        # Binary Classification Test
        y_true = self._read_csv("""
d3mIndex,class_label
1,pos
2,pos
3,neg
4,neg
5,pos
        """)

        y_pred = self._read_csv("""
d3mIndex,class_label
1,pos
2,pos
3,neg
4,neg
5,pos
        """)

        self.assertEqual(metrics.AccuracyMetric().score(y_true, y_pred), 1.0)

        y_pred_2 = self._read_csv("""
d3mIndex,class_label
1,pos
2,pos
3,pos
4,pos
5,neg
        """)

        self.assertEqual(metrics.AccuracyMetric().score(y_true, y_pred_2), 0.4)

        y_pred_3 = self._read_csv("""
d3mIndex,class_label
1,neg
2,neg
3,pos
4,pos
5,neg
        """)

        self.assertEqual(metrics.AccuracyMetric().score(y_true, y_pred_3), 0.0)

        # MultiClass Classification Test
        y_true = self._read_csv("""
d3mIndex,class_label
1,0
2,1
3,2
4,3
        """)

        y_pred = self._read_csv("""
d3mIndex,class_label
1,0
2,2
4,3
3,1
        """)

        self.assertEqual(metrics.AccuracyMetric().score(y_true, y_pred), 0.5)

        # MultiLabel Classification Test
        y_true = self._read_csv("""
d3mIndex,class_label
1,3
1,1
2,2
3,3
        """)

        y_pred = self._read_csv("""
d3mIndex,class_label
1,1
1,2
1,3
2,1
3,3
        """)

        self.assertEqual(metrics.AccuracyMetric().score(y_true, y_pred), 0.3333333333333333)

    def test_mean_squared_error(self):
        # regression univariate, regression multivariate, forecasting, collaborative filtering
        y_true = self._read_csv("""
d3mIndex,value
1,3
16,2
2,-1.0
17,7
        """)

        y_pred = self._read_csv("""
d3mIndex,value
1,2.1
2,0.0
16,2
17,8
        """)

        self.assertAlmostEqual(metrics.MeanSquareErrorMetric().score(y_true, y_pred), 0.7024999999999999)

        y_true = self._read_csv("""
d3mIndex,value1,value2
1,0.5,1
2,-1,1
16,7,-6
        """)

        y_pred = self._read_csv("""
d3mIndex,value1,value2
1,0,2
2,-1,2
16,8,-5
        """)

        self.assertAlmostEqual(metrics.MeanSquareErrorMetric().score(y_true, y_pred), 0.7083333333333334)

    def test_mean_absolute_error(self):
        # regression univariate, regression multivariate, forecasting, collaborative filtering
        y_true = self._read_csv("""
d3mIndex,value
1,3
2,-0.5
16,2
17,7
        """)

        y_pred = self._read_csv("""
d3mIndex,value
17,8
1,2.5
2,0.0
16,2
        """)

        self.assertAlmostEqual(metrics.MeanAbsoluteErrorMetric().score(y_true, y_pred), 0.5)

        y_true = self._read_csv("""
d3mIndex,value1,value2
1,0.5,1
2,-1,1
16,7,-6
        """)

        y_pred = self._read_csv("""
d3mIndex,value2,value1
1,2,0
16,-5,8
2,2,-1
        """)

        self.assertAlmostEqual(metrics.MeanAbsoluteErrorMetric().score(y_true, y_pred), 0.75)

    def test_r_squared(self):
        # regression univariate, regression multivariate, forecasting, collaborative filtering
        y_true = self._read_csv("""
d3mIndex,value
1,3
2,-0.5
16,2
17,7
        """)

        y_pred = self._read_csv("""
d3mIndex,value
1,2.5
2,0.0
16,2
17,8
        """)

        self.assertAlmostEqual(metrics.RSquaredMetric().score(y_true, y_pred), 0.9486081370449679)

        y_true = self._read_csv("""
d3mIndex,value1,value2
1,0.5,1
2,-1,1
16,7,-6
        """)

        y_pred = self._read_csv("""
d3mIndex,value2,value1
1,2,0
16,-5,8
2,2,-1
        """)

        self.assertAlmostEqual(metrics.RSquaredMetric().score(y_true, y_pred), 0.9368005266622779)

        y_true = self._read_csv("""
d3mIndex,value
1,1
2,2
16,3
        """)

        y_pred = self._read_csv("""
d3mIndex,value
1,1
2,2
16,3
        """)

        self.assertAlmostEqual(metrics.RSquaredMetric().score(y_true, y_pred), 1.0)

        y_true = self._read_csv("""
d3mIndex,value
1,1
2,2
16,3
        """)

        y_pred = self._read_csv("""
d3mIndex,value
1,2
2,2
16,2
        """)

        self.assertAlmostEqual(metrics.RSquaredMetric().score(y_true, y_pred), 0.0)

        y_true = self._read_csv("""
d3mIndex,value
1,1
2,2
16,3
        """)

        y_pred = self._read_csv("""
d3mIndex,value
1,3
2,2
16,1
        """)

        self.assertAlmostEqual(metrics.RSquaredMetric().score(y_true, y_pred), -3.0)

    def test_recall(self):
        # Binary Classification Test
        y_true = self._read_csv("""
d3mIndex,value
1,0
6,1
3,0
4,0
5,1
2,1
        """)

        y_pred = self._read_csv("""
d3mIndex,value
3,1
1,0
2,1
4,0
5,0
6,1
        """)

        self.assertAlmostEqual(metrics.RecallMetric(pos_label='1').score(y_true, y_pred), 0.6666666666666666)

    @unittest.skipUnless(sklearn.__version__ >= LooseVersion("0.21"), "jaccard_score introduced in sklearn version 0.21")
    def test_jaccard(self):
        # Binary Classification Test
        y_true = self._read_csv("""
d3mIndex,value
1,0
2,1
16,1
        """)

        y_pred = self._read_csv("""
d3mIndex,value
1,1
2,1
16,1
        """)

        self.assertAlmostEqual(metrics.JaccardSimilarityScoreMetric(pos_label='1').score(y_true, y_pred), 0.6666666666666666)


    def test_meanReciprocalRank(self):
        y_true = self._read_csv("""
d3mIndex,relationship
0,father
1,sister
2,brother
        """)

        # case 1: all correct
        y_pred = self._read_csv("""
d3mIndex,relationship,rank
0,father,1
0,cousin,2
0,mother,3
0,brother,4
0,grandfather,5
1,sister,1
1,mother,2
1,aunt,3
2,brother,1
2,father,2
2,sister,3
2,grandfather,4
2,aunt,5
        """)
        self.assertAlmostEqual(metrics.MeanReciprocalRankMetric().score(y_true, y_pred), 1.0)

        # case 2: all wrong
        y_pred = self._read_csv("""
d3mIndex,relationship,rank
0,brother,1
0,cousin,2
0,mother,3
0,grandfather,4
1,brother,1
1,mother,2
1,aunt,3
2,father,1
2,grandmother,2
2,sister,3
2,grandfather,4
2,aunt,5
        """)
        self.assertAlmostEqual(metrics.MeanReciprocalRankMetric().score(y_true, y_pred), 0.0)

        # case 3 (typical case): some correct and some low ranks
        y_pred = self._read_csv("""
d3mIndex,relationship,rank
0,brother,1
0,cousin,2
0,mother,3
0,father,4
0,grandfather,5
1,sister,1
1,mother,2
1,aunt,3
2,father,1
2,brother,2
2,sister,3
2,grandfather,4
2,aunt,5
        """)
        self.assertAlmostEqual(metrics.MeanReciprocalRankMetric().score(y_true, y_pred), 0.5833333333333334)

        # case 4: some are not ranked at all
        y_pred = self._read_csv("""
d3mIndex,relationship,rank
0,brother,1
0,cousin,2
0,mother,3
0,grandfather,4
1,sister,1
1,mother,2
1,aunt,3
2,father,1
2,uncle,2
2,sister,3
2,grandfather,4
2,aunt,5
        """)
        self.assertAlmostEqual(metrics.MeanReciprocalRankMetric().score(y_true, y_pred), 0.33466666666666667)

    def test_hitsAtK(self):
        y_true = self._read_csv("""
d3mIndex,relationship
0,father
1,sister
2,brother
        """)

        # case 1: all correct
        y_pred = self._read_csv("""
d3mIndex,relationship,rank
0,father,1
0,cousin,2
0,mother,3
0,brother,4
0,grandfather,5
1,sister,1
1,mother,2
1,aunt,3
2,brother,1
2,father,2
2,sister,3
2,grandfather,4
2,aunt,5
        """)
        self.assertAlmostEqual(metrics.HitsAtKMetric(k=3).score(y_true, y_pred), 1.0)

        # case 2: all wrong
        y_pred = self._read_csv("""
d3mIndex,relationship,rank
0,brother,1
0,cousin,2
0,mother,3
0,grandfather,4
1,brother,1
1,mother,2
1,aunt,3
2,father,1
2,grandmother,2
2,sister,3
2,grandfather,4
2,aunt,5
        """)
        self.assertAlmostEqual(metrics.HitsAtKMetric(k=3).score(y_true, y_pred), 0.0)

        # case 3 (typical case): some correct and some low ranks
        y_pred = self._read_csv("""
d3mIndex,relationship,rank
0,brother,1
0,cousin,2
0,mother,3
0,father,4
0,grandfather,5
1,sister,1
1,mother,2
1,aunt,3
2,father,1
2,brother,2
2,sister,3
2,grandfather,4
2,aunt,5
        """)
        self.assertAlmostEqual(metrics.HitsAtKMetric(k=3).score(y_true, y_pred), 0.6666666666666666)
        self.assertAlmostEqual(metrics.HitsAtKMetric(k=1).score(y_true, y_pred), 0.3333333333333333)
        self.assertAlmostEqual(metrics.HitsAtKMetric(k=5).score(y_true, y_pred), 1.0)

        # case 4: some are not ranked at all
        y_pred = self._read_csv("""
d3mIndex,relationship,rank
0,brother,1
0,cousin,2
0,mother,3
0,grandfather,4
1,sister,1
1,mother,2
1,aunt,3
2,father,1
2,uncle,2
2,sister,3
2,grandfather,4
2,aunt,5
        """)
        self.assertAlmostEqual(metrics.HitsAtKMetric(k=3).score(y_true, y_pred), 0.3333333)

    def test_custom_metric(self):
        class FooBar():
            def score(self, truth: metrics.Truth, predictions: metrics.Predictions) -> float:
                return 1.0

        problem.PerformanceMetric.register_metric('FOOBAR', best_value=1.0, worst_value=0.0, score_class=FooBar)

        self.assertEqual(problem.PerformanceMetric.FOOBAR.best_value(), 1.0)
        self.assertEqual(problem.PerformanceMetric['FOOBAR'].worst_value(), 0.0)
        self.assertEqual(problem.PerformanceMetric('FOOBAR').requires_confidence(), False)
        self.assertIs(problem.PerformanceMetric.FOOBAR.get_class(), FooBar)

    def test_roc_auc(self):
        # Binary Classification Test
        y_true = self._read_csv("""
d3mIndex,value
640,0
641,1
642,0
643,0
644,1
645,1
646,0
        """)

        y_pred = self._read_csv("""
d3mIndex,value,confidence
640,0,0.612
640,1,0.388
641,0,0.6
641,1,0.4
645,1,0.9
645,0,0.1
642,1,0.0
642,0,1.0
643,0,0.52
643,1,0.48
644,0,0.3
644,1,0.7
646,0,1.0
646,1,0.0
        """)

        self.assertAlmostEqual(metrics.RocAucMetric().score(y_true, y_pred), 0.9166666666666667)

    def test_roc_auc_micro(self):
        # Testcase 1: MultiLabel, typical

        y_true = self._read_csv("""
d3mIndex,value
3,d
4,a
4,b
4,c
7,a
7,b
7,d
9,b
9,e
        """)

        y_pred = self._read_csv("""
d3mIndex,value,confidence
9,b,0.1
4,a,0.4
4,b,0.3
3,a,0.2
3,b,0.1
3,c,0.6
3,d,0.1
3,e,0
4,c,0.1
4,e,0.1
4,d,0.1
7,a,0.1
7,b,0.1
7,d,0.7
7,c,0.1
7,e,0
9,a,0.4
9,c,0.15
9,d,0.3
9,e,0.05
        """)
        self.assertAlmostEqual(metrics.RocAucMicroMetric().score(y_true, y_pred), 0.5151515151515151)

    def test_roc_auc_macro(self):
        # Testcase 1: MultiLabel, typical

        y_true = self._read_csv("""
d3mIndex,value
3,d
4,a
4,b
4,c
7,a
7,b
7,d
9,b
9,e
        """)

        y_pred = self._read_csv("""
d3mIndex,value,confidence
3,a,0.2
3,b,0.1
3,c,0.6
3,d,0.1
3,e,0
7,b,0.1
7,a,0.1
4,a,0.4
4,b,0.3
4,c,0.1
4,d,0.1
4,e,0.1
9,a,0.4
9,b,0.1
9,c,0.15
9,d,0.3
9,e,0.05
7,c,0.1
7,d,0.7
7,e,0
        """)
        self.assertAlmostEqual(metrics.RocAucMacroMetric().score(y_true, y_pred), 0.5)


if __name__ == '__main__':
    unittest.main()
