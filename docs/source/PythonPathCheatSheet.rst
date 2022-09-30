Python Path Cheat Sheet
=======================

Data Processing
---------------

  .. list-table:: 
     :widths: 50 50
     :header-rows: 1

     * - Python Path
       - Primitives
     * - `tods.data_processing.categorical_to_binary` 
       - :ref:`tods.data_processing.CategoricalToBinary:CategoricalToBinaryPrimitive<tods.data_processing.CategoricalToBinary>`
     * - `tods.data_processing.column_filter`
       - :ref:`tods.data_processing.ColumnFilter:ColumnFilterPrimitive<tods.data_processing.ColumnFilter>`
     * - `tods.data_processing.column_parser`
       - :ref:`tods.data_processing.ColumnParser:ColumnParserPrimitive<tods.data_processing.ColumnParser>`
     * - `tods.data_processing.construct_predictions`
       - :ref:`tods.data_processing.ConstructPredictions:ConstructPredictionsPrimitive<tods.data_processing.ConstructPredictions>`
     * - `tods.data_processing.continuity_validation`
       - :ref:`tods.data_processing.ContinuityValidation:ContinuityValidationPrimitive<tods.data_processing.ContinuityValidation>`
     * - `tods.data_processing.dataset_to_dataframe`
       - :ref:`tods.data_processing.DatasetToDataframe:DatasetToDataFramePrimitive<tods.data_processing.DatasetToDataframe>`
     * - `tods.data_processing.duplication_validation`
       - :ref:`tods.data_processing.DuplicationValidation:DuplicationValidationPrimitive<tods.data_processing.DuplicationValidation>`
     * - `tods.data_processing.extract_columns_by_semantic_types`
       - :ref:`tods.data_processing.ExtractColumnsBySemanticTypes:ExtractColumnsBySemanticTypesPrimitive<tods.data_processing.ExtractColumnsBySemanticTypes>`
     * - `tods.data_processing.impute_missing`
       - :ref:`tods.data_processing.SKImputer:SKImputerPrimitive<tods.data_processing.SKImputer>`
     * - `tods.data_processing.time_interval_transform`
       - :ref:`tods.data_processing.TimeIntervalTransform:TimeIntervalTransformPrimitive<tods.data_processing.TimeIntervalTransform>`
     * - `tods.data_processing.timestamp_validation`
       - :ref:`tods.data_processing.TimeStampValidation:TimeStampValidationPrimitive<tods.data_processing.TimeStampValidation>`

Timeseries Processing
---------------------

   .. list-table:: 
      :widths: 50 50
      :header-rows: 1

      * - Python Path
        - Primitives
      * - `tods.timeseries_processing.transformation.holt_smoothing`
        - :ref:`tods.timeseries_processing.HoltSmoothing:HoltSmoothingPrimitive<tods.timeseries_processing.HoltSmoothing>`
      * - `tods.timeseries_processing.transformation.holt_winters_exponential_smoothing`
        - :ref:`tods.timeseries_processing.HoltWintersExponentialSmoothing:HoltWintersExponentialSmoothingPrimitive<tods.timeseries_processing.HoltWintersExponentialSmoothing>`
      * - `tods.timeseries_processing.transformation.moving_average_transform`
        - :ref:`tods.timeseries_processing.MovingAverageTransformer:MovingAverageTransformerPrimitive<tods.timeseries_processing.MovingAverageTransformer>`
      * - `tods.timeseries_processing.transformation.axiswise_scaler`
        - :ref:`tods.timeseries_processing.SKAxiswiseScaler:SKAxiswiseScalerPrimitive<tods.timeseries_processing.SKAxiswiseScaler>`
      * - `tods.timeseries_processing.transformation.power_transformer`
        - :ref:`tods.timeseries_processing.SKPowerTransformer:SKPowerTransformerPrimitive<tods.timeseries_processing.SKPowerTransformer>`
      * - `tods.timeseries_processing.transformation.quantile_transformer`
        - :ref:`tods.timeseries_processing.SKQuantileTransformer:SKQuantileTransformerPrimitive<tods.timeseries_processing.SKQuantileTransformer>`
      * - `tods.timeseries_processing.transformation.standard_scaler`
        - :ref:`tods.timeseries_processing.SKStandardScaler:SKStandardScalerPrimitive<tods.timeseries_processing.SKStandardScaler>`
      * - `tods.timeseries_processing.transformation.simple_exponential_smoothing`
        - :ref:`tods.timeseries_processing.SimpleExponentialSmoothing:SimpleExponentialSmoothingPrimitive<tods.timeseries_processing.SimpleExponentialSmoothing>`
      * - `tods.timeseries_processing.subsequence_segmentation`
        - :ref:`tods.timeseries_processing.SubsequenceSegmentation:SubsequenceSegmentationPrimitive<tods.timeseries_processing.SubsequenceSegmentation>`
      * - `tods.timeseries_processing.decomposition.time_series_seasonality_trend_decomposition`
        - :ref:`tods.timeseries_processing.TimeSeriesSeasonalityTrendDecomposition:TimeSeriesSeasonalityTrendDecompositionPrimitive<tods.timeseries_processing.TimeSeriesSeasonalityTrendDecomposition>`

Feature Analysis
----------------

   .. list-table:: 
      :widths: 50 50
      :header-rows: 1

      * - Python Path
        - Primitives
      * - `tods.feature_analysis.auto_correlation`
        - :ref:`tods.feature_analysis.AutoCorrelation:AutoCorrelationPrimitive<tods.feature_analysis.AutoCorrelation>`
      * - `tods.feature_analysis.bk_filter`
        - :ref:`tods.feature_analysis.BKFilter:BKFilterPrimitive<tods.feature_analysis.BKFilter>`
      * - `tods.feature_analysis.discrete_cosine_transform`
        - :ref:`tods.feature_analysis.DiscreteCosineTransform:DiscreteCosineTransformPrimitive<tods.feature_analysis.DiscreteCosineTransform>`
      * - `tods.feature_analysis.fast_fourier_transform`
        - :ref:`tods.feature_analysis.FastFourierTransform:FastFourierTransformPrimitive<tods.feature_analysis.FastFourierTransform>`
      * - `tods.feature_analysis.hp_filter` 
        - :ref:`tods.feature_analysis.HPFilter:HPFilterPrimitive<tods.feature_analysis.HPFilter>`
      * - `tods.feature_analysis.non_negative_matrix_factorization`
        - :ref:`tods.feature_analysis.NonNegativeMatrixFactorization:NonNegativeMatrixFactorizationPrimitive<tods.feature_analysis.NonNegativeMatrixFactorization>`
      * - `tods.feature_analysis.truncated_svd`
        - :ref:`tods.feature_analysis.SKTruncatedSVD:SKTruncatedSVDPrimitive<tods.feature_analysis.SKTruncatedSVD>`
      * - `tods.feature_analysis.spectral_residual_transform`
        - :ref:`tods.feature_analysis.SpectralResidualTransform:SpectralResidualTransformPrimitive<tods.feature_analysis.SpectralResidualTransform>`
      * - `tods.feature_analysis.statistical_abs_energy`
        - :ref:`tods.feature_analysis.StatisticalAbsEnergy:StatisticalAbsEnergyPrimitive<tods.feature_analysis.StatisticalAbsEnergy>`
      * - `tods.feature_analysis.statistical_abs_sum`
        - :ref:`tods.feature_analysis.StatisticalAbsSum:StatisticalAbsSumPrimitive<tods.feature_analysis.StatisticalAbsSum>`
      * - `tods.feature_analysis.statistical_g_mean`
        - :ref:`tods.feature_analysis.StatisticalGmean:StatisticalGmeanPrimitive<tods.feature_analysis.StatisticalGmean>`
      * - `tods.feature_analysis.statistical_h_mean`
        - :ref:`tods.feature_analysis.StatisticalHmean:StatisticalHmeanPrimitive<tods.feature_analysis.StatisticalHmean>`
      * - `tods.feature_analysis.statistical_kurtosis`
        - :ref:`tods.feature_analysis.StatisticalKurtosis:StatisticalKurtosisPrimitive<tods.feature_analysis.StatisticalKurtosis>`
      * - `tods.feature_analysis.statistical_maximum`
        - :ref:`tods.feature_analysis.StatisticalMaximum:StatisticalMaximumPrimitive<tods.feature_analysis.StatisticalMaximum>`
      * - `tods.feature_analysis.statistical_mean`
        - :ref:`tods.feature_analysis.StatisticalMean:StatisticalMeanPrimitive<tods.feature_analysis.StatisticalMean>`
      * - `tods.feature_analysis.statistical_mean_abs`
        - :ref:`tods.feature_analysis.StatisticalMeanAbs:StatisticalMeanAbsPrimitive<tods.feature_analysis.StatisticalMeanAbs>`
      * - `tods.feature_analysis.statistical_mean_abs_temporal_derivative`
        - :ref:`tods.feature_analysis.StatisticalMeanAbsTemporalDerivative:StatisticalMeanAbsTemporalDerivativePrimitive<tods.feature_analysis.StatisticalMeanAbsTemporalDerivative>`
      * - `tods.feature_analysis.statistical_mean_temporal_derivative`
        - :ref:`tods.feature_analysis.StatisticalMeanTemporalDerivative:StatisticalMeanTemporalDerivativePrimitive<tods.feature_analysis.StatisticalMeanTemporalDerivative>`
      * - `tods.feature_analysis.statistical_median`
        - :ref:`tods.feature_analysis.StatisticalMedian:StatisticalMedianPrimitive<tods.feature_analysis.StatisticalMedian>`
      * - `tods.feature_analysis.statistical_median_abs_deviation`
        - :ref:`tods.feature_analysis.StatisticalMedianAbsoluteDeviation:StatisticalMedianAbsoluteDeviationPrimitive<tods.feature_analysis.StatisticalMedianAbsoluteDeviation>`
      * - `tods.feature_analysis.statistical_minimum`
        - :ref:`tods.feature_analysis.StatisticalMinimum:StatisticalMinimumPrimitive<tods.feature_analysis.StatisticalMinimum>`
      * - `tods.feature_analysis.statistical_skew`
        - :ref:`tods.feature_analysis.StatisticalSkew:StatisticalSkewPrimitive<tods.feature_analysis.StatisticalSkew>`
      * - `tods.feature_analysis.statistical_std`
        - :ref:`tods.feature_analysis.StatisticalStd:StatisticalStdPrimitive<tods.feature_analysis.StatisticalStd>`
      * - `tods.feature_analysis.statistical_var`
        - :ref:`tods.feature_analysis.StatisticalVar:StatisticalVarPrimitive<tods.feature_analysis.StatisticalVar>`
      * - `tods.feature_analysis.statistical_variation`
        - :ref:`tods.feature_analysis.StatisticalVariation:StatisticalVariationPrimitive<tods.feature_analysis.StatisticalVariation>`
      * - `tods.feature_analysis.statistical_vec_sum`
        - :ref:`tods.feature_analysis.StatisticalVecSum:StatisticalVecSumPrimitive<tods.feature_analysis.StatisticalVecSum>`
      * - `tods.feature_analysis.statistical_willison_amplitude`
        - :ref:`tods.feature_analysis.StatisticalWillisonAmplitude:StatisticalWillisonAmplitudePrimitive<tods.feature_analysis.StatisticalWillisonAmplitude>`
      * - `tods.feature_analysis.statistical_zero_crossing`
        - :ref:`tods.feature_analysis.StatisticalZeroCrossing:StatisticalZeroCrossingPrimitive<tods.feature_analysis.StatisticalZeroCrossing>`
      * - `tods.feature_analysis.trmf`
        - :ref:`tods.feature_analysis.TRMF:TRMFPrimitive<tods.feature_analysis.TRMF>`
      * - `tods.feature_analysis.wavelet_transform`
        - :ref:`tods.feature_analysis.WaveletTransform:WaveletTransformPrimitive<tods.feature_analysis.WaveletTransform>`




Detection Algorithms
--------------------

   .. list-table:: 
      :widths: 50 50
      :header-rows: 1

      * - Python Path
        - Primitives
      * - `tods.detection_algorithm.AutoRegODetector`
        - :ref:`tods.detection_algorithm.AutoRegODetect:AutoRegODetectorPrimitive<tods.detection_algorithm.AutoRegODetect>`
      * - `tods.detection_algorithm.dagmm`
        - :ref:`tods.detection_algorithm.DAGMM:DAGMMPrimitive<tods.detection_algorithm.DAGMM>`
      * - `tods.detection_algorithm.deeplog`
        - :ref:`tods.detection_algorithm.DeepLog:DeepLogPrimitive<tods.detection_algorithm.DeepLog>`
      * - `tods.detection_algorithm.Ensemble`
        - :ref:`tods.detection_algorithm.Ensemble:EnsemblePrimitive<tods.detection_algorithm.Ensemble>`
      * - `tods.detection_algorithm.KDiscordODetector`
        - :ref:`tods.detection_algorithm.KDiscordODetect:KDiscordODetectorPrimitive<tods.detection_algorithm.KDiscordODetect>`
      * - `tods.detection_algorithm.LSTMODetector`
        - :ref:`tods.detection_algorithm.LSTMODetect:LSTMODetectorPrimitive<tods.detection_algorithm.LSTMODetect>`
      * - `tods.detection_algorithm.matrix_profile`
        - :ref:`tods.detection_algorithm.MatrixProfile:MatrixProfilePrimitive<tods.detection_algorithm.MatrixProfile>`
      * - `tods.detection_algorithm.PCAODetector`
        - :ref:`tods.detection_algorithm.PCAODetect:PCAODetectorPrimitive<tods.detection_algorithm.PCAODetect>`
      * - `tods.detection_algorithm.pyod_abod`
        - :ref:`tods.detection_algorithm.PyodABOD:ABODPrimitive<tods.detection_algorithm.PyodABOD>`
      * - `tods.detection_algorithm.pyod_ae`
        - :ref:`tods.detection_algorithm.PyodAE:AutoEncoderPrimitive<tods.detection_algorithm.PyodAE>`
      * - `tods.detection_algorithm.pyod_cblof`
        - :ref:`tods.detection_algorithm.PyodCBLOF:CBLOFPrimitive<tods.detection_algorithm.PyodCBLOF>`
      * - `tods.detection_algorithm.pyod_cof`
        - :ref:`tods.detection_algorithm.PyodCOF:COFPrimitive<tods.detection_algorithm.PyodCOF>`
      * - `tods.detection_algorithm.pyod_hbos`
        - :ref:`tods.detection_algorithm.PyodHBOS:HBOSPrimitive<tods.detection_algorithm.PyodHBOS>`
      * - `tods.detection_algorithm.pyod_iforest`
        - :ref:`tods.detection_algorithm.PyodIsolationForest:IsolationForestPrimitive<tods.detection_algorithm.PyodIsolationForest>`
      * - `tods.detection_algorithm.pyod_knn`
        - :ref:`tods.detection_algorithm.PyodKNN:KNNPrimitive<tods.detection_algorithm.PyodKNN>`
      * - `tods.detection_algorithm.pyod_loda`
        - :ref:`tods.detection_algorithm.PyodLODA:LODAPrimitive<tods.detection_algorithm.PyodLODA>`
      * - `tods.detection_algorithm.pyod_lof`
        - :ref:`tods.detection_algorithm.PyodLOF:LOFPrimitive<tods.detection_algorithm.PyodLOF>`
      * - `tods.detection_algorithm.pyod_mogaal`
        - :ref:`tods.detection_algorithm.PyodMoGaal:Mo_GaalPrimitive<tods.detection_algorithm.PyodMoGaal>`
      * - `tods.detection_algorithm.pyod_ocsvm`
        - :ref:`tods.detection_algorithm.PyodOCSVM:OCSVMPrimitive<tods.detection_algorithm.PyodOCSVM>`
      * - `tods.detection_algorithm.pyod_sod`
        - :ref:`tods.detection_algorithm.PyodSOD:SODPrimitive<tods.detection_algorithm.PyodSOD>`
      * - `tods.detection_algorithm.pyod_sogaal`
        - :ref:`tods.detection_algorithm.PyodSoGaal:So_GaalPrimitive<tods.detection_algorithm.PyodSoGaal>`
      * - `tods.detection_algorithm.pyod_vae`
        - :ref:`tods.detection_algorithm.PyodVAE:VariationalAutoEncoderPrimitive<tods.detection_algorithm.PyodVAE>`
      * - `tods.detection_algorithm.system_wise_detection`
        - :ref:`tods.detection_algorithm.SystemWiseDetection:SystemWiseDetectionPrimitive<tods.detection_algorithm.SystemWiseDetection>`
      * - `tods.detection_algorithm.telemanom`
        - :ref:`tods.detection_algorithm.Telemanom:TelemanomPrimitive<tods.detection_algorithm.Telemanom>`
      
Reincforcement Module
---------------------

   .. list-table:: 
      :widths: 50 50
      :header-rows: 1

      * - Python Path
        - Primitives
      * - `tods.reinforcement.rule_filter`
        - :ref:`tods.reinforcement.RuleBasedFilter:RuleBasedFilter<tods.reinforcement.RuleBasedFilter>`


