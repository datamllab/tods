# Minimal Metadata Datasets (version 4.0.0)

Begining with D3M Winter Workshop 2020, there is a push in the program to reduce the amount of manually curated metadata information that is provided in the datasets. The motivation is that as the D3M systems are transitioned to partner environments, it's unrealistic to expect that the datasets will be fully curated in D3M format. Therefore, the systems need to reduce their reliance on manually hand-coded metadata. This is a step in that direction.

D3M core package clearly requires some metadata to work. Some of that metadata information can be inferred and others will be difficult to infer automatically and will have to be provided manually. For those that are provided manually, this page lists what metadata elements can be reliably expected in minimal metadata datasets, and the rest of it will be optional (i.e., not provided in most cases).

**Note 1:** The structure of the D3M datasets will remain intact.

**Note 2:** There will be not changes to the problem metadata that is provided, i.e., the problemDoc.json files will not be affected by this change.

**Note 3:** All the changes in minimal metadata datasets (for now) will be w.r.t to the data schema, i.e, only datasetDoc.json files will be affected. For now, only the column metadata like column types and roles (with a few exceptions) have been removed when transitioning to minimal metadata format. The exceptions are listed in the table below. The resourse metadata will remain intact.

**Note4** The master branch (seed datasets) moving forward will by default contain datasets in min metadata format

**Note 5:** The original full metadata counterparts of the min metadata seeds are archived in a separate directory: training_datasets/seed_datasets_archive/

The following table lists the column metadata that can be extected per problem type.


| Problem Type                                               | Column information retained in minimal metadata |
|------------------------------------------------------------|-------------------------------------------------|
| (classification, binary, tabular)                          | (index)                                         |
| (classification, multiClass, tabular)                      | (index)                                         |
| (classification, multiLabel, tabular)                      | (index, multiIndex)                             |
| (classification, binary, lupi, tabular)                    | (index)                                         |
| (classification, binary, semiSupervised, tabular)          | (index)                                         |
| (classification, multiClass, semiSupervised, tabular)      | (index)                                         |
| (regression, univariate, tabular)                          | (index)                                         |
| (regression, multivariate, tabular)                        | (index)                                         |
| (classification, binary, tabular, relational)              | (index, refersTo)                               |
| (classification, multiClass, tabular, relational)          | (index, refersTo)                               |
| (regression, univariate, tabular, relational)              | (index, refersTo)                               |
| (regression, multivariate, tabular, relational)            | (index, refersTo)                               |
| (classification, binary, text)                             | (index, refersTo)                               |
| (classifciation, multiClass, text)                         | (index, refersTo)                               |
| (classification, binary, text, relational)                 | (index, refersTo)                               |
| ('classification', 'multiClass', 'video')                  | (index, refersTo)                               |
| ('classification', 'multiClass', 'image')                  | (index, refersTo)                               |
| (regression, univariate, image)                            | (index, refersTo)                               |
| (regression, multivariate, image)                          | (index, refersTo)                               |
| ('classification', 'multiClass', 'audio')                  | (index, refersTo)                               |
| (objectDetection, image)                                   | (index, refersTo, multiIndex)                   |
| (graph, vertexClassification,  multiClass)                 | (index, refersTo)                               |
| (graph, linkPrediction)                                    | (index, refersTo)                               |
| (graph, graphMatching)                                     | (index, refersTo)                               |
| (graph, communityDetection, nonOverlapping)                | (index, refersTo)                               |
| (graph, linkPrediction, timeSeries)                        | (index, refersTo, timeIndicator)                |
| (timeseries, forecasting, tabular)                         | (index, timeIndicator)                          |
| (timeseries, forecasting, tabular, grouped)                | (index, timeIndicator, suggestedGroupingKey)    |
| (classification, multiClass, timeseries)                   | (index, refersTo)                               |
| (classification, multiClass, timeseries)                   | (index, refersTo)                               |
| (classification, binary, timeseries, tabular, grouped)     | (index, refersTo, suggestedGroupingKey)         |
| (classification, multiClass, timeseries, tabular, grouped) | (index, refersTo, suggestedGroupingKey)         |