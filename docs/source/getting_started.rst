Getting Started
===============

In this document, we provide some toy examples for getting started. All
the examples in this document and even more examples are available in
`examples <https://github.com/datamllab/tods/tree/master/examples>`__.

Outlier Detection with Autoencoder on NAB Dataset 
-------------------------------------------------
To perform the point-wise outlier detection on NAB dataset. We provide an example to construct
such pipeline description:

.. code:: python

    from d3m import index
    from d3m.metadata.base import ArgumentType
    from d3m.metadata.pipeline import Pipeline, PrimitiveStep

    # Creating pipeline
    pipeline_description = Pipeline()
    pipeline_description.add_input(name='inputs')

    # Step 0: dataset_to_dataframe
    step_0 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.dataset_to_dataframe.Common'))
    step_0.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
    step_0.add_output('produce')
    pipeline_description.add_step(step_0)

    # Step 1: column_parser
    step_1 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.column_parser.Common'))
    step_1.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
    step_1.add_output('produce')
    pipeline_description.add_step(step_1)

    # Step 2: extract_columns_by_semantic_types(attributes)
    step_2 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common'))
    step_2.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
    step_2.add_output('produce')
    step_2.add_hyperparameter(name='semantic_types', argument_type=ArgumentType.VALUE,
                            data=['https://metadata.datadrivendiscovery.org/types/Attribute'])
    pipeline_description.add_step(step_2)

    # Step 3: extract_columns_by_semantic_types(targets)
    step_3 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common'))
    step_3.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
    step_3.add_output('produce')
    step_3.add_hyperparameter(name='semantic_types', argument_type=ArgumentType.VALUE,
                                data=['https://metadata.datadrivendiscovery.org/types/TrueTarget'])
    pipeline_description.add_step(step_3)

    attributes = 'steps.2.produce'
    targets = 'steps.3.produce'

    # Step 4: processing
    step_4 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.timeseries_processing.transformation.axiswise_scaler'))
    step_4.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=attributes)
    step_4.add_output('produce')
    pipeline_description.add_step(step_4)

    # Step 5: algorithm
    step_5 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.detection_algorithm.pyod_ae'))
    step_5.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.4.produce')
    step_5.add_output('produce')
    pipeline_description.add_step(step_5)

     # Step 6: Predictions
    step_6 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.construct_predictions.Common'))
    step_6.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.5.produce')
    step_6.add_argument(name='reference', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
    step_6.add_output('produce')
    pipeline_description.add_step(step_6)

    # Final Output
    pipeline_description.add_output(name='output predictions', data_reference='steps.6.produce')

    # Output to json
    data = pipeline_description.to_json()
    with open('example_pipeline.json', 'w') as f:
        f.write(data)
        print(data)

Note that, in order to call each primitive during pipeline construction, one may find the index (python_path) of primitives available in
`entry_points.ini <https://github.com/datamllab/tods/blob/master/tods/resources/.entry_points.ini>`__.

The output description json file (example_pipeline.json) should look like something as follows:
::

    {
    "id": "e39bf406-06cf-4c76-88f0-8c8b4447e311", 
    "schema": "https://metadata.datadrivendiscovery.org/schemas/v0/pipeline.json", 
    "created": "2020-09-15T07:26:48.365447Z", 
    "inputs": [{"name": "inputs"}], 
    "outputs": [{"data": "steps.6.produce", "name": "output predictions"}], 
    "steps": [
        {"type": "PRIMITIVE", "primitive": {"id": "4b42ce1e-9b98-4a25-b68e-fad13311eb65", "version": "0.3.0", "python_path": "d3m.primitives.data_transformation.dataset_to_dataframe.Common", "name": "Extract a DataFrame from a Dataset", "digest": "a7f5a8f8b276f474c3b40b025d157541de898e4e02555cd8ef76fdeecfbed256"}, "arguments": {"inputs": {"type": "CONTAINER", "data": "inputs.0"}}, "outputs": [{"id": "produce"}]}, 
        {"type": "PRIMITIVE", "primitive": {"id": "d510cb7a-1782-4f51-b44c-58f0236e47c7", "version": "0.6.0", "python_path": "d3m.primitives.data_transformation.column_parser.Common", "name": "Parses strings into their types", "digest": "eccfd70ed359901a625dbde6de40d6bbb4e69d9796ee0ca3a302fd95195451ed"}, "arguments": {"inputs": {"type": "CONTAINER", "data": "steps.0.produce"}}, "outputs": [{"id": "produce"}]}, 
        {"type": "PRIMITIVE", "primitive": {"id": "4503a4c6-42f7-45a1-a1d4-ed69699cf5e1", "version": "0.4.0", "python_path": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common", "name": "Extracts columns by semantic type", "digest": "9f0303c354df6cec4df7bda0ebb46fb4f101c36ad9a4d1143b9b9c88004629aa"}, "arguments": {"inputs": {"type": "CONTAINER", "data": "steps.1.produce"}}, "outputs": [{"id": "produce"}], "hyperparams": {"semantic_types": {"type": "VALUE", "data": ["https://metadata.datadrivendiscovery.org/types/Attribute"]}}}, 
        {"type": "PRIMITIVE", "primitive": {"id": "4503a4c6-42f7-45a1-a1d4-ed69699cf5e1", "version": "0.4.0", "python_path": "d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common", "name": "Extracts columns by semantic type", "digest": "9f0303c354df6cec4df7bda0ebb46fb4f101c36ad9a4d1143b9b9c88004629aa"}, "arguments": {"inputs": {"type": "CONTAINER", "data": "steps.0.produce"}}, "outputs": [{"id": "produce"}], "hyperparams": {"semantic_types": {"type": "VALUE", "data": ["https://metadata.datadrivendiscovery.org/types/TrueTarget"]}}}, 
        {"type": "PRIMITIVE", "primitive": {"id": "642de2e7-5590-3cab-9266-2a53c326c461", "version": "0.0.1", "python_path": "d3m.primitives.tods.timeseries_processing.transformation.axiswise_scaler", "name": "Axis_wise_scale"}, "arguments": {"inputs": {"type": "CONTAINER", "data": "steps.2.produce"}}, "outputs": [{"id": "produce"}]}, 
        {"type": "PRIMITIVE", "primitive": {"id": "67e7fcdf-d645-3417-9aa4-85cd369487d9", "version": "0.0.1", "python_path": "d3m.primitives.tods.detection_algorithm.pyod_ae", "name": "TODS.anomaly_detection_primitives.AutoEncoder"}, "arguments": {"inputs": {"type": "CONTAINER", "data": "steps.4.produce"}}, "outputs": [{"id": "produce"}]}, 
        {"type": "PRIMITIVE", "primitive": {"id": "8d38b340-f83f-4877-baaa-162f8e551736", "version": "0.3.0", "python_path": "d3m.primitives.data_transformation.construct_predictions.Common", "name": "Construct pipeline predictions output", "digest": "6de56912a3f84bbbcc0d1f7ffe646044209120e45bbb21a137236d00fed948e9"}, "arguments": {"inputs": {"type": "CONTAINER", "data": "steps.5.produce"}, "reference": {"type": "CONTAINER", "data": "steps.1.produce"}}, "outputs": [{"id": "produce"}]}], 
    "digest": "8c6a37e7ac9ef1b302810e56dffa43c3415826ab756ef6917d76dd8ee63d38fc"
    }

With the pre-built pipeline description file, we can then feed the NAB data (twitter_IBM) and specify the desired evaluation metric with the path of pipeline description file with 
`run_pipeline.py <https://github.com/datamllab/tods/blob/master/examples/axolotl_interface/run_pipeline.py>`__.
:: 
    python examples/run_pipeline.py --pipeline_path example_pipeline.json --table_path datasets/NAB/realTweets/labeled_Twitter_volume_IBM.csv --metric F1_MACRO --target_index 2

.. code:: python
