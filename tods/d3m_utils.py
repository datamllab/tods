def build_pipeline(config):
    """Build a pipline based on the config
    """
    from d3m import index
    from d3m.metadata.base import ArgumentType
    from d3m.metadata.pipeline import Pipeline, PrimitiveStep
    algorithm = config.pop('algorithm', "pyod_ae")
    process = config.pop('processing', "statistical_maximum")
#    transformation = config.pop('transformation', [])
#    transformation_methods = [transformation[i][0] for i in range(len(transformation))]
#    augmentation = config.pop('augmentation', [])
#    augmentation_methods = [augmentation[i][0] for i in range(len(augmentation))]

    # Read augmentation hyperparameters
#    augmentation_configs = []
#    for i in range(len(augmentation)):
#        if len(augmentation[i]) > 1:
#            augmentation_configs.append(augmentation[i][1])
#        else:
#            augmentation_configs.append(None)
#    multi_aug = config.pop('multi_aug', None)

    # Read transformation hyperparameters
#    transformation_configs = [transformation[i][1] for i in range(len(transformation))]

    # Creating pipeline
    pipeline_description = Pipeline()
    pipeline_description.add_input(name='inputs')
    
    # Step 0: dataset_to_dataframe
    step_0 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.dataset_to_dataframe'))
    step_0.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
    step_0.add_output('produce')
    pipeline_description.add_step(step_0)

    # Step 1: column_parser
    step_1 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.column_parser'))
    step_1.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
    step_1.add_output('produce')
    pipeline_description.add_step(step_1)

    # Step 2: extract_columns_by_semantic_types(attributes)
    step_2 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.extract_columns_by_semantic_types'))
    step_2.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
    step_2.add_output('produce')
    step_2.add_hyperparameter(name='semantic_types', argument_type=ArgumentType.VALUE,
                                  data=['https://metadata.datadrivendiscovery.org/types/Attribute'])
    pipeline_description.add_step(step_2)

    # Step 3: extract_columns_by_semantic_types(targets)
    step_3 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.extract_columns_by_semantic_types'))
    step_3.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
    step_3.add_output('produce')
    step_3.add_hyperparameter(name='semantic_types', argument_type=ArgumentType.VALUE,
                                data=['https://metadata.datadrivendiscovery.org/types/TrueTarget'])
    pipeline_description.add_step(step_3)

    attributes = 'steps.2.produce'
    targets = 'steps.3.produce'

    # Step 4: processing
    process_python_path = 'd3m.primitives.tods.feature_analysis.' + process
    step_4 = PrimitiveStep(primitive=index.get_primitive(process_python_path))
    #step_4 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.timeseries_processing.transformation.axiswise_scaler'))
    
    #step_4 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.feature_analysis.statistical_minimum'))
    step_4.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference=attributes)
    step_4.add_output('produce')
    pipeline_description.add_step(step_4)

    # Step 5: algorithm`
    alg_python_path = 'd3m.primitives.tods.detection_algorithm.' + algorithm
    step_5 = PrimitiveStep(primitive=index.get_primitive(alg_python_path))
    step_5.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.4.produce')
    for key,value in config.items():
        step_5.add_hyperparameter(name= key, argument_type=ArgumentType.VALUE, data= value)
    step_5.add_output('produce')
    pipeline_description.add_step(step_5)

    # Step 6: Predictions
    step_6 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.tods.data_processing.construct_predictions'))
    step_6.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.5.produce')
    step_6.add_argument(name='reference', argument_type=ArgumentType.CONTAINER, data_reference='steps.1.produce')
    step_6.add_output('produce')
    pipeline_description.add_step(step_6)

    # Final Output
    pipeline_description.add_output(name='output predictions', data_reference='steps.6.produce')

    # Output to json
#    data = pipeline_description.to_json()
#    with open('autoencoder_pipeline.json', 'w') as f:
#        f.write(data)
#        print(data)


    return pipeline_description