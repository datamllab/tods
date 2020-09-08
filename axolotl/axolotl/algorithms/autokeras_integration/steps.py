from d3m.metadata.base import ArgumentType
from d3m.metadata.pipeline import PrimitiveStep

import d3m.primitives.data_preprocessing.image_reader
import d3m.primitives.data_transformation.denormalize
import d3m.primitives.data_transformation.dataset_to_dataframe
import d3m.primitives.data_transformation.construct_predictions
import d3m.primitives.data_transformation.extract_columns_by_semantic_types
import d3m.primitives.data_transformation.replace_semantic_types

import d3m.primitives.loss_function.categorical_crossentropy
import d3m.primitives.loss_function.categorical_accuracy

import d3m.primitives.learner.model
import d3m.primitives.data_wrangling.batching

LOSS_SETUP_IDX = IP_STEP = OP_STEP = READER_STEP = -1
BATCH_SIZE = 40


def set_data(pipeline_description):
    global IP_STEP, OP_STEP, READER_STEP

    # denormalize
    denorm_step_idx = 0
    step = PrimitiveStep(
        primitive_description=d3m.primitives.data_transformation.denormalize.Common.metadata.query())
    step.add_argument(
        name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
    step.add_output('produce')
    pipeline_description.add_step(step)

    # dataset_to_dataframe
    dataset_to_dataframe_step_idx = len(pipeline_description.steps)
    step = PrimitiveStep(
        primitive_description=d3m.primitives.data_transformation.dataset_to_dataframe.Common.metadata.query())
    step.add_argument(
        name='inputs', argument_type=ArgumentType.CONTAINER,
        data_reference='steps.{}.produce'.format(denorm_step_idx))
    step.add_output('produce')
    pipeline_description.add_step(step)

    # extract targets
    extract_step_idx = len(pipeline_description.steps)
    extract_targets = PrimitiveStep(
        d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common.metadata.query())
    extract_targets.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER,
                                 data_reference='steps.{}.produce'.format(dataset_to_dataframe_step_idx))
    extract_targets.add_hyperparameter(name='semantic_types', argument_type=ArgumentType.VALUE,
                                       data=['https://metadata.datadrivendiscovery.org/types/TrueTarget'])
    extract_targets.add_output('produce')
    pipeline_description.add_step(extract_targets)

    # replace semantic types
    # Need to be used for CIFAR-10
    replace_step_idx = len(pipeline_description.steps)
    replace_semantic = PrimitiveStep(
        d3m.primitives.data_transformation.replace_semantic_types.Common.metadata.query())
    replace_semantic.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER,
                                  data_reference=f'steps.{extract_step_idx}.produce')
    replace_semantic.add_hyperparameter(name='to_semantic_types', argument_type=ArgumentType.VALUE,
                                        data=['https://metadata.datadrivendiscovery.org/types/SuggestedTarget',
                                              'https://metadata.datadrivendiscovery.org/types/TrueTarget'])
    replace_semantic.add_hyperparameter(name='from_semantic_types', argument_type=ArgumentType.VALUE,
                                        data=['https://metadata.datadrivendiscovery.org/types/TrueTarget'])
    replace_semantic.add_output('produce')
    pipeline_description.add_step(replace_semantic)

    # image reader
    reader_step_idx = len(pipeline_description.steps)
    reader = PrimitiveStep(
        primitive_description=d3m.primitives.data_preprocessing.image_reader.Common.metadata.query())
    reader.add_hyperparameter(name='return_result', argument_type=ArgumentType.VALUE, data='new')
    pipeline_description.add_step(reader)

    IP_STEP, OP_STEP, READER_STEP = dataset_to_dataframe_step_idx, replace_step_idx, reader_step_idx


def set_loss(pipeline_description):
    global LOSS_SETUP_IDX

    LOSS_SETUP_IDX = len(pipeline_description.steps)
    step = PrimitiveStep(
        primitive_description=d3m.primitives.loss_function.categorical_crossentropy.KerasWrap.metadata.query())
    pipeline_description.add_step(step)


def set_learner(pipeline_description, batch_size=BATCH_SIZE):
    learner_idx = len(pipeline_description.steps)
    step = PrimitiveStep(primitive_description=d3m.primitives.learner.model.KerasWrap.metadata.query())
    step.add_hyperparameter(name='loss', argument_type=ArgumentType.PRIMITIVE, data=LOSS_SETUP_IDX)
    step.add_hyperparameter(name='model_type', argument_type=ArgumentType.VALUE, data='classification')
    step.add_hyperparameter(name='network_last_layer', argument_type=ArgumentType.PRIMITIVE,
                            data=learner_idx - 1)
    step.add_hyperparameter(name='return_result', argument_type=ArgumentType.VALUE, data='replace')
    lr = 0.0001
    adam_hypers = d3m.primitives.learner.model.KerasWrap.metadata.get_hyperparams().defaults(path='optimizer.Adam')
    adam_hypers = adam_hypers.replace({'lr': lr})
    step.add_hyperparameter(name='optimizer', argument_type=ArgumentType.VALUE, data=adam_hypers)
    pipeline_description.add_step(step)

    bz_loader = PrimitiveStep(primitive_description=d3m.primitives.data_wrangling.batching.TAMU.metadata.query())
    bz_loader.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER,
                           data_reference=f'steps.{IP_STEP}.produce')
    bz_loader.add_argument(name='outputs', argument_type=ArgumentType.CONTAINER,
                           data_reference='steps.{}.produce'.format(OP_STEP))
    bz_loader.add_hyperparameter(name='primitive_reader', argument_type=ArgumentType.PRIMITIVE, data=READER_STEP)
    bz_loader.add_hyperparameter(name='primitive_learner', argument_type=ArgumentType.PRIMITIVE, data=learner_idx)
    bz_loader.add_hyperparameter(name='batch_size', argument_type=ArgumentType.VALUE, data=batch_size)
    bz_loader.add_hyperparameter(name='sampling_method', argument_type=ArgumentType.VALUE, data='random')
    bz_loader.add_output('produce')

    pipeline_description.add_step(bz_loader)


def set_prediction(pipeline_description):
    pred = PrimitiveStep(
        primitive_description=d3m.primitives.data_transformation.construct_predictions.Common.metadata.query())
    pred.add_argument(
        name='inputs', argument_type=ArgumentType.CONTAINER,
        data_reference=f"steps.{len(pipeline_description.steps) - 1}.produce"
    )
    pred.add_argument(name='reference', argument_type=ArgumentType.CONTAINER,
                        data_reference='steps.{}.produce'.format(IP_STEP))
    pred.add_output('produce')
    pipeline_description.add_step(pred)
