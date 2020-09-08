from d3m.metadata.pipeline import Pipeline

from axolotl.algorithms.autokeras_integration.constants import OMIT_LAYERS, step_function
from axolotl.algorithms.autokeras_integration.steps import set_learner, set_prediction, set_data, \
    set_loss


def keras2pipeline(keras_model, batch_size=32):
    # Creating pipeline
    from tensorflow.python.keras.activations import softmax
    pipeline_description = Pipeline()

    pipeline_description.add_input(name='inputs')

    set_data(pipeline_description)
    set_loss(pipeline_description)

    offset = len(pipeline_description.steps)

    previous_layer_ids = get_previous_layer_ids(keras_model)

    layers = keras_model.layers

    step_id = 0
    layer_to_step_id = {}

    total_layer_num = len(layers)
    for i, layer in enumerate(layers):
        cls_name = get_layer_class_name(layer)
        if cls_name in OMIT_LAYERS:
            continue
        layer_id = get_layer_id(layer)
        if len(previous_layer_ids[layer_id]) > 0:
            layer.previous_layer_ids = tuple(
                layer_to_step_id[i] + offset for i in previous_layer_ids[layer_id]
            )
        else:
            layer.previous_layer_ids = [None]
        # Since JPL does not support Softmax Layer, we add the workaround to make use of softmax
        if i == total_layer_num - 2 and cls_name == 'Dense':
            layer.activation = softmax
        d3m_step = step_function[cls_name](step_id, layer)
        pipeline_description.add_step(d3m_step)
        layer_to_step_id[layer_id] = step_id
        step_id += 1

    set_learner(pipeline_description, batch_size)
    set_prediction(pipeline_description)
    pipeline_description.add_output(
        name='output predictions', data_reference=f"steps.{len(pipeline_description.steps) - 1}.produce")

    return pipeline_description


def get_previous_layer_ids(keras_model):
    from tensorflow.python.util import nest
    model = keras_model
    layers = model.layers

    previous_layer_ids = {}
    for layer in layers:
        layer_id = str(id(layer))
        previous_layer_ids[layer_id] = set()
        for i, node in enumerate(layer._inbound_nodes):
            node_key = layer.name + '_ib-' + str(i)
            if node_key in model._network_nodes:
                for inbound_layer in nest.flatten(node.inbound_layers):
                    inbound_cls_name = get_layer_class_name(inbound_layer)
                    inbound_layer_id = get_layer_id(inbound_layer)
                    if inbound_cls_name in OMIT_LAYERS:
                        previous_layer_ids[layer_id].update(previous_layer_ids[inbound_layer_id])
                    else:
                        previous_layer_ids[layer_id].add(inbound_layer_id)
    return previous_layer_ids


def get_layer_id(layer):
    return str(id(layer))


def get_layer_class_name(layer):
    return layer.__class__.__name__