import os

import hashlib
import random

from d3m import utils as d3m_utils
from d3m.metadata import problem as problem_module
from axolotl.algorithms.tuners.hyperparameters import HyperParameters, PIPELINE_CHOICE

_MAX_METRICS = {
    problem_module.PerformanceMetric.ACCURACY,
    problem_module.PerformanceMetric.PRECISION,
    problem_module.PerformanceMetric.RECALL,
    problem_module.PerformanceMetric.F1,
    problem_module.PerformanceMetric.F1_MICRO,
    problem_module.PerformanceMetric.F1_MACRO,
    problem_module.PerformanceMetric.ROC_AUC,
    problem_module.PerformanceMetric.JACCARD_SIMILARITY_SCORE,
    problem_module.PerformanceMetric.NORMALIZED_MUTUAL_INFORMATION,  # not sure
    problem_module.PerformanceMetric.OBJECT_DETECTION_AVERAGE_PRECISION,
}
_MAX_METRICS_NAME = {s.name for s in _MAX_METRICS}


_MIN_METRICS = {
    problem_module.PerformanceMetric.MEAN_ABSOLUTE_ERROR,
    problem_module.PerformanceMetric.MEAN_SQUARED_ERROR,
    problem_module.PerformanceMetric.ROOT_MEAN_SQUARED_ERROR,
    problem_module.PerformanceMetric.R_SQUARED,
}
_MIN_METRICS_NAME = {s.name for s in _MIN_METRICS}


def infer_metric_direction(metric):
    # Handle str input and get canonical object.
    if isinstance(metric, str):
        metric_name = metric
        if metric_name in _MIN_METRICS_NAME:
            return 'min'
        elif metric_name in _MAX_METRICS_NAME:
            return 'max'


def random_values(hyperparameters, seed_state, tried_so_far, max_collisions):
    collisions = 0
    while 1:
        # Generate a set of random values.
        hps = HyperParameters()
        with d3m_utils.silence():
            for hp in hyperparameters.space:
                hps.merge([hp])
                if hps.is_active(hp):  # Only active params in `values`.
                    hps.values[hp.name] = hp.random_sample(seed_state)
                    seed_state += 1
        # Pick out the invalid hyper-parameters
        patch_invalid_hyperamaeters(hps)

        values = hps.values
        # Keep trying until the set of values is unique,
        # or until we exit due to too many collisions.
        values_hash = compute_values_hash(values)
        if values_hash in tried_so_far:
            collisions += 1
            if collisions > max_collisions:
                return None
            continue
        tried_so_far.add(values_hash)
        break
    return values, seed_state


def compute_values_hash(values):
    keys = sorted(values.keys())
    s = ''.join(str(k) + '=' + str(values[k]) for k in keys)
    return hashlib.sha256(s.encode('utf-8')).hexdigest()[:32]


def patch_invalid_hyperamaeters(hps):
    values = hps.values
    for full_name in values:
        if full_name == PIPELINE_CHOICE:
            continue
        hp_val = values[full_name]
        step, primitive_name, hp_name = hps.get_name_parts(full_name)
        if primitive_name == 'd3m.primitives.classification.svc.SKlearn' \
                and hp_name == 'decision_function_shape' and hp_val == 'ovo':
            # break_ties must be False if decision-function_shape == 'ovo'
            break_ties = os.path.join(step, primitive_name, 'break_ties')
            values[break_ties] = False
        if primitive_name == 'd3m.primitives.classification.logistic_regression.SKlearn':
            # elasticnet' penalty, solver must be'saga'
            if hp_name == 'penalty' and hp_val == 'elasticnet':
                solver = os.path.join(step, primitive_name, 'solver')
                values[solver] = 'saga'
            if hp_name == 'solver':
                penalty = os.path.join(step, primitive_name, 'penalty')
                # liblinear only supports 'ovr' multi_class and [l2, l1] penalty
                if hp_val == 'liblinear':
                    multi_class = os.path.join(step, primitive_name, 'multi_class')
                    values[multi_class] = 'ovr'
                    values[penalty] = random.choice(['l2', 'l1'])
                # ['lbfgs', 'newton-cg', 'sag'] only support [l2, none] penalty
                elif hp_val in ['lbfgs', 'newton-cg', 'sag']:
                    values[penalty] = random.choice(['l2', 'none'])
