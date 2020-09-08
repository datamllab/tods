import json
import math
from scipy.stats import norm

from d3m import utils as d3m_utils
from d3m.metadata import hyperparams
from d3m.metadata.hyperparams import HyperparameterMeta
from kerastuner.engine.hyperparameters import HyperParameters as KerasHyperparams

PIPELINE_CHOICE = 'pipeline_choice'


def GET_CONFIG(param_val):
    config = param_val.to_simple_structure()
    config['p'] = param_val
    if isinstance(param_val, hyperparams.SortedList) or isinstance(param_val, hyperparams.SortedSet):
        config['is_configuration'] = param_val.is_configuration
    return config


class HyperParameters(KerasHyperparams):
    def get_config(self):
        return {
            'space': [{'class_name': p.__class__.__name__,
                       'config': GET_CONFIG(p)}
                      for p in self.space],
            'values': dict((k, v) for (k, v) in self.values.items()),
        }

    def retrieve(self, name, val, parent_name=None, parent_values=None):
        """Gets or creates a `HyperParameter`."""
        config = GET_CONFIG(val)
        hp = config['p']
        hp.name = self._get_name(name)
        hp.default = get_val(hp.get_default)()
        hp.random_sample = get_val(hp.sample)
        hp.conditions = [c for c in self._conditions]
        with self._maybe_conditional_scope(parent_name, parent_values):
            return self._retrieve(hp)

    def _register(self, hp):
        """Registers a `HyperParameter` into this container."""
        self._hps[hp.name].append(hp)
        self._space.append(hp)
        value = hp.default
        if self._conditions_are_active(hp.conditions):
            self.values[hp.name] = value
            return value
        return None

    @classmethod
    def from_config(cls, config):
        hps = cls()
        for p in config['space']:
            p = p['config']['p']
            hps._hps[p.name].append(p)
            hps._space.append(p)
        hps.values = dict((k, v) for (k, v) in config['values'].items())
        return hps

    def copy(self):
        return HyperParameters.from_config(self.get_config())

    def __repr__(self):
        return self.to_json()

    def to_json(self):
        return json.dumps(self.__dict__, default=serialize)

    def _get_name_parts(self, full_name):
        """Splits `full_name` into its scopes and leaf name."""
        str_parts = full_name.split('/')
        parts = []

        for part in str_parts:
            if '=' in part:
                parent_name, parent_values = part.split('=')
                parent_values = parent_values.split(',')
                parts.append({'parent_name': parent_name,
                              'parent_values': parent_values})
            else:
                parts.append(part)

        return parts

    def get_pipeline_id(self):
        pipeline_id = self.values[PIPELINE_CHOICE]
        return pipeline_id

    def get_name_parts(self, full_name):
        step, primitive_name, hp_name = self._get_name_parts(full_name)
        return step, primitive_name, hp_name


def get_val(func):
    def wrapper(*args, **kwargs):
        val = func(*args, **kwargs)
        return val['choice'] if isinstance(val, dict) and 'choice' in val else val
    return wrapper


def serialize(obj):
    if isinstance(obj, HyperparameterMeta):
        return obj.__dict__


def value_to_cumulative_prob(value, hp):
    """Convert a hyperparameter value to [0, 1]."""
    if isinstance(hp, hyperparams.Constant):
        return 0.5
    if isinstance(hp, hyperparams.UniformBool):
        # Center the value in its probability bucket.
        if value:
            return 0.75
        return 0.25
    elif isinstance(hp, (hyperparams.Choice, hyperparams.Enumeration, hyperparams.Union)):
        if isinstance(hp, hyperparams.Choice):
            choices = hp.choices
            index = list(choices.keys()).index(value)
        elif isinstance(hp, hyperparams.Union):
            choices = hp.configuration.keys()
            for index, val_type in enumerate(hp.configuration.values()):
                if isinstance(value, val_type.structural_type):
                    break
        else:
            choices = hp.values
            index = choices.index(value)
        ele_prob = 1 / len(choices)
        # Center the value in its probability bucket.
        return (index + 0.5) * ele_prob
    elif isinstance(hp, (hyperparams.UniformInt, hyperparams.Uniform, hyperparams.Bounded)):
        lower, upper = hp.lower, hp.upper
        if lower is None or upper is None:
            return 0.5
        return (value - lower) / (upper - lower)
    elif isinstance(hp, hyperparams.LogUniform):
        lower, upper = hp.lower, hp.upper
        if lower is None or upper is None:
            return 0.5
        return (math.log(value / lower) /
                math.log(upper / lower))
    elif isinstance(hp, (hyperparams.Normal, hyperparams.LogNormal)):
        return norm.cdf(value, hp.mu, hp.sigma)
    else:
        raise ValueError('Unrecognized HyperParameter type: {}'.format(hp))


def cumulative_prob_to_value(prob, hp):
    """Convert a value from [0, 1] to a hyperparameter value."""
    if isinstance(hp, hyperparams.Constant):
        return hp.get_default()
    elif isinstance(hp, hyperparams.UniformBool):
        return bool(prob >= 0.5)
    elif isinstance(hp, (hyperparams.Choice, hyperparams.Enumeration, hyperparams.Union)):
        if isinstance(hp, hyperparams.Choice):
            choices = list(hp.choices.keys())
        elif isinstance(hp, hyperparams.Union):
            choices = list(hp.configuration.keys())
        else:
            choices = hp.values
        ele_prob = 1 / len(choices)
        index = int(math.floor(prob / ele_prob))
        # Can happen when `prob` is very close to 1.
        if index == len(choices):
            index = index - 1
        if isinstance(hp, hyperparams.Union):
            key = choices[index]
            with d3m_utils.silence():
                val = hp.configuration[key].sample()
            return val
        return choices[index]
    elif isinstance(hp, (hyperparams.UniformInt, hyperparams.Uniform, hyperparams.Bounded)):
        import sys
        epsilon = sys.float_info.epsilon
        lower, upper = hp.lower, hp.upper
        if lower is None or upper is None:
            return hp.get_default()
        value = prob * (upper - lower) + lower
        if hp.structural_type == int:
            return int(value)
        if value == lower and not hp.lower_inclusive:
            return value + epsilon
        if value == upper and not hp.upper_inclusive:
            return value - epsilon
        return value
    elif isinstance(hp, hyperparams.LogUniform):
        lower, upper = hp.lower, hp.upper
        if lower is None or upper is None:
            return hp.get_default()
        value = lower * math.pow(upper / lower, prob)
        return value
    elif isinstance(hp, (hyperparams.Normal, hyperparams.LogNormal)):
        return norm.ppf(prob, loc=hp.mu, scale=hp.sigma)
    else:
        raise ValueError('Unrecognized HyperParameter type: {}'.format(hp))
