from kerastuner import Objective
from kerastuner.engine import trial as trial_lib
from kerastuner.tuners.randomsearch import RandomSearchOracle as KerasRandomSearchOracle

from axolotl.algorithms.tuners.oracle import infer_metric_direction, random_values


class RandomSearchOracle(KerasRandomSearchOracle):
    """
    Random search oracle.
    """

    def __init__(self,
                 objective,
                 max_trials,
                 seed=None,
                 hyperparameters=None,
                 allow_new_entries=True,
                 tune_new_entries=True):
        direction = infer_metric_direction(objective)
        objective = Objective(name=objective, direction=direction)
        super(RandomSearchOracle, self).__init__(
            objective=objective,
            max_trials=max_trials,
            seed=seed,
            hyperparameters=hyperparameters,
            tune_new_entries=tune_new_entries,
            allow_new_entries=allow_new_entries)

    def _populate_space(self, _):
        values = self._random_values()
        if values is None:
            return {'status': trial_lib.TrialStatus.STOPPED,
                    'values': None}
        return {'status': trial_lib.TrialStatus.RUNNING,
                'values': values}

    def _random_values(self):
        """Fills the hyperparameter space with random values.

        Returns:
            A dictionary mapping parameter names to suggested values.
        """

        values, seed_state = random_values(hyperparameters=self.hyperparameters,
                      seed_state=self._seed_state,
                      tried_so_far=self._tried_so_far,
                      max_collisions=self._max_collisions,
                      )
        self._seed_state = seed_state
        return values

    def _save_trial(self, trial):
        pass

    def get_state(self):
        # `self.trials` are saved in their own, Oracle-agnostic files.
        # Just save the IDs for ongoing trials, since these are in `trials`.
        state = {}
        state['ongoing_trials'] = {
            tuner_id: trial.trial_id
            for tuner_id, trial in self.ongoing_trials.items()}
        # Hyperparameters are part of the state because they can be added to
        # during the course of the search.
        state['hyperparameters'] = str(self.hyperparameters.get_config())
        return state
