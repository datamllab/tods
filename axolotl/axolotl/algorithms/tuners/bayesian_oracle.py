import numpy as np

from scipy import optimize as scipy_optimize
from sklearn import exceptions

from d3m.metadata import hyperparams
from kerastuner import Objective
from kerastuner.tuners.bayesian import BayesianOptimizationOracle as KerasBayesian
from kerastuner.engine import trial as trial_lib

from axolotl.algorithms.tuners.hyperparameters import HyperParameters, \
    value_to_cumulative_prob, cumulative_prob_to_value
from axolotl.algorithms.tuners.oracle import infer_metric_direction, random_values, patch_invalid_hyperamaeters


class BayesianOptimizationOracle(KerasBayesian):
    """
    Bayesian optimization oracle.
    """

    def __init__(self,
                 objective,
                 max_trials,
                 num_initial_points=None,
                 alpha=1e-4,
                 beta=2.6,
                 seed=None,
                 hyperparameters=None,
                 allow_new_entries=True,
                 tune_new_entries=True):
        direction = infer_metric_direction(objective)
        objective = Objective(name=objective, direction=direction)
        super(BayesianOptimizationOracle, self).__init__(
            objective=objective,
            max_trials=max_trials,
            num_initial_points=num_initial_points,
            alpha=alpha,
            beta=beta,
            seed=seed,
            hyperparameters=hyperparameters,
            allow_new_entries=allow_new_entries,
            tune_new_entries=tune_new_entries,
        )
        self.num_complete_trials = 0
        self.sorted_candidates = []

    # TODO how to save a trial
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

        state.update({
            'num_initial_points': self.num_initial_points,
            'alpha': self.alpha,
            'beta': self.beta,
        })
        return state

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

    def _nonfixed_space(self):
        return [hp for hp in self.hyperparameters.space
                if not isinstance(hp, hyperparams.Constant)]

    def _vector_to_values(self, vector):
        hps = HyperParameters()
        vector_index = 0
        for hp in self.hyperparameters.space:
            hps.merge([hp])
            if isinstance(hp, hyperparams.Constant):
                value = hp.get_default()
            else:
                prob = vector[vector_index]
                vector_index += 1
                value = cumulative_prob_to_value(prob, hp)

            if hps.is_active(hp):
                hps.values[hp.name] = value
        patch_invalid_hyperamaeters(hps)
        return hps.values

    def _vectorize_trials(self):
        x = []
        y = []
        ongoing_trials = {t for t in self.ongoing_trials.values()}
        for trial in self.trials.values():
            # Create a vector representation of each Trial's hyperparameters.
            trial_hps = trial.hyperparameters
            vector = []
            for hp in self._nonfixed_space():
                # For hyperparameters not present in the trial (either added after
                # the trial or inactive in the trial), set to default value.
                if trial_hps.is_active(hp):
                    trial_value = trial_hps.values[hp.name]
                else:
                    trial_value = hp.default

                # Embed an HP value into the continuous space [0, 1].
                prob = value_to_cumulative_prob(trial_value, hp)
                vector.append(prob)

            if trial in ongoing_trials:
                # "Hallucinate" the results of ongoing trials. This ensures that
                # repeat trials are not selected when running distributed.
                x_h = np.array(vector).reshape((1, -1))
                y_h_mean, y_h_std = self.gpr.predict(x_h, return_std=True)
                # Give a pessimistic estimate of the ongoing trial.
                score = y_h_mean[0] + y_h_std[0]
            elif trial.status == 'COMPLETED':
                score = trial.score
                # Always frame the optimization as a minimization for scipy.minimize.
                if self.objective.direction == 'max':
                    score = -1*score
            else:
                continue

            x.append(vector)
            y.append(score)

        x = np.array(x)
        y = np.array(y)
        return x, y

    def _populate_space(self, trial_id):
        # Generate enough samples before training Gaussian process.
        completed_trials = [t for t in self.trials.values()
                            if t.status == 'COMPLETED']

        # Use 3 times the dimensionality of the space as the default number of
        # random points.
        dimensions = len(self.hyperparameters.space)
        num_initial_points = self.num_initial_points or 3 * dimensions
        if len(completed_trials) < num_initial_points:
            return self._random_populate_space()

        if self.num_complete_trials == len(completed_trials) and len(self.sorted_candidates) > 0:
            optimal_x = self.sorted_candidates.pop().x
            values = self._vector_to_values(optimal_x)
            return {'status': trial_lib.TrialStatus.RUNNING,
                    'values': values}

        # track the number of complete trials
        self.num_complete_trials = len(completed_trials)

        # Fit a GPR to the completed trials and return the predicted optimum values.
        x, y = self._vectorize_trials()
        try:
            self.gpr.fit(x, y)
        except exceptions.ConvergenceWarning:
            # If convergence of the GPR fails, create a random trial.
            return self._random_populate_space()

        def _upper_confidence_bound(x):
            x = x.reshape(1, -1)
            mu, sigma = self.gpr.predict(x, return_std=True)
            return mu - self.beta * sigma

        num_restarts = 50
        bounds = self._get_hp_bounds()
        x_seeds = self._random_state.uniform(bounds[:, 0], bounds[:, 1],
                                             size=(num_restarts, bounds.shape[0]))
        candidates = [
            scipy_optimize.minimize(_upper_confidence_bound,
                                    x0=x_try,
                                    bounds=bounds,
                                    method='L-BFGS-B')
            for x_try in x_seeds
        ]

        self.sorted_candidates = sorted(candidates, key=lambda x: x.fun[0], reverse=True)
        optimal_x = self.sorted_candidates.pop().x

        values = self._vector_to_values(optimal_x)
        return {'status': trial_lib.TrialStatus.RUNNING,
                'values': values}
