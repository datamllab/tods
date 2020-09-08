import enum

from axolotl.algorithms.tuners.bayesian_oracle import BayesianOptimizationOracle
from axolotl.algorithms.tuners.tunable_base import TunableBase


class BayesianSearch(TunableBase):
    def __init__(self, problem_description, backend, primitives_blocklist=None,
                 max_trials=10000, directory='.', num_initial_points=None, num_eval_trials=None):
        super(BayesianSearch, self).__init__(problem_description, backend,
                                           primitives_blocklist=primitives_blocklist, num_eval_trials=num_eval_trials)
        self.directory = directory
        self.project_name = 'random_search'

        self.objective = self.problem_description['problem']['performance_metrics'][0]['metric']
        if isinstance(self.objective, enum.Enum):
            self.objective = self.objective.name

        self.oracle = BayesianOptimizationOracle(
            objective=self.objective,
            max_trials=max_trials,  # pre-defined number,
            seed=self.random_seed,  # seed
            hyperparameters=self.hyperparameters,
            num_initial_points=num_initial_points,
        )
        self.oracle._set_project_dir(
            self.directory, self.project_name, overwrite=True)
