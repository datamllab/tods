import numpy

from d3m.metadata import hyperparams


class Sampling:
    def setup(self):
        self.numerical = hyperparams.Uniform(
            lower=0,
            upper=1,
            default=0.5,
        )
        self.enumeration = hyperparams.Enumeration(
            values=list(range(1000)),
            default=0,
        )

    def time_numerical_sampling(self):
        random_state = numpy.random.RandomState(0)
        for i in range(100000):
            self.numerical.sample(random_state)

    def time_numerical_sample_multiple(self):
        random_state = numpy.random.RandomState(0)
        for i in range(1000):
            self.numerical.sample_multiple(500, 500, random_state, with_replacement=False)

    def time_enumeration_sampling(self):
        random_state = numpy.random.RandomState(0)
        for i in range(10000):
            self.enumeration.sample(random_state)

    def time_enumeration_sample_multiple(self):
        random_state = numpy.random.RandomState(0)
        for i in range(10000):
            self.enumeration.sample_multiple(500, 500, random_state, with_replacement=False)
