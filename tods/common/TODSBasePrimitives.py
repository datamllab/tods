import typing
from typing import Any, Callable, List, Dict, Union, Optional, Sequence, Tuple
import logging
import abc

from d3m.primitive_interfaces import generator, transformer 
from d3m.primitive_interfaces.base import *
from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase
from tods.common.supervised_learning import SupervisedLearnerPrimitiveBase

from d3m.metadata import base as metadata_base, hyperparams, params
from d3m import container
from d3m import utils

__all__ = ('TODSTransformerPrimitiveBase',)

class TODSTransformerPrimitiveBase(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]): # pragma: no cover
    """
    A base class for primitives which are not fitted at all and can
    simply produce (useful) outputs from inputs directly. As such they
    also do not have any state (params).
    This class is parameterized using only three type variables, ``Inputs``,
    ``Outputs``, and ``Hyperparams``.
    """

    def __init__(self, *, hyperparams: Hyperparams) -> None:
        super().__init__(hyperparams=hyperparams)

    def produce(self, *, inputs: container.DataFrame, timeout: float = None, iterations: int = None) -> CallResult[container.DataFrame]:

        is_system = len(inputs.iloc[0, 0].shape) != 0 # check the shape of first row first column, if not a single data entry(,) then it is system-wise data (row, col)
        if is_system: 
            outputs = self._forward(inputs, '_produce')
        else:
            outputs = self._produce(inputs=inputs)
            outputs = outputs.value

        return CallResult(outputs) 

    @abc.abstractmethod
    def _produce(self, *, inputs: container.DataFrame, timeout: float = None, iterations: int = None) -> CallResult[container.DataFrame]:
        """
        make the predictions
        """
        #return CallResult(container.DataFrame)

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        """
        A noop.
        """
        return CallResult(None)

    def get_params(self) -> None:
        """
        A noop.
        """

        return None

    def set_params(self, *, params: None) -> None:
        """
        A noop.
        """

        return

    def _forward(self, data, method):
        """
        General Forward Function to feed system data one-by-one to the primitive
        """
        col_name = list(data.columns)[0]
        for i, _ in data.iterrows():
            sys_data = data.iloc[i][col_name]
            produce_func = getattr(self, method, None)
            out = produce_func(inputs=sys_data)
            data.iloc[i][col_name] = out.value
        return data

class TODSUnsupervisedLearnerPrimitiveBase(UnsupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):# pragma: no cover

    def __init__(self, *, hyperparams: Hyperparams, 
            random_seed: int=0, 
            docker_containers: Dict[str, DockerContainer] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

    def produce(self, *, inputs: container.DataFrame, timeout: float = None, iterations: int = None) -> CallResult[container.DataFrame]:

        is_system = len(inputs.iloc[0, 0].shape) != 0 # check the shape of first row first column, if not a single data entry(,) then it is system-wise data (row, col)
        if is_system: 
            outputs = self._forward(inputs, '_produce')
        else:
            outputs = self._produce(inputs=inputs)
            outputs = outputs.value

        return CallResult(outputs) 

    def produce_score(self, *, inputs: container.DataFrame, timeout: float = None, iterations: int = None) -> CallResult[container.DataFrame]:
        is_system = len(inputs.iloc[0, 0].shape) != 0 # check the shape of first row first column, if not a single data entry(,) then it is system-wise data (row, col)
        if is_system: 
            outputs = self._forward(inputs, '_produce_score')
        else:
            outputs = self._produce(inputs=inputs)
            outputs = outputs.value

        return CallResult(outputs) 

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        """
        A noop.
        """
        is_system = len(self._inputs.iloc[0, 0].shape) != 0 # check the shape of first row first column, if not a single data entry(,) then it is system-wise data (row, col)
        if is_system: 
            data = inputs
            col_name = list(data.columns)[0]
            for i, _ in data.iterrows():
                sys_data = data.iloc[i][col_name]
                self.set_training_data(inputs=sys_data)
                self._fit()
        else:
            outputs = self._fit()
            outputs = outputs.value

        return CallResult(None)

    def fit_multi_produce(self, *, produce_methods: typing.Sequence[str], inputs: Inputs, timeout: float = None, iterations: int = None) -> MultiCallResult:
        is_system = len(inputs.iloc[0, 0].shape) != 0 # check the shape of first row first column, if not a single data entry(,) then it is system-wise data (row, col)
        if is_system: 
            data = inputs
            produce_method = produce_methods[0]
            col_name = list(data.columns)[0]
            results = []
            for i, _ in data.iterrows():
                sys_data = data.iloc[i][col_name]
                self.set_training_data(inputs=sys_data)
                fit_result = self._fit()
                if produce_method == "produce":
                    out = self._produce(inputs=sys_data, timeout=timeout)
                else:
                    out = self._produce_score(inputs=sys_data, timeout=timeout)
                data.iloc[i][col_name] = out.value
                results.append(out)
            iterations_done = None
            for result in results:
                if result.iterations_done is not None:
                    if iterations_done is None:
                        iterations_done = result.iterations_done
                    else:
                        iterations_done = max(iterations_done, result.iterations_done)
            return MultiCallResult(
                values={produce_method: data},
                has_finished=all(result.has_finished for result in results),
                iterations_done=iterations_done,
                )
        else:
            return self._fit_multi_produce(produce_methods=produce_methods, timeout=timeout, iterations=iterations, inputs=inputs)

    @abc.abstractmethod
    def _produce(self, *, inputs: container.DataFrame, timeout: float = None, iterations: int = None) -> CallResult[container.DataFrame]:
        """
        abstract class
        """

    @abc.abstractmethod
    def _produce_score(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        abstract class
        """

    @abc.abstractmethod
    def _fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        """
        abstract class
        """


    def get_params(self) -> None:
        """
        A noop.
        """

        return None

    def set_params(self, *, params: None) -> None:
        """
        A noop.
        """

        return

    def _forward(self, data, method):
        """
        General Forward Function to feed system data one-by-one to the primitive
        """
        col_name = list(data.columns)[0]
        for i, _ in data.iterrows():
            sys_data = data.iloc[i][col_name]
            produce_func = getattr(self, method, None)
            out = produce_func(inputs=sys_data)
            data.iloc[i][col_name] = out.value
        return data


class TODSSupervisedLearnerPrimitiveBase(SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    def __init__(self, *, hyperparams: Hyperparams,
                 random_seed: int = 0,
                 docker_containers: Dict[str, DockerContainer] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

    def produce(self, *, inputs: container.DataFrame, timeout: float = None, iterations: int = None) -> CallResult[container.DataFrame]:
        """
            A noop.
        """
        return self._produce(inputs=inputs, timeout=timeout, iterations=iterations)

    def produce_score(self, *, inputs: container.DataFrame, timeout: float = None, iterations: int = None) -> CallResult[container.DataFrame]:
        """
            A noop.
        """
        return self._produce(inputs=inputs, timeout=timeout, iterations=iterations)

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:

        """
        A noop.
        """
        return self._fit(timeout=timeout, iterations=iterations)

    def fit_multi_produce(self, *, produce_methods: typing.Sequence[str], inputs: Inputs, outputs: Outputs, timeout: float = None, iterations: int = None) -> MultiCallResult:

        return self._fit_multi_produce(produce_methods=produce_methods, timeout=timeout, iterations=iterations, inputs=inputs, outputs=outputs)

    # def _produce(self, *, inputs: container.DataFrame, timeout: float = None, iterations: int = None) -> CallResult[container.DataFrame]:
    #
    #     pass
    #
    # def _produce_score(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
    #
    #     pass
    #
    # def _fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
    #
    #     pass

    def get_params(self) -> None:
        """
        A noop.
        """

        return None

    def set_params(self, *, params: None) -> None:
        """
        A noop.
        """

        return
