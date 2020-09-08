import abc
import inspect
import logging
import time
import typing

from d3m import exceptions, types, utils
from d3m.metadata import base as metadata_base, hyperparams, params, problem

__all__ = (
    'Inputs', 'Outputs', 'Params', 'Hyperparams', 'CallResult', 'MultiCallResult', 'DockerContainer',
    'PrimitiveBase', 'ContinueFitMixin', 'SamplingCompositionalityMixin',
    'ProbabilisticCompositionalityMixin', 'Gradients',
    'GradientCompositionalityMixin', 'LossFunctionMixin',
    'NeuralNetworkModuleMixin', 'NeuralNetworkObjectMixin',
    'singleton', 'inputs_across_samples',
)


Inputs = typing.TypeVar('Inputs', bound=typing.Union[types.Container])  # type: ignore
Outputs = typing.TypeVar('Outputs', bound=typing.Union[types.Container])  # type: ignore
# This type parameter is optional and can be set to None.
# See "TransformerPrimitiveBase" for an example.
Params = typing.TypeVar('Params', bound=params.Params)
Hyperparams = typing.TypeVar('Hyperparams', bound=hyperparams.Hyperparams)
Module = typing.TypeVar('Module')

T = typing.TypeVar('T')

# All base classes (primitive interfaces) should have docstrings starting with this language.
# This allows us to validate that primitives have changed their descriptions/docstrings to something different.
DEFAULT_DESCRIPTION = "A base class for primitives"


class CallResult(typing.Generic[T]):
    """
    Some methods return additional metadata about the method call itself
    (which is different to metadata about the value returned, which is stored
    in ``metadata`` attribute of the value itself).

    For ``produce`` method call, ``has_finished`` is ``True`` if the last call
    to ``produce`` has produced the final outputs and a call with more time or
    more iterations cannot get different outputs.

    For ``fit`` method call, ``has_finished`` is ``True`` if a primitive has been
    fully fitted on current training data and further calls to ``fit`` are
    unnecessary and will not change anything. ``False`` means that more iterations
    can be done (but it does not necessary mean that more iterations are beneficial).

    If a primitive has iterations internally, then ``iterations_done`` contains
    how many of those iterations have been made during the last call. If primitive
    does not support them, ``iterations_done`` is ``None``.

    Those methods should return value wrapped into this class.

    Parameters
    ----------
    value:
        The value itself of the method call.
    has_finished:
        Set to ``True`` if it is not reasonable to call the method again anymore.
    iterations_done:
        How many iterations have been done during a method call, if any.
    """

    def __init__(self, value: T, has_finished: bool = True, iterations_done: int = None) -> None:
        self.value = value
        self.has_finished = has_finished
        self.iterations_done = iterations_done


class MultiCallResult:
    """
    Similar to `CallResult`, but used by ``multi_produce``.

    It has no precise typing information because type would have to be a dependent type
    which is not (yet) supported in standard Python typing. Type would depend on
    ``produce_methods`` argument and output types of corresponding produce methods.

    Parameters
    ----------
    values:
        A dict of values mapping between produce method names and their value outputs.
    has_finished:
        Set to ``True`` if it is not reasonable to call the method again anymore.
    iterations_done:
        How many iterations have been done during a method call, if any.
    """

    def __init__(self, values: typing.Dict, has_finished: bool = True, iterations_done: int = None) -> None:
        self.values = values
        self.has_finished = has_finished
        self.iterations_done = iterations_done


class PrimitiveBaseMeta(utils.GenericMetaclass):
    """
    A metaclass which provides the primitive instance to metadata so that primitive
    metadata can be automatically generated.
    """

    def __new__(mcls, class_name, bases, namespace, **kwargs):  # type: ignore
        cls = super().__new__(mcls, class_name, bases, namespace, **kwargs)

        if inspect.isabstract(cls):
            return cls

        if not isinstance(cls.metadata, metadata_base.PrimitiveMetadata):
            raise TypeError("'metadata' attribute is not an instance of PrimitiveMetadata.")

        # We are creating a class-level logger so that it can be used both from class and instance methods.
        # "python_path" is a required metadata value, but we leave metadata validation to later.
        python_path = cls.metadata.query().get('python_path', None)
        if python_path is not None:
            cls.logger = logging.getLogger(python_path)

        cls.metadata.contribute_to_class(cls)

        return cls

    def __repr__(cls) -> str:
        if getattr(cls, 'metadata', None) is not None:
            return cls.metadata.query().get('python_path', super().__repr__())
        else:
            return super().__repr__()


class DockerContainer(typing.NamedTuple):
    """
    A tuple suitable to describe connection information necessary to connect
    to exposed ports of a running Docker container.

    Attributes
    ----------
    address:
        An address at which the Docker container is available.
    ports:
        Mapping between image's exposed ports and real ports. E.g.,
        ``{'80/tcp': 80}``.
    """

    address: str
    ports: typing.Dict[str, int]


class PrimitiveBase(typing.Generic[Inputs, Outputs, Params, Hyperparams], metaclass=PrimitiveBaseMeta):
    """
    A base class for primitives.

    Class is parameterized using four type variables, ``Inputs``, ``Outputs``, ``Params``,
    and ``Hyperparams``.

    ``Params`` has to be a subclass of `d3m.metadata.params.Params` and should define
    all fields and their types for parameters which the primitive is fitting.

    ``Hyperparams`` has to be a subclass of a `d3m.metadata.hyperparams.Hyperparams`.
    Hyper-parameters are those primitive's parameters which primitive is not fitting and
    generally do not change during a life-time of a primitive.

    ``Params`` and ``Hyperparams`` have to be picklable and copyable. See `pickle`,
    `copy`, and `copyreg` Python modules for more information.

    In this context we use term method arguments to mean both formal parameters and
    actual parameters of a method. We do this to not confuse method parameters with
    primitive parameters (``Params``).

    All arguments to all methods are keyword-only. No ``*args`` or ``**kwargs`` should
    ever be used in any method.

    Standardized interface use few public attributes and no other public attributes are
    allowed to assure future compatibility. For your attributes use the convention that
    private symbols should start with ``_``.

    Primitives can have methods which are not part of standardized interface classes:

    * Additional "produce" methods which are prefixed with ``produce_`` and have
      the same semantics as ``produce`` but potentially return different output
      container types instead of ``Outputs`` (in such primitive ``Outputs`` is seen as
      primary output type, but the primitive also has secondary output types).
      They should return ``CallResult`` and have ``timeout`` and ``iterations`` arguments.
    * Private methods prefixed with ``_``.

    No other public additional methods are allowed. If this represents a problem for you,
    open an issue. (The rationale is that for other methods an automatic system will not
    understand the semantics of the method.)

    Method arguments which start with ``_`` are seen as private and can be used for arguments
    useful for debugging and testing, but they should not be used by (or even known to) a
    caller during normal execution. Such arguments have to be optional (have a default value)
    so that the method can be called without the knowledge of the argument.

    All arguments to all methods and all hyper-parameters together are seen as arguments to
    the primitive as a whole. They are identified by their names. This means that any argument
    name must have the same type and semantics across all methods, effectively be the same argument.
    If a method argument matches in name a hyper-parameter, it has to match it in type and semantics
    as well. Such method argument overrides a hyper-parameter for a method call. All this is necessary
    so that callers can have easier time determine what values to pass to arguments and that it is
    easier to describe what all values are inputs to a primitive as a whole (set of all
    arguments).

    To recap, subclasses can extend arguments of standard methods with explicit typed keyword
    arguments used for the method call, or define new "produce" methods with arbitrary explicit
    typed keyword arguments. There are multiple kinds of such arguments allowed:

    * An (additional) input argument of any container type and not necessary of ``Inputs``
      (in such primitive ``Inputs`` is seen as primary input type, but the primitive also has
      secondary input types).
    * An argument which is overriding a hyper-parameter for the duration of the call.
      It should match a hyper-parameter in name and type. It should be a required argument
      (no default value) which the caller has to supply (or with a default value of a
      hyper-parameter, or with the same hyper-parameter as it was passed to the constructor,
      or with some other value). This is meant just for fine-control by a caller during fitting
      or producing, e.g., for a threshold or learning rate, and is not reasonable for most
      hyper-parameters.
    * An (additional) value argument which is one of standard data types, but not a container type.
      In this case a caller will try to satisfy the input by creating part of a pipeline which
      ends with a primitive with singleton produce method and extract the singleton value and
      pass it without a container. This kind of an argument is **discouraged** and should probably
      be a hyper-parameter instead (because it is unclear how can a caller determine which value
      is a reasonable value to pass in an automatic way), but it is defined for completeness and
      so that existing pipelines can be easier described.
    * A private argument prefixed with ``_`` which is used for debugging and testing.
      It should not be used by (or even known to) a caller during normal execution.
      Such argument has to be optional (have a default value) so that the method can be called
      without the knowledge of the argument.

    Each primitive's class automatically gets an instance of Python's logging logger stored
    into its ``logger`` class attribute. The instance is made under the name of primitive's
    ``python_path`` metadata value. Primitives can use this logger to log information at
    various levels (debug, warning, error) and even associate extra data with log record
    using the ``extra`` argument to the logger calls.

    Subclasses of this class allow functional compositionality.

    Attributes
    ----------
    metadata:
        Primitive's metadata. Available as a class attribute.
    logger:
        Primitive's logger. Available as a class attribute.
    hyperparams:
        Hyperparams passed to the constructor.
    random_seed:
        Random seed passed to the constructor.
    docker_containers:
        A dict mapping Docker image keys from primitive's metadata to (named) tuples containing
        container's address under which the container is accessible by the primitive, and a
        dict mapping exposed ports to ports on that address.
    volumes:
        A dict mapping volume keys from primitive's metadata to file and directory paths
        where downloaded and extracted files are available to the primitive.
    temporary_directory:
        An absolute path to a temporary directory a primitive can use to store any files
        for the duration of the current pipeline run phase. Directory is automatically
        cleaned up after the current pipeline run phase finishes.
    """

    # Primitive's metadata (annotation) should be put on "metadata' attribute to provide
    # all fields (which cannot be determined automatically) inside the code. In this way metadata
    # is close to the code and it is easier for consumers to make sure metadata they are using
    # is really matching the code they are using. PrimitiveMetadata class will automatically
    # extract additional metadata and update itself with metadata about code and other things
    # it can extract automatically.
    metadata: typing.ClassVar[metadata_base.PrimitiveMetadata] = None

    # This gets automatically set to primitive's logger in metaclass.
    logger: typing.ClassVar[logging.Logger] = None

    hyperparams: Hyperparams
    random_seed: int
    docker_containers: typing.Dict[str, DockerContainer]
    volumes: typing.Dict[str, str]
    temporary_directory: str

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0,
                 docker_containers: typing.Dict[str, DockerContainer] = None,
                 volumes: typing.Dict[str, str] = None,
                 temporary_directory: str = None) -> None:
        """
        All primitives should accept all their hyper-parameters in a constructor as one value,
        an instance of type ``Hyperparams``.

        Provided random seed should control all randomness used by this primitive.
        Primitive should behave exactly the same for the same random seed across multiple
        invocations. You can call `numpy.random.RandomState(random_seed)` to obtain an
        instance of a random generator using provided seed. If your primitive does not
        use randomness, consider not exposing this argument in your primitive's constructor
        to signal that.

        Primitives can be wrappers around or use one or more Docker images which they can
        specify as part of  ``installation`` field in their metadata. Each Docker image listed
        there has a ``key`` field identifying that image. When primitive is created,
        ``docker_containers`` contains a mapping between those keys and connection information
        which primitive can use to connect to a running Docker container for a particular Docker
        image and its exposed ports. Docker containers might be long running and shared between
        multiple instances of a primitive. If your primitive does not use Docker images,
        consider not exposing this argument in your primitive's constructor.

        **Note**: Support for primitives using Docker containers has been put on hold.
        Currently it is not expected that any runtime running primitives will run
        Docker containers for a primitive.

        Primitives can also use additional static files which can be added as a dependency
        to ``installation`` metadata. When done so, given volumes are provided to the
        primitive through ``volumes`` argument to the primitive's constructor as a
        dict mapping volume keys to file and directory paths where downloaded and
        extracted files are available to the primitive. All provided files and directories
        are read-only. If your primitive does not use static files, consider not exposing
        this argument in your primitive's constructor.

        Primitives can also use the provided temporary directory to store any files for
        the duration of the current pipeline run phase. Directory is automatically
        cleaned up after the current pipeline run phase finishes. Do not store in this
        directory any primitive's state you would like to preserve between "fit" and
        "produce" phases of pipeline execution. Use ``Params`` for that. The main intent
        of this temporary directory is to store files referenced by any ``Dataset`` object
        your primitive might create and followup primitives in the pipeline should have
        access to. When storing files into this directory consider using capabilities
        of Python's `tempfile` module to generate filenames which will not conflict with
        any other files stored there. Use provided temporary directory as ``dir`` argument
        to set it as base directory to generate additional temporary files and directories
        as needed. If your primitive does not use temporary directory, consider not exposing
        this argument in your primitive's constructor.

        No other arguments to the constructor are allowed (except for private arguments)
        because we want instances of primitives to be created without a need for any other
        prior computation.

        Module in which a primitive is defined should be kept lightweight and on import not do
        any (pre)computation, data loading, or resource allocation/reservation. Any loading
        and resource allocation/reservation should be done in the constructor. Any (pre)computation
        should be done lazily when needed once requested through other methods and not in the constructor.
        """

        self.hyperparams = hyperparams
        self.random_seed = random_seed
        if docker_containers is None:
            self.docker_containers: typing.Dict[str, DockerContainer] = {}
        else:
            self.docker_containers = docker_containers
        if volumes is None:
            self.volumes: typing.Dict[str, str] = {}
        else:
            self.volumes = volumes
        self.temporary_directory = temporary_directory

    @abc.abstractmethod
    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Produce primitive's best choice of the output for each of the inputs.

        The output value should be wrapped inside ``CallResult`` object before returning.

        In many cases producing an output is a quick operation in comparison with ``fit``, but not
        all cases are like that. For example, a primitive can start a potentially long optimization
        process to compute outputs. ``timeout`` and ``iterations`` can serve as a way for a caller
        to guide the length of this process.

        Ideally, a primitive should adapt its call to try to produce the best outputs possible
        inside the time allocated. If this is not possible and the primitive reaches the timeout
        before producing outputs, it should raise a ``TimeoutError`` exception to signal that the
        call was unsuccessful in the given time. The state of the primitive after the exception
        should be as the method call has never happened and primitive should continue to operate
        normally. The purpose of ``timeout`` is to give opportunity to a primitive to cleanly
        manage its state instead of interrupting execution from outside. Maintaining stable internal
        state should have precedence over respecting the ``timeout`` (caller can terminate the
        misbehaving primitive from outside anyway). If a longer ``timeout`` would produce
        different outputs, then ``CallResult``'s ``has_finished`` should be set to ``False``.

        Some primitives have internal iterations (for example, optimization iterations).
        For those, caller can provide how many of primitive's internal iterations
        should a primitive do before returning outputs. Primitives should make iterations as
        small as reasonable. If ``iterations`` is ``None``, then there is no limit on
        how many iterations the primitive should do and primitive should choose the best amount
        of iterations on its own (potentially controlled through hyper-parameters).
        If ``iterations`` is a number, a primitive has to do those number of iterations,
        if possible. ``timeout`` should still be respected and potentially less iterations
        can be done because of that. Primitives with internal iterations should make
        ``CallResult`` contain correct values.

        For primitives which do not have internal iterations, any value of ``iterations``
        means that they should run fully, respecting only ``timeout``.

        If primitive should have been fitted before calling this method, but it has not been,
        primitive should raise a ``PrimitiveNotFittedError`` exception.

        Parameters
        ----------
        inputs:
            The inputs of shape [num_inputs, ...].
        timeout:
            A maximum time this primitive should take to produce outputs during this method call, in seconds.
        iterations:
            How many of internal iterations should the primitive do.

        Returns
        -------
        The outputs of shape [num_inputs, ...] wrapped inside ``CallResult``.
        """

    def multi_produce(self, *, produce_methods: typing.Sequence[str], inputs: Inputs, timeout: float = None, iterations: int = None) -> MultiCallResult:
        """
        A method calling multiple produce methods at once.

        When a primitive has multiple produce methods it is common that they might compute the
        same internal results for same inputs but return different representations of those results.
        If caller is interested in multiple of those representations, calling multiple produce
        methods might lead to recomputing same internal results multiple times. To address this,
        this method allows primitive author to implement an optimized version which computes
        internal results only once for multiple calls of produce methods, but return those different
        representations.

        If any additional method arguments are added to primitive's produce method(s), they have
        to be added to this method as well. This method should accept an union of all arguments
        accepted by primitive's produce method(s) and then use them accordingly when computing
        results.

        The default implementation of this method just calls all produce methods listed in
        ``produce_methods`` in order and is potentially inefficient.

        If primitive should have been fitted before calling this method, but it has not been,
        primitive should raise a ``PrimitiveNotFittedError`` exception.

        Parameters
        ----------
        produce_methods:
            A list of names of produce methods to call.
        inputs:
            The inputs given to all produce methods.
        timeout:
            A maximum time this primitive should take to produce outputs for all produce methods
            listed in ``produce_methods`` argument, in seconds.
        iterations:
            How many of internal iterations should the primitive do.

        Returns
        -------
        A dict of values for each produce method wrapped inside ``MultiCallResult``.
        """

        return self._multi_produce(produce_methods=produce_methods, timeout=timeout, iterations=iterations, inputs=inputs)

    def _multi_produce(self, *, produce_methods: typing.Sequence[str], timeout: float = None, iterations: int = None, **kwargs: typing.Dict[str, typing.Any]) -> MultiCallResult:
        """
        We do not want a public API to use ``kwargs``, but such implementation allows easier subclassing and reuse
        of a default implementation. Do not call directly.
        """

        results = []
        for method_name in produce_methods:
            if method_name != 'produce' and not method_name.startswith('produce_'):
                raise exceptions.InvalidArgumentValueError("Invalid produce method name '{method_name}'.".format(method_name=method_name))

            if not hasattr(self, method_name):
                raise exceptions.InvalidArgumentValueError("Unknown produce method name '{method_name}'.".format(method_name=method_name))

            try:
                expected_arguments = set(self.metadata.query()['primitive_code'].get('instance_methods', {})[method_name]['arguments'])
            except KeyError as error:
                raise exceptions.InvalidArgumentValueError("Unknown produce method name '{method_name}'.".format(method_name=method_name)) from error

            arguments = {name: value for name, value in kwargs.items() if name in expected_arguments}

            start = time.perf_counter()
            results.append(getattr(self, method_name)(timeout=timeout, iterations=iterations, **arguments))
            delta = time.perf_counter() - start

            # Decrease the amount of time available to other calls. This delegates responsibility
            # of raising a "TimeoutError" exception to produce methods themselves. It also assumes
            # that if one passes a negative timeout value to a produce method, it raises a
            # "TimeoutError" exception correctly.
            if timeout is not None:
                timeout -= delta

            if not isinstance(results[-1], CallResult):
                raise exceptions.InvalidReturnTypeError("Primitive's produce method '{method_name}' has not returned a CallResult.".format(
                    method_name=method_name,
                ))

        # We return the maximum number of iterations done by any produce method we called.
        iterations_done = None
        for result in results:
            if result.iterations_done is not None:
                if iterations_done is None:
                    iterations_done = result.iterations_done
                else:
                    iterations_done = max(iterations_done, result.iterations_done)

        return MultiCallResult(
            values={name: result.value for name, result in zip(produce_methods, results)},
            has_finished=all(result.has_finished for result in results),
            iterations_done=iterations_done,
        )

    def fit_multi_produce(self, *, produce_methods: typing.Sequence[str], inputs: Inputs, outputs: Outputs, timeout: float = None, iterations: int = None) -> MultiCallResult:
        """
        A method calling ``fit`` and after that multiple produce methods at once.

        This method allows primitive author to implement an optimized version of both fitting
        and producing a primitive on same data.

        If any additional method arguments are added to primitive's ``set_training_data`` method
        or produce method(s), or removed from them, they have to be added to or removed from this
        method as well. This method should accept an union of all arguments accepted by primitive's
        ``set_training_data`` method and produce method(s) and then use them accordingly when
        computing results.

        The default implementation of this method just calls first ``set_training_data`` method,
        ``fit`` method, and all produce methods listed in ``produce_methods`` in order and is
        potentially inefficient.

        Parameters
        ----------
        produce_methods:
            A list of names of produce methods to call.
        inputs:
            The inputs given to ``set_training_data`` and all produce methods.
        outputs:
            The outputs given to ``set_training_data``.
        timeout:
            A maximum time this primitive should take to both fit the primitive and produce outputs
            for all produce methods listed in ``produce_methods`` argument, in seconds.
        iterations:
            How many of internal iterations should the primitive do for both fitting and producing
            outputs of all produce methods.

        Returns
        -------
        A dict of values for each produce method wrapped inside ``MultiCallResult``.
        """

        return self._fit_multi_produce(produce_methods=produce_methods, timeout=timeout, iterations=iterations, inputs=inputs, outputs=outputs)

    def _fit_multi_produce(self, *, produce_methods: typing.Sequence[str], timeout: float = None, iterations: int = None, **kwargs: typing.Dict[str, typing.Any]) -> MultiCallResult:
        """
        We do not want a public API to use ``kwargs``, but such implementation allows easier subclassing and reuse
        of a default implementation. Do not call directly.
        """

        try:
            expected_arguments = set(self.metadata.query()['primitive_code'].get('instance_methods', {})['set_training_data']['arguments'])
        except KeyError as error:
            raise exceptions.InvalidArgumentValueError("Unknown produce method name '{method_name}'.".format(method_name='set_training_data')) from error

        arguments = {name: value for name, value in kwargs.items() if name in expected_arguments}

        start = time.perf_counter()
        self.set_training_data(**arguments)  # type: ignore
        delta = time.perf_counter() - start

        # Decrease the amount of time available to other calls. This delegates responsibility
        # of raising a "TimeoutError" exception to fit and produce methods themselves.
        # It also assumes that if one passes a negative timeout value to a fit or a produce
        # method, it raises a "TimeoutError" exception correctly.
        if timeout is not None:
            timeout -= delta

        start = time.perf_counter()
        fit_result = self.fit(timeout=timeout, iterations=iterations)
        delta = time.perf_counter() - start

        if timeout is not None:
            timeout -= delta

        if not isinstance(fit_result, CallResult):
            raise exceptions.InvalidReturnTypeError("Primitive's fit method has not returned a CallResult.")

        produce_results = self._multi_produce(produce_methods=produce_methods, timeout=timeout, iterations=iterations, **kwargs)

        results: typing.List[typing.Union[CallResult, MultiCallResult]] = [fit_result, produce_results]

        # We return the maximum number of iterations done by a fit method or any produce method we called.
        iterations_done = None
        for result in results:
            if result.iterations_done is not None:
                if iterations_done is None:
                    iterations_done = result.iterations_done
                else:
                    iterations_done = max(iterations_done, result.iterations_done)

        return MultiCallResult(
            # We return values just from produce methods.
            values=produce_results.values,
            has_finished=all(result.has_finished for result in results),
            iterations_done=iterations_done,
        )

    @abc.abstractmethod
    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        """
        Sets current training data of this primitive.

        This marks training data as changed even if new training data is the same as
        previous training data.

        Standard sublasses in this package do not adhere to the Liskov substitution principle when
        inheriting this method because they do not necessary accept all arguments found in the base
        class. This means that one has to inspect which arguments are accepted at runtime, or in
        other words, one has to inspect which exactly subclass a primitive implements, if
        you are accepting a wider range of primitives. This relaxation is allowed only for
        standard subclasses found in this package. Primitives themselves should not break
        the Liskov substitution principle but should inherit from a suitable base class.

        Parameters
        ----------
        inputs:
            The inputs.
        outputs:
            The outputs.
        """

    @abc.abstractmethod
    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        """
        Fits primitive using inputs and outputs (if any) using currently set training data.

        The returned value should be a ``CallResult`` object with ``value`` set to ``None``.

        If ``fit`` has already been called in the past on different training data,
        this method fits it **again from scratch** using currently set training data.

        On the other hand, caller can call ``fit`` multiple times on the same training data
        to continue fitting.

        If ``fit`` fully fits using provided training data, there is no point in making further
        calls to this method with same training data, and in fact further calls can be noops,
        or a primitive can decide to fully refit from scratch.

        In the case fitting can continue with same training data (even if it is maybe not reasonable,
        because the internal metric primitive is using looks like fitting will be degrading), if ``fit``
        is called again (without setting training data), the primitive has to continue fitting.

        Caller can provide ``timeout`` information to guide the length of the fitting process.
        Ideally, a primitive should adapt its fitting process to try to do the best fitting possible
        inside the time allocated. If this is not possible and the primitive reaches the timeout
        before fitting, it should raise a ``TimeoutError`` exception to signal that fitting was
        unsuccessful in the given time. The state of the primitive after the exception should be
        as the method call has never happened and primitive should continue to operate normally.
        The purpose of ``timeout`` is to give opportunity to a primitive to cleanly manage
        its state instead of interrupting execution from outside. Maintaining stable internal state
        should have precedence over respecting the ``timeout`` (caller can terminate the misbehaving
        primitive from outside anyway). If a longer ``timeout`` would produce different fitting,
        then ``CallResult``'s ``has_finished`` should be set to ``False``.

        Some primitives have internal fitting iterations (for example, epochs). For those, caller
        can provide how many of primitive's internal iterations should a primitive do before returning.
        Primitives should make iterations as small as reasonable. If ``iterations`` is ``None``,
        then there is no limit on how many iterations the primitive should do and primitive should
        choose the best amount of iterations on its own (potentially controlled through
        hyper-parameters). If ``iterations`` is a number, a primitive has to do those number of
        iterations (even if not reasonable), if possible. ``timeout`` should still be respected
        and potentially less iterations can be done because of that. Primitives with internal
        iterations should make ``CallResult`` contain correct values.

        For primitives which do not have internal iterations, any value of ``iterations``
        means that they should fit fully, respecting only ``timeout``.

        Parameters
        ----------
        timeout:
            A maximum time this primitive should be fitting during this method call, in seconds.
        iterations:
            How many of internal iterations should the primitive do.

        Returns
        -------
        A ``CallResult`` with ``None`` value.
        """

    @abc.abstractmethod
    def get_params(self) -> Params:
        """
        Returns parameters of this primitive.

        Parameters are all parameters of the primitive which can potentially change during a life-time of
        a primitive. Parameters which cannot are passed through constructor.

        Parameters should include all data which is necessary to create a new instance of this primitive
        behaving exactly the same as this instance, when the new instance is created by passing the same
        parameters to the class constructor and calling ``set_params``.

        No other arguments to the method are allowed (except for private arguments).

        Returns
        -------
        An instance of parameters.
        """

    @abc.abstractmethod
    def set_params(self, *, params: Params) -> None:
        """
        Sets parameters of this primitive.

        Parameters are all parameters of the primitive which can potentially change during a life-time of
        a primitive. Parameters which cannot are passed through constructor.

        No other arguments to the method are allowed (except for private arguments).

        Parameters
        ----------
        params:
            An instance of parameters.
        """

    def __getstate__(self) -> dict:
        """
        Returns state which is used to pickle an instance of a primitive.

        By default it returns standard constructor arguments and value
        returned from ``get_params`` method.

        Consider extending default implementation if your primitive accepts
        additional constructor arguments you would like to preserve when pickling.

        Note that unpickled primitive instances can generally continue to work only
        inside the same environment they were pickled in because they continue to use
        same ``docker_containers``, ``volumes``, and ``temporary_directory`` values
        passed initially to primitive's constructor. Those generally do not work in
        another environment where those resources might be available differently.
        Consider constructing primitive instance directly providing updated constructor
        arguments and then using ``get_params``/``set_params`` to restore primitive's
        state.

        Returns
        -------
        State to pickle.
        """

        standard_arguments = {
            'hyperparams': self.hyperparams,
            'random_seed': self.random_seed,
            'docker_containers': self.docker_containers,
            'volumes': self.volumes,
            'temporary_directory': self.temporary_directory,
        }
        expected_constructor_arguments = self.metadata.query()['primitive_code'].get('instance_methods', {})['__init__']['arguments']

        return {
            'constructor': {name: value for name, value in standard_arguments.items() if name in expected_constructor_arguments},
            'params': self.get_params(),
        }

    def __setstate__(self, state: dict) -> None:
        """
        Uses ``state`` to restore the state of a primitive when unpickling.

        By default it passes constructor arguments to the constructor and
        calls ``get_params``.

        Parameters
        ----------
        state:
            Unpickled state.
        """

        self.__init__(**state['constructor'])  # type: ignore
        self.set_params(params=state['params'])

    def __repr__(self) -> str:
        if 'random_seed' in self.metadata.query().get('primitive_code', {}).get('instance_methods', {}).get('__init__', {}).get('arguments', []):
            return '{class_name}(hyperparams={hyperparams}, random_seed={random_seed})'.format(
                class_name=self.metadata.query()['python_path'],
                hyperparams=self.hyperparams,
                random_seed=self.random_seed,
            )
        else:
            return '{class_name}(hyperparams={hyperparams})'.format(
                class_name=self.metadata.query()['python_path'],
                hyperparams=self.hyperparams,
            )


class ContinueFitMixin(typing.Generic[Inputs, Outputs, Params, Hyperparams], metaclass=utils.GenericMetaclass):
    @abc.abstractmethod
    def continue_fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        """
        Similar to base ``fit``, this method fits the primitive using inputs and outputs (if any)
        using currently set training data.

        The difference is what happens when currently set training data is different from
        what the primitive might have already been fitted on. ``fit`` resets parameters and
        refits the primitive (restarts fitting), while ``continue_fit`` fits the primitive
        further on new training data. ``fit`` does **not** have to be called before ``continue_fit``,
        calling ``continue_fit`` first starts fitting as well.

        Caller can still call ``continue_fit`` multiple times on the same training data as well,
        in which case primitive should try to improve the fit in the same way as with ``fit``.

        From the perspective of a caller of all other methods, the training data in effect
        is still just currently set training data. If a caller wants to call ``gradient_output``
        on all data on which the primitive has been fitted through multiple calls of ``continue_fit``
        on different training data, the caller should pass all this data themselves through
        another call to ``set_training_data``, do not call ``fit`` or ``continue_fit`` again,
        and use ``gradient_output`` method. In this way primitives which truly support
        continuation of fitting and need only the latest data to do another fitting, do not
        have to keep all past training data around themselves.

        If a primitive supports this mixin, then both ``fit`` and ``continue_fit`` can be
        called. ``continue_fit`` always continues fitting, if it was started through ``fit``
        or ``continue_fit`` and fitting has not already finished. Calling ``fit`` always restarts
        fitting after ``continue_fit`` has been called, even if training data has not changed.

        Primitives supporting this mixin and which operate on categorical target columns should
        use ``all_distinct_values`` metadata to obtain which all values (labels) can be in
        a target column, even if currently set training data does not contain all those values.

        Parameters
        ----------
        timeout:
            A maximum time this primitive should be fitting during this method call, in seconds.
        iterations:
            How many of internal iterations should the primitive do.

        Returns
        -------
        A ``CallResult`` with ``None`` value.
        """


class SamplingCompositionalityMixin(typing.Generic[Inputs, Outputs, Params, Hyperparams], metaclass=utils.GenericMetaclass):
    """
    This mixin signals to a caller that the primitive is probabilistic but
    may be likelihood free.
    """

    @abc.abstractmethod
    def sample(self, *, inputs: Inputs, num_samples: int = 1, timeout: float = None, iterations: int = None) -> CallResult[typing.Sequence[Outputs]]:
        """
        Sample output for each input from ``inputs`` ``num_samples`` times.

        Semantics of ``timeout`` and ``iterations`` is the same as in ``produce``.

        Parameters
        ----------
        inputs:
            The inputs of shape [num_inputs, ...].
        num_samples:
            The number of samples to return in a set of samples.
        timeout:
            A maximum time this primitive should take to sample outputs during this method call, in seconds.
        iterations:
            How many of internal iterations should the primitive do.

        Returns
        -------
        The multiple sets of samples of shape [num_samples, num_inputs, ...] wrapped inside
        ``CallResult``. While the output value type is specified as ``Sequence[Outputs]``, the
        output value can be in fact any container type with dimensions/shape equal to combined
        ``Sequence[Outputs]`` dimensions/shape. Subclasses should specify which exactly type
        the output is.
        """


class ProbabilisticCompositionalityMixin(typing.Generic[Inputs, Outputs, Params, Hyperparams], metaclass=utils.GenericMetaclass):
    """
    This mixin provides additional abstract methods which primitives should implement to
    help callers with doing various end-to-end refinements using probabilistic
    compositionality.

    This mixin adds methods to support at least:

    * Metropolis-Hastings

    Mixin should be used together with ``SamplingCompositionalityMixin`` mixin.
    """

    @abc.abstractmethod
    def log_likelihoods(self, *, outputs: Outputs, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Returns log probability of outputs given inputs and params under this primitive:

        log(p(output_i | input_i, params))

        Parameters
        ----------
        outputs:
            The outputs. The number of samples should match ``inputs``.
        inputs:
            The inputs. The number of samples should match ``outputs``.
        timeout:
            A maximum time this primitive should take to produce outputs during this method call, in seconds.
        iterations:
            How many of internal iterations should the primitive do.

        Returns
        -------
        log(p(output_i | input_i, params))) wrapped inside ``CallResult``.
        The number of columns should match the number of target columns in ``outputs``.
        """

    def log_likelihood(self, *, outputs: Outputs, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Returns log probability of outputs given inputs and params under this primitive:

        sum_i(log(p(output_i | input_i, params)))

        By default it calls ``log_likelihoods`` and tries to automatically compute a sum, but subclasses can
        implement a more efficient or even correct version.

        Parameters
        ----------
        outputs:
            The outputs. The number of samples should match ``inputs``.
        inputs:
            The inputs. The number of samples should match ``outputs``.
        timeout:
            A maximum time this primitive should take to produce outputs during this method call, in seconds.
        iterations:
            How many of internal iterations should the primitive do.

        Returns
        -------
        sum_i(log(p(output_i | input_i, params))) wrapped inside ``CallResult``.
        The number of returned samples is always 1.
        The number of columns should match the number of target columns in ``outputs``.
        """

        result = self.log_likelihoods(outputs=outputs, inputs=inputs, timeout=timeout, iterations=iterations)

        return CallResult(utils.columns_sum(result.value), result.has_finished, result.iterations_done)


Container = typing.TypeVar('Container', bound=typing.Union[types.Container])  # type: ignore


# TODO: This is not yet a properly defined type which would really be recognized similar to Container.
#       You should specify a proper type in your subclass. Type checking might complain that your
#       type does not match the parent type, but ignore it (add "type: ignore" comment to that line).
#       This type will be fixed in the future.
class Gradients(typing.Generic[Container]):
    """
    A type representing a structure similar to ``Container``, but the values are of type ``Optional[float]``.
    Value is ``None`` if gradient for that part of the structure is not possible.
    """


class GradientCompositionalityMixin(typing.Generic[Inputs, Outputs, Params, Hyperparams], metaclass=utils.GenericMetaclass):
    """
    This mixin provides additional abstract methods which primitives should implement to
    help callers with doing various end-to-end refinements using gradient-based
    compositionality.

    This mixin adds methods to support at least:

    * gradient-based, compositional end-to-end training
    * regularized pre-training
    * multi-task adaptation
    * black box variational inference
    * Hamiltonian Monte Carlo
    """

    @abc.abstractmethod
    def gradient_output(self, *, outputs: Outputs, inputs: Inputs) -> Gradients[Outputs]:
        """
        Returns the gradient of loss sum_i(L(output_i, produce_one(input_i))) with respect to outputs.

        When fit term temperature is set to non-zero, it should return the gradient with respect to outputs of:

        sum_i(L(output_i, produce_one(input_i))) + temperature * sum_i(L(training_output_i, produce_one(training_input_i)))

        When used in combination with the ``ProbabilisticCompositionalityMixin``, it returns gradient
        of sum_i(log(p(output_i | input_i, params))) with respect to outputs.

        When fit term temperature is set to non-zero, it should return the gradient with respect to outputs of:

        sum_i(log(p(output_i | input_i, params))) + temperature * sum_i(log(p(training_output_i | training_input_i, params)))

        Parameters
        ----------
        outputs:
            The outputs.
        inputs:
            The inputs.

        Returns
        -------
        A structure similar to ``Container`` but the values are of type ``Optional[float]``.
        """

    @abc.abstractmethod
    def gradient_params(self, *, outputs: Outputs, inputs: Inputs) -> Gradients[Params]:
        """
        Returns the gradient of loss sum_i(L(output_i, produce_one(input_i))) with respect to params.

        When fit term temperature is set to non-zero, it should return the gradient with respect to params of:

        sum_i(L(output_i, produce_one(input_i))) + temperature * sum_i(L(training_output_i, produce_one(training_input_i)))

        When used in combination with the ``ProbabilisticCompositionalityMixin``, it returns gradient
        of sum_i(log(p(output_i | input_i, params))) with respect to params.

        When fit term temperature is set to non-zero, it should return the gradient with respect to params of:

        sum_i(log(p(output_i | input_i, params))) + temperature * sum_i(log(p(training_output_i | training_input_i, params)))

        Parameters
        ----------
        outputs:
            The outputs.
        inputs:
            The inputs.

        Returns
        -------
        A version of ``Params`` with all differentiable fields from ``Params`` and values set to gradient for each parameter.
        """

    def forward(self, *, inputs: Inputs) -> Outputs:
        """
        Similar to ``produce`` method but it is meant to be used for a forward pass during
        backpropagation-based end-to-end training. Primitive can implement it differently
        than ``produce``, e.g., forward pass during training can enable dropout layers, or
        ``produce`` might not compute gradients while ``forward`` does.

        By default it calls ``produce`` for one iteration.

        Parameters
        ----------
        inputs:
            The inputs of shape [num_inputs, ...].

        Returns
        -------
        The outputs of shape [num_inputs, ...].
        """

        return self.produce(inputs=inputs, timeout=None, iterations=1).value  # type: ignore

    @abc.abstractmethod
    def backward(self, *, gradient_outputs: Gradients[Outputs], fine_tune: bool = False, fine_tune_learning_rate: float = 0.00001,
                 fine_tune_weight_decay: float = 0.00001) -> typing.Tuple[Gradients[Inputs], Gradients[Params]]:
        """
        Returns the gradient with respect to inputs and with respect to params of a loss
        that is being backpropagated end-to-end in a pipeline.

        This is the standard backpropagation algorithm: backpropagation needs to be preceded by a
        forward propagation (``forward`` method call).

        Parameters
        ----------
        gradient_outputs:
            The gradient of the loss with respect to this primitive's output. During backpropagation,
            this comes from the next primitive in the pipeline, i.e., the primitive whose input
            is the output of this primitive during the forward execution with ``forward`` (and ``produce``).
        fine_tune:
            If ``True``, executes a fine-tuning gradient descent step as a part of this call.
            This provides the most straightforward way of end-to-end training/fine-tuning.
        fine_tune_learning_rate:
            Learning rate for end-to-end training/fine-tuning gradient descent steps.
        fine_tune_weight_decay:
            L2 regularization (weight decay) coefficient for end-to-end training/fine-tuning gradient
            descent steps.

        Returns
        -------
        A tuple of the gradient with respect to inputs and with respect to params.
        """

    @abc.abstractmethod
    def set_fit_term_temperature(self, *, temperature: float = 0) -> None:
        """
        Sets the temperature used in ``gradient_output`` and ``gradient_params``.

        Parameters
        ----------
        temperature:
            The temperature to use, [0, inf), typically, [0, 1].
        """


class LossFunctionMixin(typing.Generic[Inputs, Outputs, Params, Hyperparams], metaclass=utils.GenericMetaclass):
    """
    Mixin which provides abstract methods for a caller to call to inspect which
    loss function or functions a primitive is using internally, and to compute
    loss on given inputs and outputs.
    """

    @abc.abstractmethod
    def get_loss_functions(self) -> typing.Sequence[typing.Tuple[problem.PerformanceMetric, PrimitiveBase, None]]:  # type: ignore
        """
        Returns a list of loss functions used by the primitive. Each element of the list can be:

        * A D3M metric value of the loss function used by the primitive during the last fitting.
        * Primitives can be passed to other primitives as arguments. As such, some primitives
          can accept another primitive as a loss function to use, or use it internally. A primitive
          can expose this loss primitive to others, providing directly an instance of the primitive
          being used during the last fitting.
        * ``None`` if using a non-standard loss function. Used so that the loss function can still
          be exposed through ``loss`` and ``losses`` methods.

        It should return an empty list if the primitive does not use loss functions at all.

        The order in the list matters because the loss function index is used for ``loss`` and ``losses`` methods.

        Returns
        -------
        A list of: a D3M standard metric value of the loss function used,
        or a D3M primitive used to compute loss, or ``None``.
        """

    @abc.abstractmethod
    def losses(self, *, loss_function: int, inputs: Inputs, outputs: Outputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Returns the loss L(output_i, produce_one(input_i)) for each (input_i, output_i) pair
        using a loss function used by the primitive during the last fitting, identified by the
        ``loss_function`` index in the list of loss functions as returned by the ``get_loss_functions``.

        Parameters
        ----------
        loss_function:
            An index of the loss function to use.
        inputs:
            The inputs.
        outputs:
            The outputs.
        timeout:
            A maximum time this primitive should take to produce outputs during this method call, in seconds.
        iterations:
            How many of internal iterations should the primitive do.

        Returns
        -------
        L(output_i, produce_one(input_i)) for each (input_i, output_i) pair
        wrapped inside ``CallResult``.
        The number of columns should match the number of target columns in ``outputs``.
        """

    def loss(self, *, loss_function: int, inputs: Inputs, outputs: Outputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Returns the loss sum_i(L(output_i, produce_one(input_i))) for all (input_i, output_i) pairs
        using a loss function used by the primitive during the last fitting, identified by the
        ``loss_function`` index in the list of loss functions as returned by the ``get_loss_functions``.

        By default it calls ``losses`` and tries to automatically compute a sum, but subclasses can
        implement a more efficient or even correct version.

        Parameters
        ----------
        loss_function:
            An index of the loss function to use.
        inputs:
            The inputs.
        outputs:
            The outputs.
        timeout:
            A maximum time this primitive should take to produce outputs during this method call, in seconds.
        iterations:
            How many of internal iterations should the primitive do.

        Returns
        -------
        sum_i(L(output_i, produce_one(input_i))) for all (input_i, output_i) pairs
        wrapped inside ``CallResult``.
        The number of returned samples is always 1.
        The number of columns should match the number of target columns in ``outputs``.
        """

        result = self.losses(loss_function=loss_function, inputs=inputs, outputs=outputs, timeout=timeout, iterations=iterations)

        return CallResult(utils.columns_sum(result.value), result.has_finished, result.iterations_done)


class NeuralNetworkModuleMixin(typing.Generic[Inputs, Outputs, Params, Hyperparams, Module], metaclass=utils.GenericMetaclass):
    """
    Mixin which provides an abstract method for connecting neural network
    modules together. Mixin is parameterized with type variable ``Module``.
    These modules can be either single layers, or they can be blocks of layers.
    The construction of these modules is done by mapping the neural network
    to the pipeline structure, where primitives (exposing modules through this
    abstract method) are passed to followup layers through hyper-parameters.
    The whole such structure is then passed for the final time as a hyper-parameter
    to a training primitive which then builds the internal representation of the neural
    network and trains it.
    """

    @abc.abstractmethod
    def get_neural_network_module(self, *, input_module: Module) -> Module:
        """
        Returns a neural network module corresponding to this primitive. That module
        might be already connected to other modules, which can be done by
        primitive calling this method recursively on other primitives. If this
        is initial layer of the neural network, it input is provided through
        ``input_module`` argument.

        Parameters
        ----------
        input_module:
            The input module to the initial layer of the neural network.

        Returns
        -------
        The ``Module`` instance corresponding to this primitive.
        """


class NeuralNetworkObjectMixin(typing.Generic[Inputs, Outputs, Params, Hyperparams, Module], metaclass=utils.GenericMetaclass):
    """
    Mixin which provides an abstract method which returns auxiliary objects for use
    in representing neural networks as pipelines: loss functions, optimizers, etc.

    One should consider the use of other primitive metadata (primitive family, algorithm
    types) to describe the primitive implementing this mixin and limit primitives
    in hyper-parameters.
    """

    @abc.abstractmethod
    def get_neural_network_object(self, module: Module) -> typing.Any:
        """
        Returns a neural network object. The object is opaque from the perspective
        of the pipeline. The caller is responsible to assure that the returned
        object is of correct type and interface and that it is passed on to
        a correct consumer understanding the object.

        Parameters
        ----------
        module:
            The module representing the neural network for which the object is requested.
            It should be always provided even if particular implementation does not use it.

        Returns
        -------
        An opaque object.
        """


def singleton(f: typing.Callable) -> typing.Callable:
    """
    If a produce method is using this decorator, it is signaling that all outputs from the produce method are
    sequences of length 1. This is useful because a caller can then directly extract this element.

    Example of such produce methods are produce methods of primitives which compute loss, which are returning
    one number for multiple inputs. With this decorator they can return a sequence with this one number, but
    caller which cares about the loss can extract it out. At the same time, other callers which operate
    only on sequences can continue to operate normally.

    We can see other produce methods as mapping produce methods, and produce methods with this decorator as
    reducing produce methods.
    """

    # Mark a produce method as a singleton. This is our custom flag.
    f.__singleton__ = True  # type: ignore

    return f


def inputs_across_samples(func: typing.Callable = None, inputs: typing.Sequence[str] = None, *args: str) -> typing.Callable:
    """
    A produce method can use this decorator to signal which of the inputs (arguments) is using across
    all samples and not sample by sample.

    For many produce methods it does not matter if it is called 100x on 1 sample or 1x on 100 samples,
    but not all produce methods are like that and some produce results based on which all inputs were
    given to them. If just a subset of inputs is given, results are different. An example of this is
    ``produce_distance_matrix`` method which returns a NxN matrix where N is number of samples, computing
    a distance from each sample to each other sample.

    When inputs have a primary key without uniqueness constraint, then "sample" for the purpose of
    this decorator means all samples with the same primary key value.

    Decorator accepts a list of inputs which are used across all samples. By default, `inputs`
    argument name is used.
    """

    if callable(func):
        if inputs is None:
            inputs = ('inputs',)

        # Make sure values are unique and sorted.
        inputs = tuple(sorted(set(inputs)))

        # List inputs which a produce method computes across samples. This is our custom flag.
        # That listed names are really argument names is checked during metadata generation.
        func.__inputs_across_samples__ = inputs  # type: ignore

        return func

    else:
        def decorator(f):
            # We do not have to call "functool.update_wrapper" or something similar
            # because we are in fact returning the same function "f", just with
            # set "__inputs_across_samples__" attribute
            return inputs_across_samples(f, [s for s in [func, inputs] + list(args) if isinstance(s, str)])

        return decorator


# We register additional immutable types. We are doing it this way to overcome issues with import cycles.
# This is a tricky one. Primitive instances are generally mutable, they can change state when they are used.
# But as part of hyper-parameters, they can be used as instances and are seen as immutable because the idea
# is that TA2 will make a copy of the primitive before passing it in as a hyper-parameter, leaving initial
# instance intact.
if PrimitiveBase not in utils.additional_immutable_types:
    utils.additional_immutable_types += (PrimitiveBase,)
