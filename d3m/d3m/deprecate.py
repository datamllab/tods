import functools
import logging
import sys
import typing

logger = logging.getLogger(__name__)


class Context(typing.NamedTuple):
    function: typing.Optional[str]
    argument: typing.Optional[str]
    filename: str
    module: str
    lineno: int


def function(message: str = None) -> typing.Callable:
    """
    A decorator which issues a warning if a wrapped function is called.
    """

    def decorator(f: typing.Callable) -> typing.Callable:
        already_warned: typing.Set[Context] = set()

        @functools.wraps(f)
        def wrapper(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
            frame = sys._getframe(1)
            try:
                while frame:
                    # If function has multiple decorators, skip decorators as callers and find the real caller.
                    if frame.f_code.co_filename != __file__:
                        break

                    frame = frame.f_back

                if not frame:
                    if message is None:
                        logger.warning(
                            "Calling a deprecated function '%(function)s'.",
                            {
                                'function': f.__name__,
                            },
                        )
                    else:
                        logger.warning(
                            "Calling a deprecated function '%(function)s': %(message)s",
                            {
                                'function': f.__name__,
                                'message': message,
                            },
                        )
                    return f(*args, **kwargs)

                context = Context(f.__name__, None, frame.f_code.co_filename, frame.f_globals.get('__name__', None), frame.f_lineno)

            finally:
                del frame

            if context in already_warned:
                return f(*args, **kwargs)
            already_warned.add(context)

            if message is None:
                logger.warning("%(module)s: Calling a deprecated function '%(function)s' in '%(filename)s' at line %(lineno)s.", context._asdict())
            else:
                logger.warning("%(module)s: Calling a deprecated function '%(function)s' in '%(filename)s' at line %(lineno)s: %(message)s", dict(context._asdict(), message=message))

            return f(*args, **kwargs)

        return wrapper

    return decorator


def arguments(*deprecated_arguments: str, message: str = None) -> typing.Callable:
    """
    A decorator which issues a warning if any of the ``deprecated_arguments`` is being
    passed to the wrapped function.
    """

    def decorator(f: typing.Callable) -> typing.Callable:
        already_warned: typing.Set[Context] = set()

        @functools.wraps(f)
        def wrapper(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
            for argument in deprecated_arguments:
                if argument in kwargs:
                    frame = sys._getframe(1)
                    try:
                        while frame:
                            # If function has multiple decorators, skip decorators as callers and find the real caller.
                            if frame.f_code.co_filename != __file__:
                                break

                            frame = frame.f_back

                        if not frame:
                            if message is None:
                                logger.warning(
                                    "Providing a deprecated argument '%(argument)s' to '%(function)s' function.",
                                    {
                                        'argument': argument,
                                        'function': f.__name__,
                                    },
                                )
                            else:
                                logger.warning(
                                    "Providing a deprecated argument '%(argument)s' to '%(function)s' function: %(message)s",
                                    {
                                        'argument': argument,
                                        'function': f.__name__,
                                        'message': message,
                                    },
                                )
                            break

                        context = Context(f.__name__, argument, frame.f_code.co_filename, frame.f_globals.get('__name__', None), frame.f_lineno)

                    finally:
                        del frame

                    if context in already_warned:
                        break
                    already_warned.add(context)

                    if message is None:
                        logger.warning(
                            "%(module)s: Providing a deprecated argument '%(argument)s' to '%(function)s' function in '%(filename)s' at line %(lineno)s.",
                            context._asdict(),
                        )
                    else:
                        logger.warning(
                            "%(module)s: Providing a deprecated argument '%(argument)s' to '%(function)s' function in '%(filename)s' at line %(lineno)s: %(message)s",
                            dict(context._asdict(), message=message),
                        )

                    break

            return f(*args, **kwargs)

        return wrapper

    return decorator
