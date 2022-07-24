__all__ = [
    'deprecated', 'loggingdecorator', 'LoggerWriter'
]

# helpers
import functools
from functools import wraps as _wraps
from io import TextIOBase
import inspect
from typing import Callable, Any
import warnings
import logging

string_types = (type(b''), type(u''))


def deprecated(reason):
    """
    This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.
    """

    if isinstance(reason, string_types):

        # The @deprecated is used with a 'reason'.
        #
        # .. code-block:: python
        #
        #    @deprecated("please, use another function")
        #    def old_function(x, y):
        #      pass

        def decorator(func1):

            if inspect.isclass(func1):
                fmt1 = "Call to deprecated class {name} ({reason})."
            else:
                fmt1 = "Call to deprecated function {name} ({reason})."

            @functools.wraps(func1)
            def new_func1(*args, **kwargs):
                warnings.simplefilter('always', DeprecationWarning)
                warnings.warn(
                    fmt1.format(name=func1.__name__, reason=reason),
                    category=DeprecationWarning,
                    stacklevel=2
                )
                warnings.simplefilter('default', DeprecationWarning)
                return func1(*args, **kwargs)

            return new_func1

        return decorator

    elif inspect.isclass(reason) or inspect.isfunction(reason):

        # The @deprecated is used without any 'reason'.
        #
        # .. code-block:: python
        #
        #    @deprecated
        #    def old_function(x, y):
        #      pass

        func2 = reason

        if inspect.isclass(func2):
            fmt2 = "Call to deprecated class {name}."
        else:
            fmt2 = "Call to deprecated function {name}."

        @functools.wraps(func2)
        def new_func2(*args, **kwargs):
            warnings.simplefilter('always', DeprecationWarning)
            warnings.warn(
                fmt2.format(name=func2.__name__),
                category=DeprecationWarning,
                stacklevel=2
            )
            warnings.simplefilter('default', DeprecationWarning)
            return func2(*args, **kwargs)

        return new_func2

    else:
        raise TypeError(repr(type(reason)))

# A Python decorator to log the function call and return value


class loggingdecorator(object):

    def __init__(self, name: str, level: int = logging.DEBUG,
                 input: bool = False, output: bool = False) -> None:
        self.name = name
        self.level = level
        self.input = input
        self.output = output
        self.logger = logging.getLogger(self.name)

    """A decorator to log the function call and return value (i.e. signature)

    Args:
        name (str): ``name`` in ``logging.GetLogger(name)``
        level (int, optional): logging level, eg. ``INFO``, ``DEBUG``, etc.
            Defaults to ``DEBUG``. See logging.setLevel_ for more info.
        input (bool, optional): whether or not include the input of
            decorated function in logs. Defaults to False.
        output (bool, optional): whether or not include the output of
            decorated function in logs. Defaults to False.

    .. _logging.setLevel: https://docs.python.org/3/library/logging.html#levels

    References:

        * https://machinelearningmastery.com/logging-in-python/


    Returns:
        Callable: A callable decorator
    """

    def __call__(self, fn, *args: Any, **kwds: Any) -> Any:

        @_wraps(fn)
        def _decor(*args, **kwds):
            function_name = fn.__name__

            def _fn(*args, **kwds):
                ret = fn(*args, **kwds)
                if self.input:
                    argstr = [str(x) for x in args]
                    argstr += [key+"="+str(val) for key, val in kwds.items()]
                else:
                    argstr = ''
                ret_str = ret if self.output else ''
                self.logger.debug("%s(%s) -> %s", function_name,
                                  ", ".join(argstr), ret_str)
                return ret
            return _fn
        return _decor(*args, **kwds)


class LoggerWriter(TextIOBase):
    """Writes std out and err to a logger

    References:
        - https://stackoverflow.com/a/66209331/18971263

    """
    def __init__(self, logfct):
        self.logfct = logfct
        self.buf = []

    def write(self, msg):
        if msg.endswith('\n'):
            self.buf.append(msg.rstrip('\n'))
            self.logfct(''.join(self.buf))
            self.buf = []
        else:
            self.buf.append(msg)

    def flush(self):
        pass
