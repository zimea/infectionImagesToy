import functools
import time


def timeit(func):
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print(
            "function [{}] finished in {} ms".format(
                func.__name__, int(elapsed_time * 1_000)
            )
        )
        return result

    return new_func

def injector_model(function, name=None, mod=None):
    if name == None:
        name = function.__name__
    def wrapper(k):
        if mod == None:
            setattr(k, name, eval(name))
        else:
            setattr(k, name, mod(eval(name)))
        return k
    return wrapper

def inject_static_method(function, name):
    return injector_model(function, name, staticmethod)

def inject_class_method(function, name):
    return injector_model(function, name, classmethod)

def inject_instance_method(function, name):
    return injector_model(function, name)