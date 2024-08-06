
import warnings
from functools import lru_cache, wraps

def deprecated(func):
    def wrapper(*args, **kwargs):
        warnings.warn(f"\033[91mCall to deprecated function '{func.__name__}'\033[0m", category=DeprecationWarning, stacklevel=2)
        return func(*args, **kwargs)
    return wrapper

def not_implemented(func):
    def wrapper(*args, **kwargs):
        warnings.warn(f"\033[91mCall to not implemented function '{func.__name__}'\033[0m", category=DeprecationWarning, stacklevel=2)
        return func(*args, **kwargs)
    return wrapper

def lru_cache_with_doc(maxsize=128, typed=False):
    def decorator(func):
        cached_func = lru_cache(maxsize=maxsize, typed=typed)(func)
        @wraps(func)
        def wrapper(*args, **kwargs):
            return cached_func(*args, **kwargs)
        return wrapper
    return decorator

# Set warnings to raise an error by default
warnings.simplefilter('error', DeprecationWarning)
