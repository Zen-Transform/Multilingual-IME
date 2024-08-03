
import warnings

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

# Set warnings to raise an error by default
warnings.simplefilter('error', DeprecationWarning)
