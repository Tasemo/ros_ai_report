import pickle

def memoize(function):
    values = {}
    def wrapper(x):
        if x not in values:
            values[x] = function(x)
        return values[x]
    return wrapper

@memoize
def persistent_memoize(function):
    def wrapper(*x):
        filename = "cache/" + function.__name__
        with open(filename, "rb+") as file:
            try:
                cache = pickle.load(file)
            except EOFError:
                cache = {}
            key = x[1:]
            if key not in cache:
                cache[key] = function(*x)
                pickle.dump(cache, file)
        return cache[key]
    return wrapper
    