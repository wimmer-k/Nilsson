#https://stackoverflow.com/questions/36684319/decorator-for-a-class-method-that-caches-return-value-after-first-access
#https://stackoverflow.com/questions/4037481/caching-class-attributes-in-python
class Memoize(object):
    def __init__(self, func):
        self.fname = func.__name__
        self.f = func
        
    def __get__(self, instance, owner):
        # Build the attribute.
        attr = self.f(instance)

        # Cache the value; hide ourselves.
        setattr(instance, self.fname, attr)

        return attr


### DOES NOT WORK with classes
#https://www.python-course.eu/python3_memoization.php
#class Memoize:
#    def __init__(self, fn):
#        self.fn = fn
#        self.memo = {}
#
#    def __call__(self, *args):
#        if args not in self.memo:
#            self.memo[args] = self.fn(*args)
#        return self.memo[args]
