import copy


class ParamContainer(object):

    def __init__(self, d=None):
        if d is not None and isinstance(d, dict):
            self.populate(d)

    def populate(self, d):
        self.__dict__ = copy.deepcopy(d)

    def __deepcopy__(self, memo):
        res = ParamContainer.__new__(ParamContainer)
        for (k, v) in self.__dict__.iteritems():
            if not k.startswith('__'):
                res.__dict__[k] = copy.deepcopy(v, memo)

        return res
