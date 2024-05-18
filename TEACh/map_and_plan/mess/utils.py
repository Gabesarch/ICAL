import cv2
import numpy as np


class Foo(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        str_ = ''
        for v in vars(self).keys():
            a = getattr(self, v)
            if True:  # isinstance(v, object):
                str__ = str(a)
                str__ = str__.replace('\n', '\n  ')
            else:
                str__ = str(a)
            str_ += '{:s}: {:s}'.format(v, str__)
            str_ += '\n'
        return str_