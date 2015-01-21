
from libc.stdio cimport printf

import numpy as np

DEF DEBUG = True


cdef struct SomeObj:
    unsigned int foo

cdef class MyClass:
    cdef unsigned int foo

ctypedef SomeObj obj_t

cdef inline void dbg_print(str loc, str msg) nogil:
    IF DEBUG:
        with gil:
            print('>>> {:s}: {:s}'.format(loc, msg))


cdef inline void dbg_class(MyClass obj, str msg) nogil:
    IF DEBUG:
        with gil:
            print('>>> {obj.foo:d}'.format(obj))

cdef void some_func(obj_t obj, MyClass obj2, str msg) nogil:

    if DEBUG:
        with gil:
            msg = 'i={:.4f}'.format(123.23)
            dbg_print('some_func', msg)


        dbg_print("some_func", "message")
        dbg_class(obj2, 'foo')

def run():

    cdef obj_t obj
    obj.foo = 1
    cdef MyClass obj1 = MyClass()

    msg = "Some message"

    some_func(obj, obj1, msg)