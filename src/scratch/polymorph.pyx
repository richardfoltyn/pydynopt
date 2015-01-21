
cdef class UltraBase:

    def print2(UltraBase self):
        pass

cdef class Base(UltraBase):

    def print1(Base self):
        print('Base, print 1')

cdef class Child1(Base):
    def print1(Child1 self):
        print('Child 1, print 1')

    def print2(Child1 self):
        print('Child1, print2')

cdef class Child2(Base):
    def print1(Child2 self):
        print('Child2, print 1')

    def print2(Child2 self):
        print('Child2, print 2')


def print_test1(Base base):
    base.print1()
    base.print2()