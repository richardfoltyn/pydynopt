"""
Wrap timeit.main() from the Python standard library to have the same
functionality (3 repeats; automatically finding adequate number of runs;
reporting minimum result) as when calling python -m timeit.
"""

__author__ = 'Richard Foltyn'

from timeit import Timer, default_number, default_repeat, default_timer


def repeat(stmt='pass', setup='pass', timer=default_timer,
           repeat=default_repeat, number=0):

    precision = 3

    t = Timer(stmt, setup, timer)
    number = int(number)

    # Copy this directly from the Python standard library timeit.main() function
    if number == 0:
        # determine number so that 0.2 <= total time < 2.0
        for i in range(1, 10):
            number = 10**i
            try:
                x = t.timeit(number)
            except:
                t.print_exc()
                return 1
            if x >= 0.2:
                break
    try:
        r = t.repeat(repeat, number)
    except:
        t.print_exc()
        return 1
    best = min(r)

    print("%d loops," % number, end=' ')
    usec = best * 1e6 / number
    if usec < 1000:
        print("best of %d: %.*g usec per loop" % (repeat, precision, usec))
    else:
        msec = usec / 1000
        if msec < 1000:
            print("best of %d: %.*g msec per loop" % (repeat, precision, msec))
        else:
            sec = msec / 1000
            print("best of %d: %.*g sec per loop" % (repeat, precision, sec))

    # return the list of averages for potential further analysis
    return r
