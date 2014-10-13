__author__ = 'Richard Foltyn'

import time


def print_dbg_factory(debug, t_start=None, cpu_time=False, label=''):
    if cpu_time:
        f_time = time.clock
    else:
        f_time = time.time

    if t_start is None:
        t_start = f_time()

    if label is not None and len(label) > 0:
        label += ': '

    last_tstamp = t_start

    if debug:
        def _impl(s):
            nonlocal last_tstamp
            curr_tstamp = f_time()
            print('%s%s (elapsed: %4.3fs; delta: %4.3fs)' %
                  (label, s, curr_tstamp - t_start, curr_tstamp - last_tstamp))
            last_tstamp = curr_tstamp
    else:
        def _impl(s):
            pass
    return _impl