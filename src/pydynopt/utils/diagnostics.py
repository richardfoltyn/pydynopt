__author__ = 'Richard Foltyn'

import time


def print_factory(*flags, t_start=None, cpu_time=False, label=''):
    if cpu_time:
        f_time = time.clock
        time_type = 'CPU time '
    else:
        f_time = time.time
        time_type = ''

    if t_start is None:
        t_start = f_time()

    if label is not None and len(label) > 0:
        label = '[' + label + '] '

    last_tstamp = t_start

    funcs = list()
    for flag in flags:
        if flag:
            def _impl(s):
                nonlocal last_tstamp
                curr_tstamp = f_time()
                print('%s%s (%selapsed: %4.3fs; delta: %4.3fs)' %
                      (label, s, time_type, curr_tstamp - t_start, curr_tstamp -
                      last_tstamp))
                last_tstamp = curr_tstamp
        else:
            def _impl(s):
                pass
        funcs.append(_impl)

    return funcs