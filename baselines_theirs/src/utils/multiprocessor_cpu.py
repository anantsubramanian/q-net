import numpy as np
import collections
import multiprocessing as mp


class MultiProcessorCPU(object):
    '''
    manage parallel computing using multiprocessing lib
    with multipe CPUs. 
    The procs list should be constructed using the following way:

    p = mp.Process(target=func, args=(...) )
    procs.append(p)

    The func should use returnDict as argument to 
    get the return value. the return value should be indexed
    by some id. e.g. the title of an article.
    '''
    def __init__(self, procs, nLeaveOut=2):
        '''
        @param nLeaveOut: we leave out nLeaveOut cores to deal with the
        calling main process and other tasks.
        '''
        self.procs = procs
        self.nCores = mp.cpu_count()
        self.nLeaveOut = nLeaveOut

    def run(self, conservative=True):
        curProcs = []
        doneProcs = []
        if conservative:
            while len(self.procs) != 0 or len(curProcs) != 0:
                # preserve one core for the calling process
                while len(curProcs) < self.nCores - self.nLeaveOut\
                    and len(self.procs) != 0:
                    p = self.procs.pop()
                    curProcs.append(p)
                    curProcs[-1].start()
                i = 0
                while i < len(curProcs):
                    if curProcs[i].exitcode is None:
                        i += 1
                    elif curProcs[i].exitcode < 0:
                        assert 0
                    else:
                        doneProcs.append(curProcs[i] )
                        del curProcs[i]
        else:
            for proc in self.procs:
                proc.start()
            doneProcs = self.procs
        for proc in doneProcs:
            proc.join()
        for proc in doneProcs:
            proc.terminate()
